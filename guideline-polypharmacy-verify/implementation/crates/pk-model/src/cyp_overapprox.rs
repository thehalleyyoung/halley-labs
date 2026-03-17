//! CYP-aware over-approximation for non-Metzler PK systems.
//!
//! When CYP enzyme interactions (mechanism-based inactivation, auto-induction)
//! introduce negative off-diagonal entries in the system Jacobian, the Metzler
//! property is violated and the monotonicity-based decidability guarantee
//! (Theorem 3.2) no longer applies directly.
//!
//! This module decomposes A = M + P where M is the Metzler part and P is the
//! non-Metzler perturbation, then computes a sound interval over-approximation
//! of the reachable set using the matrix measure bound:
//!
//!   ||e^{At} - e^{Mt}|| ≤ t · ||P|| · e^{μ(A)·t}
//!
//! The result is wider safety intervals but a preserved soundness guarantee.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use super::metzler::MetzlerMatrix;
use guardpharma_types::error::PkModelError;

// ---------------------------------------------------------------------------
// MetzlerChecker
// ---------------------------------------------------------------------------

/// Diagnoses whether a PK system matrix satisfies the Metzler property
/// and identifies the specific entries that violate it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetzlerViolation {
    pub row: usize,
    pub col: usize,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetzlerCheckResult {
    pub is_metzler: bool,
    pub violations: Vec<MetzlerViolation>,
    /// Frobenius norm of the non-Metzler perturbation.
    pub perturbation_norm: f64,
}

/// Check if a raw system matrix satisfies Metzler structure and report
/// all off-diagonal entries that violate it.
pub fn check_metzler(matrix: &DMatrix<f64>) -> MetzlerCheckResult {
    let n = matrix.nrows();
    assert_eq!(n, matrix.ncols(), "Matrix must be square");

    let mut violations = Vec::new();
    let mut p_frob_sq = 0.0_f64;

    for i in 0..n {
        for j in 0..n {
            if i != j && matrix[(i, j)] < -1e-15 {
                violations.push(MetzlerViolation {
                    row: i,
                    col: j,
                    value: matrix[(i, j)],
                });
                p_frob_sq += matrix[(i, j)] * matrix[(i, j)];
            }
        }
    }

    MetzlerCheckResult {
        is_metzler: violations.is_empty(),
        violations,
        perturbation_norm: p_frob_sq.sqrt(),
    }
}

// ---------------------------------------------------------------------------
// Metzler-Perturbation Decomposition
// ---------------------------------------------------------------------------

/// Decomposition A = M + P where M is Metzler and P holds the negative
/// off-diagonal entries.
#[derive(Debug, Clone)]
pub struct MetzlerDecomposition {
    /// Metzler part: negative off-diagonals zeroed out.
    pub metzler_part: DMatrix<f64>,
    /// Perturbation: only the negative off-diagonal entries.
    pub perturbation: DMatrix<f64>,
    /// ||P||_∞  (induced infinity norm of perturbation).
    pub perturbation_norm_inf: f64,
    /// ||P||_F  (Frobenius norm of perturbation).
    pub perturbation_norm_frob: f64,
    pub dim: usize,
}

/// Decompose a system matrix A = M + P.
pub fn decompose_metzler(a: &DMatrix<f64>) -> MetzlerDecomposition {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "Matrix must be square");

    let mut m = a.clone();
    let mut p = DMatrix::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            if i != j && a[(i, j)] < -1e-15 {
                p[(i, j)] = a[(i, j)];
                m[(i, j)] = 0.0;
            }
        }
    }

    let norm_inf = matrix_inf_norm(&p);
    let norm_frob = p.iter().map(|v| v * v).sum::<f64>().sqrt();

    MetzlerDecomposition {
        metzler_part: m,
        perturbation: p,
        perturbation_norm_inf: norm_inf,
        perturbation_norm_frob: norm_frob,
        dim: n,
    }
}

// ---------------------------------------------------------------------------
// CypOverApproximator
// ---------------------------------------------------------------------------

/// Over-approximation result for a single time point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverApproxResult {
    /// Lower bound on concentration vector.
    pub lo: Vec<f64>,
    /// Upper bound on concentration vector.
    pub hi: Vec<f64>,
    /// Tightness ratio: Metzler-only interval width / CYP-aware interval width.
    /// 1.0 = no widening (pure Metzler); closer to 0 = large widening.
    pub tightness_ratio: f64,
    /// Absolute widening added per component (in concentration units).
    pub widening_per_component: Vec<f64>,
}

/// Sound over-approximation of e^{At}·c₀ when A is non-Metzler due to CYP
/// interactions.
///
/// Uses the bound:
///   e^{At}·c₀ ∈ [e^{Mt}·c₀ - δ(t), e^{Mt}·c₀ + δ(t)]
/// where
///   δ(t) = ||c₀|| · t · ε · e^{μ(M)·t}
/// with ε = ||P||_∞ and μ(M) = matrix measure of M.
pub struct CypOverApproximator {
    decomposition: MetzlerDecomposition,
    metzler_matrix: MetzlerMatrix,
    /// Matrix measure μ_∞(M) = max_i (m_{ii} + Σ_{j≠i} |m_{ij}|).
    matrix_measure: f64,
}

impl CypOverApproximator {
    /// Build from a (possibly non-Metzler) system matrix.
    pub fn new(system_matrix: &DMatrix<f64>) -> Result<Self, PkModelError> {
        let decomp = decompose_metzler(system_matrix);

        // Wrap the Metzler part.
        let n = decomp.dim;
        let m_data: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| decomp.metzler_part[(i, j)]).collect())
            .collect();
        let metzler_matrix = MetzlerMatrix::new(m_data)?;

        // μ_∞(M) = max_i { m_{ii} + Σ_{j≠i} |m_{ij}| }
        let matrix_measure = (0..n)
            .map(|i| {
                let mut row_sum = decomp.metzler_part[(i, i)];
                for j in 0..n {
                    if j != i {
                        row_sum += decomp.metzler_part[(i, j)].abs();
                    }
                }
                row_sum
            })
            .fold(f64::NEG_INFINITY, f64::max);

        Ok(Self {
            decomposition: decomp,
            metzler_matrix,
            matrix_measure,
        })
    }

    /// Is the original system already Metzler (no perturbation)?
    pub fn is_pure_metzler(&self) -> bool {
        self.decomposition.perturbation_norm_inf < 1e-15
    }

    /// Perturbation norm ε = ||P||_∞.
    pub fn perturbation_epsilon(&self) -> f64 {
        self.decomposition.perturbation_norm_inf
    }

    /// Access the Metzler part for callers that need it.
    pub fn metzler_part(&self) -> &MetzlerMatrix {
        &self.metzler_matrix
    }

    /// Compute the sound over-approximation of x(t) = e^{At}·c₀.
    ///
    /// Returns interval [lo, hi] such that the true trajectory is guaranteed
    /// to lie within, even when A is non-Metzler.
    pub fn overapproximate(
        &self,
        c0: &DVector<f64>,
        t: f64,
    ) -> OverApproxResult {
        let n = self.decomposition.dim;
        let eps = self.decomposition.perturbation_norm_inf;

        // Metzler-only trajectory: e^{Mt}·c₀
        let exp_mt = self.metzler_matrix.matrix_exponential(t);
        let x_metzler = &exp_mt * c0;

        if eps < 1e-15 {
            // Pure Metzler: no widening needed.
            let lo: Vec<f64> = (0..n).map(|i| x_metzler[i]).collect();
            let hi = lo.clone();
            return OverApproxResult {
                lo,
                hi,
                tightness_ratio: 1.0,
                widening_per_component: vec![0.0; n],
            };
        }

        // Perturbation bound: δ(t) = ||c₀|| · t · ε · e^{μ(M)·t}
        let c0_norm: f64 = c0.iter().map(|v| v.abs()).sum(); // L1 norm
        let growth = if self.matrix_measure * t > 500.0 {
            f64::MAX // overflow guard
        } else {
            (self.matrix_measure * t).exp()
        };
        let delta = c0_norm * t * eps * growth;

        let mut lo = vec![0.0; n];
        let mut hi = vec![0.0; n];
        let mut widening = vec![0.0; n];
        let mut metzler_width_sum = 0.0_f64;
        let mut total_width_sum = 0.0_f64;

        for i in 0..n {
            // Concentrations are non-negative; clamp lower bound to 0.
            lo[i] = (x_metzler[i] - delta).max(0.0);
            hi[i] = x_metzler[i] + delta;
            widening[i] = delta;

            let metzler_w = 0.0_f64; // point estimate from Metzler part
            let total_w = hi[i] - lo[i];
            metzler_width_sum += metzler_w;
            total_width_sum += total_w;
        }

        // Tightness ratio: 0/total → 0 when purely widened from a point;
        // we define it as 1 - (widening / total_interval) averaged.
        let tightness_ratio = if total_width_sum > 1e-30 {
            (1.0 - (2.0 * delta * n as f64) / total_width_sum).max(0.0)
        } else {
            1.0
        };

        OverApproxResult {
            lo,
            hi,
            tightness_ratio,
            widening_per_component: widening,
        }
    }

    /// Over-approximate an interval initial condition [c0_lo, c0_hi].
    pub fn overapproximate_interval(
        &self,
        c0_lo: &DVector<f64>,
        c0_hi: &DVector<f64>,
        t: f64,
    ) -> OverApproxResult {
        let n = self.decomposition.dim;
        let eps = self.decomposition.perturbation_norm_inf;

        let exp_mt = self.metzler_matrix.matrix_exponential(t);

        // Metzler monotonicity: e^{Mt} has non-negative entries, so
        // e^{Mt}·c0_lo ≤ e^{Mt}·c0 ≤ e^{Mt}·c0_hi componentwise.
        let x_lo_metzler = &exp_mt * c0_lo;
        let x_hi_metzler = &exp_mt * c0_hi;

        if eps < 1e-15 {
            let lo: Vec<f64> = (0..n).map(|i| x_lo_metzler[i]).collect();
            let hi: Vec<f64> = (0..n).map(|i| x_hi_metzler[i]).collect();
            let metzler_width: Vec<f64> = (0..n).map(|i| hi[i] - lo[i]).collect();
            return OverApproxResult {
                lo,
                hi: hi.clone(),
                tightness_ratio: 1.0,
                widening_per_component: metzler_width,
            };
        }

        // Worst-case c₀ norm for the bound
        let c0_norm: f64 = (0..n)
            .map(|i| c0_lo[i].abs().max(c0_hi[i].abs()))
            .sum();
        let growth = if self.matrix_measure * t > 500.0 {
            f64::MAX
        } else {
            (self.matrix_measure * t).exp()
        };
        let delta = c0_norm * t * eps * growth;

        let mut lo = vec![0.0; n];
        let mut hi = vec![0.0; n];
        let mut widening = vec![0.0; n];
        let mut metzler_width_sum = 0.0_f64;
        let mut total_width_sum = 0.0_f64;

        for i in 0..n {
            lo[i] = (x_lo_metzler[i] - delta).max(0.0);
            hi[i] = x_hi_metzler[i] + delta;
            widening[i] = delta;

            let metzler_w = x_hi_metzler[i] - x_lo_metzler[i];
            let total_w = hi[i] - lo[i];
            metzler_width_sum += metzler_w;
            total_width_sum += total_w;
        }

        let tightness_ratio = if total_width_sum > 1e-30 {
            (metzler_width_sum / total_width_sum).clamp(0.0, 1.0)
        } else {
            1.0
        };

        OverApproxResult {
            lo,
            hi,
            tightness_ratio,
            widening_per_component: widening,
        }
    }

    /// Return a diagnostic report suitable for logging or user display.
    pub fn confidence_report(&self, c0: &DVector<f64>, t: f64) -> ConfidenceReport {
        let result = self.overapproximate(c0, t);
        let n = self.decomposition.dim;

        ConfidenceReport {
            is_pure_metzler: self.is_pure_metzler(),
            num_violations: self.decomposition.perturbation.iter()
                .filter(|v| **v < -1e-15)
                .count(),
            perturbation_norm: self.decomposition.perturbation_norm_inf,
            matrix_measure: self.matrix_measure,
            time_horizon: t,
            tightness_ratio: result.tightness_ratio,
            max_widening: result.widening_per_component
                .iter()
                .cloned()
                .fold(0.0_f64, f64::max),
            interval_lo: result.lo,
            interval_hi: result.hi,
        }
    }
}

// ---------------------------------------------------------------------------
// ConfidenceReport
// ---------------------------------------------------------------------------

/// Quantifies how much the CYP over-approximation widens bounds relative
/// to the pure-Metzler case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceReport {
    pub is_pure_metzler: bool,
    /// Number of negative off-diagonal entries that violate Metzler structure.
    pub num_violations: usize,
    /// ||P||_∞ of the non-Metzler perturbation.
    pub perturbation_norm: f64,
    /// Matrix measure μ_∞(M) of the Metzler part.
    pub matrix_measure: f64,
    /// Time horizon used for the bound.
    pub time_horizon: f64,
    /// Tightness ratio ∈ [0,1]; 1.0 = no widening.
    pub tightness_ratio: f64,
    /// Maximum widening (concentration units) across all components.
    pub max_widening: f64,
    /// Interval lower bounds.
    pub interval_lo: Vec<f64>,
    /// Interval upper bounds.
    pub interval_hi: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Induced infinity norm of a matrix: max row sum of absolute values.
fn matrix_inf_norm(m: &DMatrix<f64>) -> f64 {
    let mut max_sum = 0.0_f64;
    for i in 0..m.nrows() {
        let row_sum: f64 = (0..m.ncols()).map(|j| m[(i, j)].abs()).sum();
        max_sum = max_sum.max(row_sum);
    }
    max_sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    /// Pure Metzler system: no widening should occur.
    #[test]
    fn test_pure_metzler_no_widening() {
        let a = DMatrix::from_row_slice(2, 2, &[-2.0, 1.0, 1.0, -3.0]);
        let check = check_metzler(&a);
        assert!(check.is_metzler);
        assert!(check.violations.is_empty());
        assert!(check.perturbation_norm < 1e-15);

        let approx = CypOverApproximator::new(&a).unwrap();
        assert!(approx.is_pure_metzler());
        let c0 = DVector::from_vec(vec![10.0, 5.0]);
        let result = approx.overapproximate(&c0, 1.0);
        assert!((result.tightness_ratio - 1.0).abs() < 1e-10);
        for w in &result.widening_per_component {
            assert!(*w < 1e-15);
        }
    }

    /// Non-Metzler system: CYP inhibition causes negative off-diagonal.
    #[test]
    fn test_non_metzler_widens_bounds() {
        // Drug B inhibits Drug A metabolism → negative a[0][1] entry
        let a = DMatrix::from_row_slice(2, 2, &[-2.0, -0.5, 1.0, -3.0]);
        let check = check_metzler(&a);
        assert!(!check.is_metzler);
        assert_eq!(check.violations.len(), 1);
        assert_eq!(check.violations[0].row, 0);
        assert_eq!(check.violations[0].col, 1);
        assert!((check.violations[0].value - (-0.5)).abs() < 1e-15);

        let approx = CypOverApproximator::new(&a).unwrap();
        assert!(!approx.is_pure_metzler());
        assert!((approx.perturbation_epsilon() - 0.5).abs() < 1e-10);

        let c0 = DVector::from_vec(vec![10.0, 5.0]);
        let result = approx.overapproximate(&c0, 1.0);
        // Tightness < 1 means bounds were widened
        assert!(result.tightness_ratio < 1.0);
        for w in &result.widening_per_component {
            assert!(*w > 0.0);
        }
        // lo ≤ hi for all components
        for i in 0..2 {
            assert!(result.lo[i] <= result.hi[i]);
            assert!(result.lo[i] >= 0.0); // concentrations non-negative
        }
    }

    /// Decomposition roundtrip: M + P == A.
    #[test]
    fn test_decomposition_roundtrip() {
        let a = DMatrix::from_row_slice(3, 3, &[
            -2.0, -0.3,  0.5,
             0.4, -3.0, -0.1,
             0.0,  0.6, -1.5,
        ]);
        let decomp = decompose_metzler(&a);
        let reconstructed = &decomp.metzler_part + &decomp.perturbation;
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (a[(i, j)] - reconstructed[(i, j)]).abs() < 1e-14,
                    "Mismatch at ({},{})", i, j,
                );
            }
        }
    }

    /// Metzler part has no negative off-diagonals.
    #[test]
    fn test_metzler_part_is_valid() {
        let a = DMatrix::from_row_slice(2, 2, &[-2.0, -0.5, -0.3, -3.0]);
        let decomp = decompose_metzler(&a);
        let check = check_metzler(&decomp.metzler_part);
        assert!(check.is_metzler);
    }

    /// Interval over-approximation is wider than point over-approximation.
    #[test]
    fn test_interval_overapprox_contains_point() {
        let a = DMatrix::from_row_slice(2, 2, &[-2.0, -0.4, 0.5, -3.0]);
        let approx = CypOverApproximator::new(&a).unwrap();

        let c0 = DVector::from_vec(vec![8.0, 4.0]);
        let c0_lo = DVector::from_vec(vec![7.0, 3.0]);
        let c0_hi = DVector::from_vec(vec![9.0, 5.0]);
        let t = 2.0;

        let point_res = approx.overapproximate(&c0, t);
        let interval_res = approx.overapproximate_interval(&c0_lo, &c0_hi, t);

        for i in 0..2 {
            assert!(interval_res.lo[i] <= point_res.lo[i] + 1e-10);
            assert!(interval_res.hi[i] >= point_res.hi[i] - 1e-10);
        }
    }

    /// Confidence report is consistent.
    #[test]
    fn test_confidence_report() {
        let a = DMatrix::from_row_slice(2, 2, &[-2.0, -0.5, 1.0, -3.0]);
        let approx = CypOverApproximator::new(&a).unwrap();
        let c0 = DVector::from_vec(vec![10.0, 5.0]);
        let report = approx.confidence_report(&c0, 1.0);

        assert!(!report.is_pure_metzler);
        assert_eq!(report.num_violations, 1);
        assert!(report.perturbation_norm > 0.0);
        assert!(report.tightness_ratio < 1.0);
        assert!(report.max_widening > 0.0);
        assert_eq!(report.interval_lo.len(), 2);
        assert_eq!(report.interval_hi.len(), 2);
    }
}
