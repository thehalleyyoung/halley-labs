//! Piecewise linear value function representation.
//!
//! Stores φ(x) as a collection of affine pieces over critical regions,
//! supports evaluation, subdifferential computation, interpolation, and refinement.

use bicut_types::{AffineFunction, Polyhedron};
use serde::{Deserialize, Serialize};

use crate::critical_region::CriticalRegion;
use crate::oracle::ValueFunctionOracle;
use crate::{VFError, VFResult, TOLERANCE};

// ---------------------------------------------------------------------------
// Affine piece
// ---------------------------------------------------------------------------

/// A single affine piece of the piecewise linear value function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffinePiece {
    /// Gradient coefficients a such that piece(x) = a^T x + b.
    pub coefficients: Vec<f64>,
    /// Constant term b.
    pub constant: f64,
    /// The region over which this piece is active (optional).
    pub region: Option<Polyhedron>,
}

impl AffinePiece {
    /// Evaluate this piece at x.
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        self.coefficients
            .iter()
            .zip(x.iter())
            .map(|(a, xi)| a * xi)
            .sum::<f64>()
            + self.constant
    }

    /// Check if x is in this piece's region (returns true if no region is set).
    pub fn contains(&self, x: &[f64]) -> bool {
        match &self.region {
            Some(poly) => poly.contains(x, TOLERANCE),
            None => true,
        }
    }

    /// Dimension of the piece.
    pub fn dim(&self) -> usize {
        self.coefficients.len()
    }

    /// Create a constant piece.
    pub fn constant_piece(dim: usize, value: f64) -> Self {
        Self {
            coefficients: vec![0.0; dim],
            constant: value,
            region: None,
        }
    }

    /// Create a piece from gradient and intercept.
    pub fn from_gradient(gradient: Vec<f64>, constant: f64) -> Self {
        Self {
            coefficients: gradient,
            constant,
            region: None,
        }
    }

    /// Create a piece from an AffineFunction.
    pub fn from_affine_function(af: &AffineFunction) -> Self {
        Self {
            coefficients: af.coefficients.clone(),
            constant: af.constant,
            region: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Subdifferential
// ---------------------------------------------------------------------------

/// Information about the subdifferential at a point.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubdifferentialInfo {
    /// Set of subgradients at the point.
    pub subgradients: Vec<Vec<f64>>,
    /// Whether the function is differentiable at this point.
    pub is_differentiable: bool,
    /// Active piece indices.
    pub active_pieces: Vec<usize>,
}

impl SubdifferentialInfo {
    /// Compute a representative subgradient (average of all).
    pub fn representative_subgradient(&self) -> Vec<f64> {
        if self.subgradients.is_empty() {
            return Vec::new();
        }
        let n = self.subgradients[0].len();
        let count = self.subgradients.len() as f64;
        let mut avg = vec![0.0; n];
        for sg in &self.subgradients {
            for (i, &v) in sg.iter().enumerate() {
                avg[i] += v / count;
            }
        }
        avg
    }

    /// Norm of the steepest subgradient.
    pub fn max_subgradient_norm(&self) -> f64 {
        self.subgradients
            .iter()
            .map(|sg| sg.iter().map(|v| v * v).sum::<f64>().sqrt())
            .fold(0.0f64, f64::max)
    }
}

// ---------------------------------------------------------------------------
// Piecewise linear value function
// ---------------------------------------------------------------------------

/// A piecewise linear (convex) value function representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiecewiseLinearVF {
    /// The affine pieces.
    pub pieces: Vec<AffinePiece>,
    /// Dimension of x-space.
    pub dim: usize,
}

impl PiecewiseLinearVF {
    pub fn new(dim: usize) -> Self {
        Self {
            pieces: Vec::new(),
            dim,
        }
    }

    pub fn add_piece(&mut self, piece: AffinePiece) {
        self.pieces.push(piece);
    }

    pub fn num_pieces(&self) -> usize {
        self.pieces.len()
    }

    /// Evaluate the value function at x as the pointwise maximum of all pieces
    /// (valid for convex piecewise linear functions represented as max of affines).
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        if self.pieces.is_empty() {
            return 0.0;
        }

        // First try to find a piece whose region contains x
        for piece in &self.pieces {
            if let Some(ref poly) = piece.region {
                if poly.contains(x, TOLERANCE) {
                    return piece.evaluate(x);
                }
            }
        }

        // Fall back to pointwise max (for convex representation)
        self.pieces
            .iter()
            .map(|p| p.evaluate(x))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Evaluate the value function at x as pointwise minimum (for concave VF).
    pub fn evaluate_min(&self, x: &[f64]) -> f64 {
        if self.pieces.is_empty() {
            return 0.0;
        }
        self.pieces
            .iter()
            .map(|p| p.evaluate(x))
            .fold(f64::INFINITY, f64::min)
    }

    /// Evaluate using only pieces whose region contains x.
    pub fn evaluate_region_aware(&self, x: &[f64]) -> Option<f64> {
        self.pieces
            .iter()
            .filter(|p| p.contains(x))
            .map(|p| p.evaluate(x))
            .next()
    }

    /// Compute the subdifferential at x.
    pub fn subdifferential(&self, x: &[f64]) -> SubdifferentialInfo {
        let val = self.evaluate(x);
        let mut active_pieces = Vec::new();
        let mut subgradients = Vec::new();

        for (i, piece) in self.pieces.iter().enumerate() {
            let piece_val = piece.evaluate(x);
            if (piece_val - val).abs() < TOLERANCE * 100.0 {
                active_pieces.push(i);
                subgradients.push(piece.coefficients.clone());
            }
        }

        let is_differentiable = active_pieces.len() <= 1;

        SubdifferentialInfo {
            subgradients,
            is_differentiable,
            active_pieces,
        }
    }

    /// Compute error bound against an oracle at sample points.
    pub fn error_bound(
        &self,
        oracle: &dyn ValueFunctionOracle,
        sample_points: &[Vec<f64>],
    ) -> ErrorBoundResult {
        let mut max_error = 0.0f64;
        let mut avg_error = 0.0f64;
        let mut count = 0usize;
        let mut errors = Vec::new();

        for x in sample_points {
            if let Ok(true_val) = oracle.value(x) {
                let approx_val = self.evaluate(x);
                let error = (true_val - approx_val).abs();
                errors.push(error);
                max_error = max_error.max(error);
                avg_error += error;
                count += 1;
            }
        }

        if count > 0 {
            avg_error /= count as f64;
        }

        ErrorBoundResult {
            max_error,
            avg_error,
            num_samples: count,
            per_sample_errors: errors,
        }
    }

    /// Refine the approximation by adding a new piece derived from an oracle evaluation.
    pub fn refine_at(&mut self, oracle: &dyn ValueFunctionOracle, x: &[f64]) -> VFResult<()> {
        let info = oracle.evaluate(x)?;
        let dual = oracle.dual_info(x)?;

        // The new piece is: φ(x₀) + g^T (x - x₀)
        // = g^T x + (φ(x₀) - g^T x₀)
        let constant = info.value
            - dual
                .subgradient
                .iter()
                .zip(x.iter())
                .map(|(g, xi)| g * xi)
                .sum::<f64>();

        let piece = AffinePiece {
            coefficients: dual.subgradient,
            constant,
            region: None,
        };

        self.add_piece(piece);
        Ok(())
    }

    /// Build a piecewise linear function from critical regions.
    pub fn from_critical_regions(regions: &[CriticalRegion]) -> Self {
        let dim = if regions.is_empty() {
            0
        } else {
            regions[0].dim()
        };

        let mut pwl = Self::new(dim);

        for region in regions {
            let piece = AffinePiece {
                coefficients: region.value_function.coefficients.clone(),
                constant: region.value_function.constant,
                region: Some(region.polyhedron.clone()),
            };
            pwl.add_piece(piece);
        }

        pwl
    }

    /// Interpolate between two PWL value functions.
    pub fn interpolate(a: &PiecewiseLinearVF, b: &PiecewiseLinearVF, t: f64) -> Self {
        let dim = a.dim;
        let mut result = Self::new(dim);

        // For each pair of pieces, create an interpolated piece
        for pa in &a.pieces {
            let interp_coeff: Vec<f64> = pa
                .coefficients
                .iter()
                .enumerate()
                .map(|(i, &ca)| {
                    let cb = b
                        .pieces
                        .first()
                        .and_then(|p| p.coefficients.get(i))
                        .copied()
                        .unwrap_or(0.0);
                    (1.0 - t) * ca + t * cb
                })
                .collect();

            let interp_const = {
                let cb = b.pieces.first().map(|p| p.constant).unwrap_or(0.0);
                (1.0 - t) * pa.constant + t * cb
            };

            result.add_piece(AffinePiece {
                coefficients: interp_coeff,
                constant: interp_const,
                region: pa.region.clone(),
            });
        }

        // Add pieces from b that don't overlap
        for pb in &b.pieces {
            let interp_coeff: Vec<f64> = pb
                .coefficients
                .iter()
                .enumerate()
                .map(|(i, &cb)| {
                    let ca = a
                        .pieces
                        .first()
                        .and_then(|p| p.coefficients.get(i))
                        .copied()
                        .unwrap_or(0.0);
                    (1.0 - t) * ca + t * cb
                })
                .collect();

            let interp_const = {
                let ca = a.pieces.first().map(|p| p.constant).unwrap_or(0.0);
                (1.0 - t) * ca + t * pb.constant
            };

            result.add_piece(AffinePiece {
                coefficients: interp_coeff,
                constant: interp_const,
                region: pb.region.clone(),
            });
        }

        result
    }

    /// Remove redundant pieces that are always dominated by another piece.
    pub fn remove_redundant_pieces(&mut self, sample_points: &[Vec<f64>]) {
        if self.pieces.len() <= 1 || sample_points.is_empty() {
            return;
        }

        let num_pieces = self.pieces.len();
        let mut is_active = vec![false; num_pieces];

        for x in sample_points {
            let val = self.evaluate(x);
            for (i, piece) in self.pieces.iter().enumerate() {
                if (piece.evaluate(x) - val).abs() < TOLERANCE * 100.0 {
                    is_active[i] = true;
                }
            }
        }

        let mut new_pieces = Vec::new();
        for (i, piece) in self.pieces.drain(..).enumerate() {
            if is_active[i] {
                new_pieces.push(piece);
            }
        }

        if new_pieces.is_empty() {
            // Keep at least one piece
            // (shouldn't happen if sample_points is non-empty)
        } else {
            self.pieces = new_pieces;
        }
    }

    /// Compute L-infinity error bound over a box.
    pub fn linf_error_bound(
        &self,
        oracle: &dyn ValueFunctionOracle,
        x_lower: &[f64],
        x_upper: &[f64],
        grid_size: usize,
    ) -> VFResult<f64> {
        let nx = self.dim;
        let mut max_error = 0.0f64;

        let total = grid_size.pow(nx.min(5) as u32).min(10000);
        let mut idx = 0usize;

        for trial in 0..total {
            let mut t = trial;
            let x: Vec<f64> = (0..nx)
                .map(|d| {
                    let k = t % grid_size;
                    t /= grid_size;
                    let frac = (k as f64 + 0.5) / grid_size as f64;
                    x_lower[d] + frac * (x_upper[d] - x_lower[d])
                })
                .collect();

            if let Ok(true_val) = oracle.value(&x) {
                let approx_val = self.evaluate(&x);
                let error = (true_val - approx_val).abs();
                max_error = max_error.max(error);
            }
        }

        Ok(max_error)
    }
}

/// Result of an error bound computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBoundResult {
    pub max_error: f64,
    pub avg_error: f64,
    pub num_samples: usize,
    pub per_sample_errors: Vec<f64>,
}

impl ErrorBoundResult {
    /// Compute the p-th percentile of errors.
    pub fn percentile(&self, p: f64) -> f64 {
        if self.per_sample_errors.is_empty() {
            return 0.0;
        }
        let mut sorted = self.per_sample_errors.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Fraction of samples with error below threshold.
    pub fn fraction_below(&self, threshold: f64) -> f64 {
        if self.per_sample_errors.is_empty() {
            return 1.0;
        }
        let below = self
            .per_sample_errors
            .iter()
            .filter(|&&e| e <= threshold)
            .count();
        below as f64 / self.per_sample_errors.len() as f64
    }
}

/// Build a PWL lower bound by collecting cutting planes from oracle evaluations.
pub fn build_cutting_plane_approximation(
    oracle: &dyn ValueFunctionOracle,
    sample_points: &[Vec<f64>],
    dim: usize,
) -> VFResult<PiecewiseLinearVF> {
    let mut pwl = PiecewiseLinearVF::new(dim);

    for x in sample_points {
        if let Ok(info) = oracle.evaluate(x) {
            if let Ok(dual) = oracle.dual_info(x) {
                let constant = info.value
                    - dual
                        .subgradient
                        .iter()
                        .zip(x.iter())
                        .map(|(g, xi)| g * xi)
                        .sum::<f64>();

                pwl.add_piece(AffinePiece {
                    coefficients: dual.subgradient,
                    constant,
                    region: None,
                });
            }
        }
    }

    Ok(pwl)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_piece_evaluate() {
        let piece = AffinePiece {
            coefficients: vec![2.0, 3.0],
            constant: 1.0,
            region: None,
        };
        assert!((piece.evaluate(&[1.0, 1.0]) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_constant_piece() {
        let piece = AffinePiece::constant_piece(3, 5.0);
        assert!((piece.evaluate(&[1.0, 2.0, 3.0]) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_pwl_evaluate() {
        let mut pwl = PiecewiseLinearVF::new(1);
        pwl.add_piece(AffinePiece::from_gradient(vec![1.0], 0.0));
        pwl.add_piece(AffinePiece::from_gradient(vec![-1.0], 0.0));

        // max(x, -x) = |x|
        assert!((pwl.evaluate(&[2.0]) - 2.0).abs() < 1e-12);
        assert!((pwl.evaluate(&[-3.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_subdifferential() {
        let mut pwl = PiecewiseLinearVF::new(1);
        pwl.add_piece(AffinePiece::from_gradient(vec![1.0], 0.0));
        pwl.add_piece(AffinePiece::from_gradient(vec![-1.0], 0.0));

        let subdiff = pwl.subdifferential(&[0.0]);
        assert_eq!(subdiff.active_pieces.len(), 2);
        assert!(!subdiff.is_differentiable);
    }

    #[test]
    fn test_subdifferential_differentiable() {
        let mut pwl = PiecewiseLinearVF::new(1);
        pwl.add_piece(AffinePiece::from_gradient(vec![1.0], 0.0));
        pwl.add_piece(AffinePiece::from_gradient(vec![-1.0], 0.0));

        let subdiff = pwl.subdifferential(&[2.0]);
        assert_eq!(subdiff.active_pieces.len(), 1);
        assert!(subdiff.is_differentiable);
    }

    #[test]
    fn test_interpolate() {
        let mut a = PiecewiseLinearVF::new(1);
        a.add_piece(AffinePiece::from_gradient(vec![1.0], 0.0));

        let mut b = PiecewiseLinearVF::new(1);
        b.add_piece(AffinePiece::from_gradient(vec![2.0], 0.0));

        let interp = PiecewiseLinearVF::interpolate(&a, &b, 0.5);
        // Should have pieces with coefficient 1.5
        let val = interp.evaluate(&[1.0]);
        assert!((val - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_error_bound_result_percentile() {
        let result = ErrorBoundResult {
            max_error: 1.0,
            avg_error: 0.5,
            num_samples: 5,
            per_sample_errors: vec![0.1, 0.3, 0.5, 0.7, 1.0],
        };
        let p50 = result.percentile(50.0);
        assert!((p50 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fraction_below() {
        let result = ErrorBoundResult {
            max_error: 1.0,
            avg_error: 0.5,
            num_samples: 5,
            per_sample_errors: vec![0.1, 0.3, 0.5, 0.7, 1.0],
        };
        let frac = result.fraction_below(0.5);
        assert!((frac - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_from_critical_regions() {
        let region = CriticalRegion {
            polyhedron: Polyhedron::new(1),
            optimal_basis: vec![0],
            affine_solution: vec![],
            value_function: AffineFunction {
                coefficients: vec![2.0],
                constant: 1.0,
            },
            region_id: 0,
            is_bounded: true,
        };
        let pwl = PiecewiseLinearVF::from_critical_regions(&[region]);
        assert_eq!(pwl.num_pieces(), 1);
        assert!((pwl.evaluate(&[1.0]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_remove_redundant() {
        let mut pwl = PiecewiseLinearVF::new(1);
        pwl.add_piece(AffinePiece::from_gradient(vec![1.0], 0.0));
        pwl.add_piece(AffinePiece::from_gradient(vec![-1.0], 0.0));
        // This piece is always dominated
        pwl.add_piece(AffinePiece::from_gradient(vec![0.0], -10.0));

        let samples = vec![vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]];
        pwl.remove_redundant_pieces(&samples);
        assert!(pwl.num_pieces() <= 2);
    }
}
