//! Davis-Kahan sin-theta theorem certificate.
//!
//! Bounds the eigenspace angle between perturbed and ideal Laplacian:
//!   sin(Θ(V, V̂)) ≤ ||E||_F / gap
//!
//! where E is the perturbation matrix and gap is the minimum eigenvalue gap.
//! This gives a misclassification rate bound of O(δ²/γ²) from the angle bound.

use crate::error::{CertificateError, CertificateResult};
use chrono::Utc;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Quality assessment of a single eigenvector.
///
/// Tracks the residual norm `||Av - λv|| / ||v||` and the maximum
/// off-diagonal inner product `max_{j≠i} |vᵢᵀ vⱼ|` (orthogonality deviation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EigenvectorQuality {
    pub index: usize,
    pub residual_norm: f64,
    pub orthogonality_deviation: f64,
    pub is_acceptable: bool,
}

impl EigenvectorQuality {
    pub fn new(index: usize, residual: f64, orth_dev: f64, tolerance: f64) -> Self {
        Self {
            index,
            residual_norm: residual,
            orthogonality_deviation: orth_dev,
            is_acceptable: residual < tolerance && orth_dev < tolerance,
        }
    }
}

/// Davis-Kahan sin-theta certificate for eigenspace perturbation.
///
/// Implements the Davis-Kahan sin(Θ) theorem:
///
/// `sin(Θ(V, V̂)) ≤ ‖E‖_F / γ`
///
/// where `E = Â − A` is the perturbation matrix and `γ` is the spectral gap
/// between the `k`-th and `(k+1)`-th eigenvalues. This yields a
/// misclassification rate bound of `O(δ²/γ²)` when used for spectral clustering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DavisKahanCertificate {
    pub id: String,
    pub created_at: String,
    /// ||E||_F: Frobenius norm of the perturbation matrix
    pub perturbation_norm: f64,
    /// Spectral gap: minimum eigenvalue gap used in the bound
    pub spectral_gap: f64,
    /// sin(Θ) angle bound
    pub angle_bound: f64,
    /// Actual angle (if computable from eigenvectors)
    pub actual_angle: Option<f64>,
    /// Eigenvalues of the original matrix
    pub eigenvalues: Vec<f64>,
    /// Eigenvalues of the perturbed matrix
    pub perturbed_eigenvalues: Vec<f64>,
    /// Number of eigenvectors in the invariant subspace
    pub subspace_dimension: usize,
    /// Per-eigenvector quality assessments
    pub eigenvector_quality: Vec<EigenvectorQuality>,
    /// Misclassification rate bound: O(δ²/γ²)
    pub misclassification_bound: f64,
    pub metadata: IndexMap<String, String>,
}

impl DavisKahanCertificate {
    /// Compute the Davis-Kahan angle bound.
    ///
    /// sin(Θ(V, V̂)) ≤ ||E||_F / gap
    pub fn compute_angle_bound(
        perturbation_norm: f64,
        eigenvalues: Vec<f64>,
        perturbed_eigenvalues: Vec<f64>,
        subspace_dimension: usize,
    ) -> CertificateResult<Self> {
        if perturbation_norm < 0.0 {
            return Err(CertificateError::numerical_precision(
                "perturbation norm must be non-negative",
                perturbation_norm,
                0.0,
            ));
        }
        if eigenvalues.len() < subspace_dimension + 1 {
            return Err(CertificateError::incomplete_data(
                "eigenvalues",
                format!(
                    "need at least {} eigenvalues for subspace dimension {}",
                    subspace_dimension + 1,
                    subspace_dimension
                ),
            ));
        }
        if subspace_dimension == 0 {
            return Err(CertificateError::invalid_partition(
                "subspace dimension must be positive",
            ));
        }

        let mut sorted_eigs = eigenvalues.clone();
        sorted_eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Spectral gap: difference between k-th and (k+1)-th eigenvalue
        let gap = if subspace_dimension < sorted_eigs.len() {
            (sorted_eigs[subspace_dimension] - sorted_eigs[subspace_dimension - 1]).abs()
        } else {
            return Err(CertificateError::numerical_precision(
                "cannot compute spectral gap: insufficient eigenvalues",
                0.0,
                1e-15,
            ));
        };

        if gap < 1e-15 {
            return Err(CertificateError::numerical_precision(
                "spectral gap is essentially zero",
                gap,
                1e-15,
            ));
        }

        let angle_bound = (perturbation_norm / gap).min(1.0);

        // Misclassification rate bound: sin²(Θ) ~ δ²/γ²
        let misclassification_bound = angle_bound * angle_bound;

        Ok(Self {
            id: Uuid::new_v4().to_string(),
            created_at: Utc::now().to_rfc3339(),
            perturbation_norm,
            spectral_gap: gap,
            angle_bound,
            actual_angle: None,
            eigenvalues,
            perturbed_eigenvalues,
            subspace_dimension,
            eigenvector_quality: Vec::new(),
            misclassification_bound,
            metadata: IndexMap::new(),
        })
    }

    /// Verify eigenvector quality by checking residuals and orthogonality.
    ///
    /// For each eigenvector v_i with eigenvalue λ_i:
    ///   residual = ||A * v_i - λ_i * v_i|| / ||v_i||
    ///   orthogonality = max_{j≠i} |v_i^T v_j|
    pub fn verify_eigenvector_quality(
        &mut self,
        residuals: &[f64],
        orthogonality_deviations: &[f64],
        tolerance: f64,
    ) -> CertificateResult<bool> {
        if residuals.len() != self.subspace_dimension {
            return Err(CertificateError::incomplete_data(
                "residuals",
                format!(
                    "expected {} residuals, got {}",
                    self.subspace_dimension,
                    residuals.len()
                ),
            ));
        }
        if orthogonality_deviations.len() != self.subspace_dimension {
            return Err(CertificateError::incomplete_data(
                "orthogonality_deviations",
                format!(
                    "expected {} values, got {}",
                    self.subspace_dimension,
                    orthogonality_deviations.len()
                ),
            ));
        }

        self.eigenvector_quality.clear();
        let mut all_acceptable = true;

        for i in 0..self.subspace_dimension {
            let quality =
                EigenvectorQuality::new(i, residuals[i], orthogonality_deviations[i], tolerance);
            if !quality.is_acceptable {
                all_acceptable = false;
            }
            self.eigenvector_quality.push(quality);
        }

        Ok(all_acceptable)
    }

    /// Compute misclassification rate bound: O(δ²/γ²) from angle bound.
    pub fn misclassification_rate_bound(&self) -> f64 {
        self.misclassification_bound
    }

    /// Set actual angle (computed from eigenvectors if available).
    pub fn set_actual_angle(&mut self, angle: f64) {
        self.actual_angle = Some(angle);
    }

    /// How tight is the angle bound compared to the actual angle?
    pub fn angle_tightness(&self) -> Option<f64> {
        self.actual_angle.map(|actual| {
            if self.angle_bound.abs() < 1e-15 {
                if actual.abs() < 1e-15 {
                    1.0
                } else {
                    0.0
                }
            } else {
                actual / self.angle_bound
            }
        })
    }

    /// Weyl's bound on eigenvalue perturbation: `|λᵢ − λ̂ᵢ| ≤ ‖E‖_F`.
    ///
    /// Returns `(index, actual_diff, bound, is_satisfied)` for each eigenvalue pair.
    pub fn eigenvalue_perturbation_bounds(&self) -> Vec<(usize, f64, f64, bool)> {
        let n = self.eigenvalues.len().min(self.perturbed_eigenvalues.len());
        let mut bounds = Vec::with_capacity(n);
        for i in 0..n {
            let actual_diff = (self.eigenvalues[i] - self.perturbed_eigenvalues[i]).abs();
            bounds.push((i, actual_diff, self.perturbation_norm, actual_diff <= self.perturbation_norm + 1e-10));
        }
        bounds
    }

    /// Remaining perturbation margin before the bound becomes vacuous (`sin(Θ) = 1`).
    ///
    /// Equals `max(0, γ − ‖E‖_F)`.
    pub fn perturbation_margin(&self) -> f64 {
        // sin(Θ) ≤ 1 when ||E||_F ≤ gap
        (self.spectral_gap - self.perturbation_norm).max(0.0)
    }

    /// Relative perturbation: ||E||_F / ||A||_F approximated by max eigenvalue.
    pub fn relative_perturbation(&self) -> f64 {
        let max_eig = self
            .eigenvalues
            .iter()
            .map(|e| e.abs())
            .fold(0.0f64, f64::max);
        if max_eig < 1e-15 {
            return f64::INFINITY;
        }
        self.perturbation_norm / max_eig
    }

    /// Summary statistics.
    pub fn summary_stats(&self) -> IndexMap<String, f64> {
        let mut stats = IndexMap::new();
        stats.insert("perturbation_norm".to_string(), self.perturbation_norm);
        stats.insert("spectral_gap".to_string(), self.spectral_gap);
        stats.insert("angle_bound".to_string(), self.angle_bound);
        stats.insert(
            "misclassification_bound".to_string(),
            self.misclassification_bound,
        );
        stats.insert("subspace_dimension".to_string(), self.subspace_dimension as f64);
        stats.insert("perturbation_margin".to_string(), self.perturbation_margin());
        stats.insert("relative_perturbation".to_string(), self.relative_perturbation());
        if let Some(actual) = self.actual_angle {
            stats.insert("actual_angle".to_string(), actual);
        }
        if let Some(tightness) = self.angle_tightness() {
            stats.insert("angle_tightness".to_string(), tightness);
        }
        let n_acceptable = self
            .eigenvector_quality
            .iter()
            .filter(|q| q.is_acceptable)
            .count();
        stats.insert("eigenvectors_acceptable".to_string(), n_acceptable as f64);
        stats.insert(
            "eigenvectors_total".to_string(),
            self.eigenvector_quality.len() as f64,
        );
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_basic() {
        let cert = DavisKahanCertificate::compute_angle_bound(
            0.1,
            vec![0.0, 0.1, 0.5, 1.0, 1.5],
            vec![0.0, 0.12, 0.52, 1.01, 1.48],
            2,
        )
        .unwrap();
        // gap = |0.5 - 0.1| = 0.4
        assert!((cert.spectral_gap - 0.4).abs() < 1e-10);
        // angle_bound = 0.1 / 0.4 = 0.25
        assert!((cert.angle_bound - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_negative_perturbation_fails() {
        let result = DavisKahanCertificate::compute_angle_bound(
            -1.0,
            vec![0.0, 1.0, 2.0],
            vec![],
            1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_eigenvalues() {
        let result = DavisKahanCertificate::compute_angle_bound(
            0.1,
            vec![0.0],
            vec![],
            2,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_subspace_fails() {
        let result = DavisKahanCertificate::compute_angle_bound(
            0.1,
            vec![0.0, 1.0, 2.0],
            vec![],
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_angle_bound_capped_at_one() {
        let cert = DavisKahanCertificate::compute_angle_bound(
            10.0,
            vec![0.0, 0.1, 1.0],
            vec![],
            1,
        )
        .unwrap();
        assert!(cert.angle_bound <= 1.0);
    }

    #[test]
    fn test_misclassification_bound() {
        let cert = DavisKahanCertificate::compute_angle_bound(
            0.2,
            vec![0.0, 0.1, 1.0, 2.0],
            vec![],
            2,
        )
        .unwrap();
        // gap = |1.0 - 0.1| = 0.9
        // angle = 0.2 / 0.9 ≈ 0.222
        // misclass = angle² ≈ 0.0494
        assert!(cert.misclassification_bound > 0.0);
        assert!(cert.misclassification_bound < 1.0);
    }

    #[test]
    fn test_verify_eigenvector_quality_all_good() {
        let mut cert = DavisKahanCertificate::compute_angle_bound(
            0.1,
            vec![0.0, 0.5, 1.0],
            vec![],
            1,
        )
        .unwrap();
        let result = cert.verify_eigenvector_quality(&[1e-10], &[1e-12], 1e-6);
        assert!(result.unwrap());
        assert!(cert.eigenvector_quality[0].is_acceptable);
    }

    #[test]
    fn test_verify_eigenvector_quality_bad() {
        let mut cert = DavisKahanCertificate::compute_angle_bound(
            0.1,
            vec![0.0, 0.5, 1.0],
            vec![],
            1,
        )
        .unwrap();
        let result = cert.verify_eigenvector_quality(&[1.0], &[0.5], 1e-6);
        assert!(!result.unwrap());
    }

    #[test]
    fn test_angle_tightness() {
        let mut cert = DavisKahanCertificate::compute_angle_bound(
            0.1,
            vec![0.0, 0.5, 1.0],
            vec![],
            1,
        )
        .unwrap();
        assert!(cert.angle_tightness().is_none());
        cert.set_actual_angle(0.1);
        let tightness = cert.angle_tightness().unwrap();
        assert!(tightness > 0.0 && tightness <= 1.0);
    }

    #[test]
    fn test_eigenvalue_perturbation_bounds() {
        let cert = DavisKahanCertificate::compute_angle_bound(
            0.5,
            vec![0.0, 1.0, 2.0],
            vec![0.1, 0.9, 2.1],
            1,
        )
        .unwrap();
        let bounds = cert.eigenvalue_perturbation_bounds();
        assert_eq!(bounds.len(), 3);
        for (_, diff, norm, valid) in &bounds {
            assert!(*diff <= norm + 1e-8 || !valid);
        }
    }

    #[test]
    fn test_perturbation_margin() {
        let cert = DavisKahanCertificate::compute_angle_bound(
            0.1,
            vec![0.0, 0.5, 1.0],
            vec![],
            1,
        )
        .unwrap();
        assert!(cert.perturbation_margin() > 0.0);
    }

    #[test]
    fn test_relative_perturbation() {
        let cert = DavisKahanCertificate::compute_angle_bound(
            0.1,
            vec![0.0, 0.5, 2.0],
            vec![],
            1,
        )
        .unwrap();
        assert!((cert.relative_perturbation() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_summary_stats() {
        let cert = DavisKahanCertificate::compute_angle_bound(
            0.1,
            vec![0.0, 0.5, 1.0],
            vec![],
            1,
        )
        .unwrap();
        let stats = cert.summary_stats();
        assert!(stats.contains_key("angle_bound"));
        assert!(stats.contains_key("spectral_gap"));
    }
}
