//! T2 Spectral Scaling Law certificate.
//!
//! The spectral scaling law bounds the decomposition gap as:
//!   z_LP - z_D(π̂) ≤ C * δ²/γ²
//!
//! where:
//!   - δ² is the coupling energy (off-diagonal Laplacian mass)
//!   - γ² is the spectral gap squared (eigenvalue gap after the k-th eigenvalue)
//!   - C = O(k * κ⁴ * ||c||∞) is a constant depending on problem structure
//!   - k is the number of blocks
//!   - κ is the condition number of the constraint matrix
//!   - ||c||∞ is the infinity norm of the objective

use crate::error::{CertificateError, CertificateResult};
use chrono::Utc;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Vacuousness assessment levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VacuousnessLevel {
    /// Bound is tight enough to be practically useful
    Useful,
    /// Bound is loose but still informative
    Loose,
    /// Bound is too loose to be practically useful
    Vacuous,
    /// Bound is astronomically large (numerical issues likely)
    Degenerate,
}

impl std::fmt::Display for VacuousnessLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Useful => write!(f, "USEFUL"),
            Self::Loose => write!(f, "LOOSE"),
            Self::Vacuous => write!(f, "VACUOUS"),
            Self::Degenerate => write!(f, "DEGENERATE"),
        }
    }
}

/// T2 Spectral Scaling Law Certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralScalingCertificate {
    pub id: String,
    pub created_at: String,
    /// δ²: coupling energy (off-diagonal Laplacian mass)
    pub delta_squared: f64,
    /// γ²: spectral gap squared
    pub gamma_squared: f64,
    /// k: number of blocks
    pub k: usize,
    /// κ: condition number
    pub kappa: f64,
    /// ||c||∞: objective infinity norm
    pub c_inf_norm: f64,
    /// Computed constant C = O(k * κ⁴ * ||c||∞)
    pub constant_c: f64,
    /// Bound value = C * δ²/γ²
    pub bound_value: f64,
    /// Whether the bound is vacuous
    pub is_vacuous: bool,
    /// Vacuousness level
    pub vacuousness_level: VacuousnessLevel,
    /// Empirical gap for comparison
    pub empirical_gap: Option<f64>,
    /// Ratio of bound to empirical gap
    pub overestimation_ratio: Option<f64>,
    /// Raw eigenvalues used in computation
    pub eigenvalues: Vec<f64>,
    pub metadata: IndexMap<String, String>,
}

impl SpectralScalingCertificate {
    /// Compute the T2 spectral scaling law bound.
    ///
    /// bound = C * δ² / γ²
    /// where C = k * κ⁴ * ||c||∞
    pub fn compute(
        delta_squared: f64,
        gamma_squared: f64,
        k: usize,
        kappa: f64,
        c_inf_norm: f64,
        eigenvalues: Vec<f64>,
    ) -> CertificateResult<Self> {
        if delta_squared < 0.0 {
            return Err(CertificateError::numerical_precision(
                "δ² must be non-negative",
                delta_squared,
                0.0,
            ));
        }
        if gamma_squared <= 0.0 {
            return Err(CertificateError::numerical_precision(
                "γ² must be positive (zero spectral gap means problem is disconnected)",
                gamma_squared,
                1e-15,
            ));
        }
        if k == 0 {
            return Err(CertificateError::invalid_partition("k must be positive"));
        }
        if kappa < 1.0 {
            return Err(CertificateError::numerical_precision(
                "condition number κ must be ≥ 1",
                kappa,
                1.0,
            ));
        }
        if c_inf_norm < 0.0 {
            return Err(CertificateError::numerical_precision(
                "||c||∞ must be non-negative",
                c_inf_norm,
                0.0,
            ));
        }

        let constant_c = k as f64 * kappa.powi(4) * c_inf_norm;
        let bound_value = constant_c * delta_squared / gamma_squared;

        let (is_vacuous, vacuousness_level) = Self::assess_vacuousness_internal(
            bound_value,
            kappa,
            c_inf_norm,
            gamma_squared,
        );

        Ok(Self {
            id: Uuid::new_v4().to_string(),
            created_at: Utc::now().to_rfc3339(),
            delta_squared,
            gamma_squared,
            k,
            kappa,
            c_inf_norm,
            constant_c,
            bound_value,
            is_vacuous,
            vacuousness_level,
            empirical_gap: None,
            overestimation_ratio: None,
            eigenvalues,
            metadata: IndexMap::new(),
        })
    }

    fn assess_vacuousness_internal(
        bound: f64,
        kappa: f64,
        c_inf: f64,
        gamma_sq: f64,
    ) -> (bool, VacuousnessLevel) {
        if !bound.is_finite() || bound > 1e20 {
            return (true, VacuousnessLevel::Degenerate);
        }
        if kappa > 1e3 {
            return (true, VacuousnessLevel::Vacuous);
        }
        if gamma_sq < 1e-8 {
            return (true, VacuousnessLevel::Vacuous);
        }
        // If bound is more than 100x the objective scale, it's likely vacuous
        if c_inf > 0.0 && bound / c_inf > 100.0 {
            return (true, VacuousnessLevel::Loose);
        }
        if bound > 1e6 {
            return (true, VacuousnessLevel::Loose);
        }
        (false, VacuousnessLevel::Useful)
    }

    /// Assess whether the bound is practically useful.
    pub fn assess_vacuousness(&self) -> VacuousnessLevel {
        self.vacuousness_level
    }

    /// Compare the theoretical bound with the empirical gap.
    pub fn compare_with_empirical(&mut self, empirical_gap: f64) -> IndexMap<String, f64> {
        self.empirical_gap = Some(empirical_gap);
        let ratio = if empirical_gap.abs() > 1e-15 {
            self.bound_value / empirical_gap
        } else if self.bound_value.abs() < 1e-15 {
            1.0
        } else {
            f64::INFINITY
        };
        self.overestimation_ratio = Some(ratio);

        let mut comparison = IndexMap::new();
        comparison.insert("theoretical_bound".to_string(), self.bound_value);
        comparison.insert("empirical_gap".to_string(), empirical_gap);
        comparison.insert("overestimation_ratio".to_string(), ratio);
        comparison.insert("log10_ratio".to_string(), ratio.log10());
        comparison.insert(
            "bound_is_valid".to_string(),
            if self.bound_value >= empirical_gap - 1e-8 {
                1.0
            } else {
                0.0
            },
        );
        comparison
    }

    /// Sensitivity analysis: how the bound changes with each parameter.
    pub fn sensitivity_analysis(&self) -> IndexMap<String, f64> {
        let mut sens = IndexMap::new();

        // ∂bound/∂(δ²) = C / γ²
        let d_delta = self.constant_c / self.gamma_squared;
        sens.insert("d_bound_d_delta_sq".to_string(), d_delta);

        // ∂bound/∂(γ²) = -C * δ² / γ⁴
        let d_gamma = -self.constant_c * self.delta_squared / self.gamma_squared.powi(2);
        sens.insert("d_bound_d_gamma_sq".to_string(), d_gamma);

        // ∂bound/∂κ = 4 * k * κ³ * ||c||∞ * δ² / γ²
        let d_kappa = 4.0 * self.k as f64 * self.kappa.powi(3) * self.c_inf_norm
            * self.delta_squared / self.gamma_squared;
        sens.insert("d_bound_d_kappa".to_string(), d_kappa);

        // Elasticities (proportional sensitivities)
        if self.bound_value.abs() > 1e-15 {
            sens.insert(
                "elasticity_delta_sq".to_string(),
                d_delta * self.delta_squared / self.bound_value,
            );
            sens.insert(
                "elasticity_gamma_sq".to_string(),
                d_gamma * self.gamma_squared / self.bound_value,
            );
            sens.insert(
                "elasticity_kappa".to_string(),
                d_kappa * self.kappa / self.bound_value,
            );
        }

        sens
    }

    /// What kappa threshold would make the bound non-vacuous?
    pub fn critical_kappa(&self 
    ) -> f64 {
        // Solve: k * κ⁴ * ||c||∞ * δ²/γ² ≤ target
        // where target = ||c||∞ (bound should be at most objective scale)
        let target = self.c_inf_norm.max(1.0);
        if self.k == 0 || self.delta_squared.abs() < 1e-15 || self.c_inf_norm.abs() < 1e-15 {
            return f64::INFINITY;
        }
        let rhs = target * self.gamma_squared / (self.k as f64 * self.c_inf_norm * self.delta_squared);
        if rhs <= 0.0 {
            return f64::INFINITY;
        }
        rhs.powf(0.25)
    }

    /// What spectral gap would make the bound non-vacuous?
    pub fn critical_spectral_gap(&self) -> f64 {
        let target = self.c_inf_norm.max(1.0);
        if target.abs() < 1e-15 {
            return 0.0;
        }
        (self.constant_c * self.delta_squared / target).sqrt()
    }

    /// Spectral ratio δ²/γ².
    pub fn spectral_ratio(&self) -> f64 {
        self.delta_squared / self.gamma_squared
    }

    /// Contribution breakdown: which factor dominates the bound?
    pub fn factor_breakdown(&self) -> IndexMap<String, f64> {
        let mut breakdown = IndexMap::new();
        let log_bound = self.bound_value.ln();

        let log_k = (self.k as f64).ln();
        let log_kappa4 = 4.0 * self.kappa.ln();
        let log_c = self.c_inf_norm.ln();
        let log_ratio = (self.delta_squared / self.gamma_squared).ln();

        let total_log = log_k + log_kappa4 + log_c + log_ratio;

        if total_log.abs() > 1e-15 {
            breakdown.insert("k_fraction".to_string(), log_k / total_log);
            breakdown.insert("kappa4_fraction".to_string(), log_kappa4 / total_log);
            breakdown.insert("c_inf_fraction".to_string(), log_c / total_log);
            breakdown.insert("spectral_ratio_fraction".to_string(), log_ratio / total_log);
        }

        breakdown.insert("log_bound".to_string(), log_bound);
        breakdown.insert("log_k".to_string(), log_k);
        breakdown.insert("log_kappa4".to_string(), log_kappa4);
        breakdown.insert("log_c_inf".to_string(), log_c);
        breakdown.insert("log_spectral_ratio".to_string(), log_ratio);

        breakdown
    }

    /// Summary statistics.
    pub fn summary_stats(&self) -> IndexMap<String, f64> {
        let mut stats = IndexMap::new();
        stats.insert("bound_value".to_string(), self.bound_value);
        stats.insert("delta_squared".to_string(), self.delta_squared);
        stats.insert("gamma_squared".to_string(), self.gamma_squared);
        stats.insert("k".to_string(), self.k as f64);
        stats.insert("kappa".to_string(), self.kappa);
        stats.insert("c_inf_norm".to_string(), self.c_inf_norm);
        stats.insert("constant_c".to_string(), self.constant_c);
        stats.insert("spectral_ratio".to_string(), self.spectral_ratio());
        stats.insert(
            "is_vacuous".to_string(),
            if self.is_vacuous { 1.0 } else { 0.0 },
        );
        stats.insert("critical_kappa".to_string(), self.critical_kappa());
        stats.insert("critical_spectral_gap".to_string(), self.critical_spectral_gap());
        if let Some(emp) = self.empirical_gap {
            stats.insert("empirical_gap".to_string(), emp);
        }
        if let Some(ratio) = self.overestimation_ratio {
            stats.insert("overestimation_ratio".to_string(), ratio);
        }
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_basic() {
        let cert = SpectralScalingCertificate::compute(
            0.1,  // delta^2
            0.5,  // gamma^2
            3,    // k
            2.0,  // kappa
            10.0, // c_inf
            vec![0.0, 0.1, 0.6, 1.2],
        )
        .unwrap();
        // C = 3 * 16 * 10 = 480
        // bound = 480 * 0.1 / 0.5 = 96
        assert!((cert.constant_c - 480.0).abs() < 1e-10);
        assert!((cert.bound_value - 96.0).abs() < 1e-10);
    }

    #[test]
    fn test_negative_delta_fails() {
        let result = SpectralScalingCertificate::compute(-1.0, 0.5, 3, 2.0, 10.0, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_gamma_fails() {
        let result = SpectralScalingCertificate::compute(0.1, 0.0, 3, 2.0, 10.0, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_k_fails() {
        let result = SpectralScalingCertificate::compute(0.1, 0.5, 0, 2.0, 10.0, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_kappa_below_one_fails() {
        let result = SpectralScalingCertificate::compute(0.1, 0.5, 3, 0.5, 10.0, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vacuousness_useful() {
        let cert = SpectralScalingCertificate::compute(0.01, 1.0, 2, 2.0, 10.0, vec![]).unwrap();
        assert!(!cert.is_vacuous);
        assert_eq!(cert.vacuousness_level, VacuousnessLevel::Useful);
    }

    #[test]
    fn test_vacuousness_high_kappa() {
        let cert =
            SpectralScalingCertificate::compute(0.1, 0.5, 3, 1e4, 10.0, vec![]).unwrap();
        assert!(cert.is_vacuous);
        assert_eq!(cert.vacuousness_level, VacuousnessLevel::Vacuous);
    }

    #[test]
    fn test_compare_with_empirical() {
        let mut cert =
            SpectralScalingCertificate::compute(0.1, 0.5, 3, 2.0, 10.0, vec![]).unwrap();
        let comparison = cert.compare_with_empirical(5.0);
        assert!(comparison.contains_key("overestimation_ratio"));
        assert!(cert.overestimation_ratio.unwrap() > 1.0);
    }

    #[test]
    fn test_sensitivity_analysis() {
        let cert = SpectralScalingCertificate::compute(0.1, 0.5, 3, 2.0, 10.0, vec![]).unwrap();
        let sens = cert.sensitivity_analysis();
        assert!(sens.contains_key("d_bound_d_delta_sq"));
        assert!(sens.contains_key("d_bound_d_gamma_sq"));
        assert!(sens.contains_key("d_bound_d_kappa"));
        // d_bound/d_gamma_sq should be negative (larger gap = smaller bound)
        assert!(sens["d_bound_d_gamma_sq"] < 0.0);
    }

    #[test]
    fn test_critical_kappa() {
        let cert = SpectralScalingCertificate::compute(0.1, 0.5, 3, 2.0, 10.0, vec![]).unwrap();
        let ck = cert.critical_kappa();
        assert!(ck > 0.0);
        assert!(ck.is_finite());
    }

    #[test]
    fn test_spectral_ratio() {
        let cert = SpectralScalingCertificate::compute(0.3, 0.6, 2, 1.5, 5.0, vec![]).unwrap();
        assert!((cert.spectral_ratio() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_factor_breakdown() {
        let cert = SpectralScalingCertificate::compute(0.1, 0.5, 3, 2.0, 10.0, vec![]).unwrap();
        let breakdown = cert.factor_breakdown();
        assert!(breakdown.contains_key("kappa4_fraction"));
        assert!(breakdown.contains_key("log_bound"));
    }

    #[test]
    fn test_summary_stats() {
        let cert = SpectralScalingCertificate::compute(0.1, 0.5, 3, 2.0, 10.0, vec![]).unwrap();
        let stats = cert.summary_stats();
        assert!(stats.contains_key("bound_value"));
        assert!(stats.contains_key("spectral_ratio"));
    }
}
