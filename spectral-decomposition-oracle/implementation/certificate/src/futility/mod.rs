//! Futility certificates: empirically calibrated predictions.
//!
//! These are NOT formal proofs but rather calibrated predictions about whether
//! decomposition is likely to be futile (i.e., no method will achieve a
//! meaningful gap closure within time budget).

pub mod calibration;
pub mod certificate;

pub use calibration::{CalibrationResult, TemperatureScaling};
pub use certificate::FutilityCertificate;

use serde::{Deserialize, Serialize};

/// Futility prediction: whether decomposition is predicted to be futile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FutilityPrediction {
    Futile,
    NotFutile,
    Uncertain,
}

impl std::fmt::Display for FutilityPrediction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Futile => write!(f, "FUTILE"),
            Self::NotFutile => write!(f, "NOT_FUTILE"),
            Self::Uncertain => write!(f, "UNCERTAIN"),
        }
    }
}

/// Distinguish from formal certificate: this is empirical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificateType {
    /// Mathematically proven bound (L3, T2, Davis-Kahan)
    Formal,
    /// Empirically calibrated prediction (futility)
    Empirical,
}

impl std::fmt::Display for CertificateType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Formal => write!(f, "FORMAL"),
            Self::Empirical => write!(f, "EMPIRICAL"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_futility_prediction_display() {
        assert_eq!(format!("{}", FutilityPrediction::Futile), "FUTILE");
        assert_eq!(format!("{}", FutilityPrediction::NotFutile), "NOT_FUTILE");
        assert_eq!(format!("{}", FutilityPrediction::Uncertain), "UNCERTAIN");
    }

    #[test]
    fn test_certificate_type_display() {
        assert_eq!(format!("{}", CertificateType::Formal), "FORMAL");
        assert_eq!(format!("{}", CertificateType::Empirical), "EMPIRICAL");
    }

    #[test]
    fn test_prediction_equality() {
        assert_eq!(FutilityPrediction::Futile, FutilityPrediction::Futile);
        assert_ne!(FutilityPrediction::Futile, FutilityPrediction::NotFutile);
    }
}
