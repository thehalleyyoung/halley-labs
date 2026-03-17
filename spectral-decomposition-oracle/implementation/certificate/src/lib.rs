//! Certificate generation and verification for decomposition quality guarantees.
//!
//! This crate provides formal and empirical certificates for mathematical
//! decomposition quality. Certificates give provable or empirically-validated
//! bounds on solution quality, enabling users to trust decomposition results
//! without re-solving the original problem.
//!
//! # Modules
//!
//! - [`spectral_bound`] — Davis-Kahan sin-theta certificates, partition quality,
//!   and spectral scaling laws.
//! - [`l3_bound`] — Lemma L3 partition-to-bound bridges (Benders, Dantzig-Wolfe).
//! - [`futility`] — Futility prediction with calibrated confidence.
//! - [`verification`] — Independent bound and partition checkers.
//! - [`report`] — Certificate reports, comparisons, and visualization data.

pub mod error;
pub mod futility;
pub mod l3_bound;
pub mod report;
pub mod spectral_bound;
pub mod verification;

pub use error::CertificateError;
pub use futility::calibration::{CalibrationResult, TemperatureScaling};
pub use futility::certificate::FutilityCertificate;
pub use l3_bound::benders_cert::BendersCertificate;
pub use l3_bound::dw_cert::DWCertificate;
pub use l3_bound::partition_bound::L3PartitionCertificate;
pub use report::comparison::ComparisonReport;
pub use report::generator::CertificateReport;
pub use report::visualization::VisualizationData;
pub use spectral_bound::davis_kahan::DavisKahanCertificate;
pub use spectral_bound::partition_quality::PartitionQualityCertificate;
pub use spectral_bound::scaling_law::SpectralScalingCertificate;
pub use verification::bound_checker::BoundChecker;
pub use verification::dual_checker::DualChecker;
pub use verification::partition_checker::PartitionChecker;
