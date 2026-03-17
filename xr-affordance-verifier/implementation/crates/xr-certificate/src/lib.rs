//! XR Certificate – Certificate generation and validation for the XR Affordance Verifier.
//!
//! Provides the full Tier 2 certificate generation pipeline:
//! 1. Stratify the body-parameter space Θ\_target
//! 2. Run Tier 1 interval verification for green/red classification
//! 3. Sample yellow (uncertain) regions adaptively
//! 4. Target SMT solvers at frontier regions
//! 5. Compose results into a coverage certificate C = ⟨S, V, U, ε_a, ε_e, δ, κ⟩
//!
//! # Key components
//!
//! - [`sampling::StratifiedSampler`] – adaptive stratified sampling with LHS and Halton
//! - [`hoeffding::HoeffdingBound`] – statistical error bounds via Hoeffding's inequality
//! - [`certificate_builder::CertificateAssembler`] – assembles all evidence into a certificate
//! - [`coverage::CoverageAnalyzer`] – spatial coverage analysis and heatmap generation
//! - [`composition::CertificateComposer`] – combines Tier 1, Tier 2, and SMT results
//! - [`frontier::FrontierDetector`] – identifies accessibility boundaries
//! - [`boundary::DiscontinuityDetector`] – detects Lipschitz-violating joint-limit boundaries
//! - [`boundary::BoundaryVerifier`] – exhaustive boundary-corridor verification
//! - [`boundary::MultiStepStratifier`] – dependency-guided stratification for k≥3 interactions
//! - [`tier2_engine::Tier2Verifier`] – top-level orchestration engine
//! - [`validation::CertificateValidator`] – internal consistency checking
//! - [`export::CertificateExporter`] – serialization and compliance export

pub mod sampling;
pub mod hoeffding;
pub mod certificate_builder;
pub mod coverage;
pub mod composition;
pub mod frontier;
pub mod boundary;
pub mod tier2_engine;
pub mod validation;
pub mod export;

pub use sampling::StratifiedSampler;
pub use hoeffding::HoeffdingBound;
pub use certificate_builder::CertificateAssembler;
pub use coverage::{CoverageAnalyzer, CoverageMap};
pub use composition::CertificateComposer;
pub use frontier::FrontierDetector;
pub use boundary::{
    AdaptiveBoundarySampler, BoundaryVerifier, DiscontinuityDetector, MultiStepStratifier,
};
pub use tier2_engine::Tier2Verifier;
pub use validation::CertificateValidator;
pub use export::CertificateExporter;

use xr_types::certificate::{
    CertificateBuilder, CoverageCertificate, SampleVerdict,
    VerifiedRegion, ViolationSurface,
};
use xr_types::error::VerifierResult;
use xr_types::scene::SceneModel;
use serde::{Deserialize, Serialize};

/// Configuration for certificate generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateConfig {
    /// Minimum kappa required for a valid certificate.
    pub min_kappa: f64,
    /// Target confidence (1 - delta).
    pub target_confidence: f64,
    /// Whether to include all sample data in the certificate.
    pub include_samples: bool,
    /// Maximum allowed linearization error.
    pub max_linearization_error: f64,
}

impl Default for CertificateConfig {
    fn default() -> Self {
        Self {
            min_kappa: 0.90,
            target_confidence: 0.95,
            include_samples: true,
            max_linearization_error: 0.01,
        }
    }
}

/// High-level certificate generator.
pub struct CertificateGenerator {
    config: CertificateConfig,
}

impl CertificateGenerator {
    pub fn new() -> Self {
        Self {
            config: CertificateConfig::default(),
        }
    }

    pub fn with_config(config: CertificateConfig) -> Self {
        Self { config }
    }

    /// Generate a certificate from sample verdicts and verified regions.
    pub fn generate(
        &self,
        scene: &SceneModel,
        samples: Vec<SampleVerdict>,
        verified_regions: Vec<VerifiedRegion>,
        violations: Vec<ViolationSurface>,
        epsilon_analytical: f64,
        total_time_s: f64,
    ) -> VerifierResult<CoverageCertificate> {
        let delta = 1.0 - self.config.target_confidence;
        let total_param_volume = 1.0;

        let mut builder = CertificateBuilder::new(scene.id);
        builder = builder
            .add_samples(samples)
            .epsilon_analytical(epsilon_analytical)
            .delta(delta)
            .total_time(total_time_s)
            .total_param_volume(total_param_volume)
            .meta("scene_name", &scene.name)
            .meta("generator", "xr-certificate")
            .meta("protocol_version", xr_types::PROTOCOL_VERSION);

        for region in &verified_regions {
            builder = builder.add_verified_region(region.clone());
        }
        for violation in &violations {
            builder = builder.add_violation(violation.clone());
        }

        let cert = builder.build()?;
        Ok(cert)
    }

    /// Validate that a certificate meets minimum requirements.
    pub fn validate_certificate(&self, cert: &CoverageCertificate) -> Vec<String> {
        let mut issues = Vec::new();

        if cert.kappa < self.config.min_kappa {
            issues.push(format!(
                "Coverage kappa {:.4} is below minimum {:.4}",
                cert.kappa, self.config.min_kappa
            ));
        }

        if cert.delta > (1.0 - self.config.target_confidence) {
            issues.push(format!(
                "Confidence delta {:.4} exceeds target {:.4}",
                cert.delta,
                1.0 - self.config.target_confidence
            ));
        }

        if cert.epsilon_analytical > self.config.max_linearization_error {
            issues.push(format!(
                "Linearization error {:.6} exceeds max {:.6}",
                cert.epsilon_analytical, self.config.max_linearization_error
            ));
        }

        if cert.samples.is_empty() && cert.verified_regions.is_empty() {
            issues.push("Certificate has no evidence (no samples or verified regions)".into());
        }

        issues
    }

    /// Generate a summary report string from a certificate.
    pub fn format_summary(cert: &CoverageCertificate) -> String {
        let summary = cert.summary();
        format!(
            "Certificate {} | Grade: {:?} | kappa={:.4} | eps_a={:.6} eps_e={:.6} | delta={:.4} | \
             Samples: {}/{} pass | Regions: {} | Violations: {} | Time: {:.2}s",
            summary.id,
            summary.grade,
            summary.kappa,
            summary.epsilon_analytical,
            summary.epsilon_estimated,
            summary.delta,
            summary.num_pass,
            summary.num_samples,
            summary.num_verified_regions,
            summary.num_violations,
            summary.total_time_s,
        )
    }
}

impl Default for CertificateGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_certificate_generator_default() {
        let gen = CertificateGenerator::new();
        assert!((gen.config.min_kappa - 0.90).abs() < 1e-10);
    }

    #[test]
    fn test_certificate_config_default() {
        let cfg = CertificateConfig::default();
        assert!(cfg.include_samples);
        assert!((cfg.target_confidence - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_generate_certificate() {
        let scene = SceneModel::new("test_scene");
        let gen = CertificateGenerator::new();
        let element_id = Uuid::new_v4();
        let samples = vec![
            SampleVerdict::pass(vec![1.7, 0.34, 0.42, 0.26, 0.19], element_id),
            SampleVerdict::pass(vec![1.6, 0.32, 0.40, 0.24, 0.18], element_id),
        ];
        let cert = gen
            .generate(&scene, samples, vec![], vec![], 0.001, 1.5)
            .unwrap();
        assert!(cert.kappa > 0.0);
        assert!(!cert.samples.is_empty());
    }

    #[test]
    fn test_validate_certificate() {
        let scene = SceneModel::new("test_scene");
        let gen = CertificateGenerator::new();
        let element_id = Uuid::new_v4();
        let samples = vec![SampleVerdict::pass(
            vec![1.7, 0.34, 0.42, 0.26, 0.19],
            element_id,
        )];
        let cert = gen
            .generate(&scene, samples, vec![], vec![], 0.001, 1.0)
            .unwrap();
        let issues = gen.validate_certificate(&cert);
        // With only 1 sample, kappa will be low
        assert!(!issues.is_empty() || cert.kappa >= gen.config.min_kappa);
    }

    #[test]
    fn test_format_summary() {
        let scene = SceneModel::new("test");
        let gen = CertificateGenerator::new();
        let cert = gen
            .generate(&scene, vec![], vec![], vec![], 0.0, 0.0)
            .unwrap();
        let summary = CertificateGenerator::format_summary(&cert);
        assert!(summary.contains("Grade:"));
        assert!(summary.contains("kappa="));
    }
}
