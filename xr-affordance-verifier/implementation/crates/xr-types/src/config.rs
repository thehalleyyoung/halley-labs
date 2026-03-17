//! Verifier configuration types.
//!
//! Provides [`VerifierConfig`] with nested configs for the two verification
//! tiers, sampling parameters, SMT parameters, population settings, and
//! output settings. Supports JSON serialization, validation, and a builder
//! pattern for ergonomic construction.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::device::MovementMode;
use crate::error::{VerifierError, VerifierResult};

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// Full configuration for the XR Affordance Verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierConfig {
    /// Name / label for this configuration.
    pub name: String,
    /// Configuration for Tier 1 (sampling-based falsification).
    pub tier1: Tier1Config,
    /// Configuration for Tier 2 (SMT-based formal verification).
    pub tier2: Tier2Config,
    /// Sampling parameters.
    pub sampling: SamplingConfig,
    /// SMT solver parameters.
    pub smt: SmtConfig,
    /// Population / anthropometric parameters.
    pub population: PopulationConfig,
    /// Output settings.
    pub output: OutputConfig,
    /// General flags.
    pub general: GeneralConfig,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            name: "default".into(),
            tier1: Tier1Config::default(),
            tier2: Tier2Config::default(),
            sampling: SamplingConfig::default(),
            smt: SmtConfig::default(),
            population: PopulationConfig::default(),
            output: OutputConfig::default(),
            general: GeneralConfig::default(),
        }
    }
}

impl VerifierConfig {
    /// Create a default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load from a JSON string.
    pub fn from_json(json: &str) -> VerifierResult<Self> {
        let cfg: Self = serde_json::from_str(json).map_err(VerifierError::from)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Serialize to a pretty-printed JSON string.
    pub fn to_json(&self) -> VerifierResult<String> {
        serde_json::to_string_pretty(self).map_err(VerifierError::from)
    }

    /// Load from a JSON file.
    pub fn load(path: &std::path::Path) -> VerifierResult<Self> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_json(&contents)
    }

    /// Save to a JSON file.
    pub fn save(&self, path: &std::path::Path) -> VerifierResult<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Validate the configuration, returning an error on the first problem found.
    pub fn validate(&self) -> VerifierResult<()> {
        let mut errors = Vec::new();
        errors.extend(self.tier1.validate());
        errors.extend(self.tier2.validate());
        errors.extend(self.sampling.validate());
        errors.extend(self.smt.validate());
        errors.extend(self.population.validate());
        if !errors.is_empty() {
            return Err(VerifierError::Configuration(errors.join("; ")));
        }
        Ok(())
    }

    /// Collect all validation problems (non-fatal).
    pub fn validation_problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        problems.extend(self.tier1.validate());
        problems.extend(self.tier2.validate());
        problems.extend(self.sampling.validate());
        problems.extend(self.smt.validate());
        problems.extend(self.population.validate());
        problems
    }

    /// Return a builder initialized from the defaults.
    pub fn builder() -> VerifierConfigBuilder {
        VerifierConfigBuilder {
            config: Self::default(),
        }
    }

    /// Total number of sample points that will be drawn.
    pub fn total_samples(&self) -> usize {
        if self.sampling.use_stratified {
            self.sampling.strata_per_dim.pow(crate::NUM_BODY_PARAMS as u32)
        } else {
            self.sampling.num_samples
        }
    }

    /// Quick-check preset for development / CI.
    pub fn quick() -> Self {
        Self {
            name: "quick".into(),
            sampling: SamplingConfig {
                num_samples: 100,
                strata_per_dim: 3,
                ..SamplingConfig::default()
            },
            smt: SmtConfig {
                timeout_s: 5.0,
                ..SmtConfig::default()
            },
            ..Self::default()
        }
    }

    /// Thorough preset for production verification.
    pub fn thorough() -> Self {
        Self {
            name: "thorough".into(),
            sampling: SamplingConfig {
                num_samples: 10_000,
                strata_per_dim: 10,
                use_stratified: true,
                use_latin_hypercube: true,
                ..SamplingConfig::default()
            },
            smt: SmtConfig {
                timeout_s: 120.0,
                max_refinements: 10,
                ..SmtConfig::default()
            },
            tier1: Tier1Config {
                enabled: true,
                max_time_s: 600.0,
                ..Tier1Config::default()
            },
            tier2: Tier2Config {
                enabled: true,
                max_time_s: 1800.0,
                ..Tier2Config::default()
            },
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Tier configs
// ---------------------------------------------------------------------------

/// Configuration for Tier 1: sampling-based falsification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier1Config {
    /// Whether Tier 1 is enabled.
    pub enabled: bool,
    /// Maximum wall-clock time for the tier (seconds).
    pub max_time_s: f64,
    /// Number of parallel workers (0 = auto-detect).
    pub num_workers: usize,
    /// Whether to use adaptive sampling refinement.
    pub adaptive_refinement: bool,
    /// Stop early if any failure is found.
    pub stop_on_first_failure: bool,
    /// Minimum coverage κ target for Tier 1 alone.
    pub min_coverage: f64,
}

impl Default for Tier1Config {
    fn default() -> Self {
        Self {
            enabled: true,
            max_time_s: 60.0,
            num_workers: 0,
            adaptive_refinement: true,
            stop_on_first_failure: false,
            min_coverage: 0.90,
        }
    }
}

impl Tier1Config {
    /// Validate this config.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.max_time_s <= 0.0 {
            errors.push("tier1.max_time_s must be positive".into());
        }
        if self.min_coverage < 0.0 || self.min_coverage > 1.0 {
            errors.push("tier1.min_coverage must be in [0, 1]".into());
        }
        errors
    }
}

/// Configuration for Tier 2: SMT-based formal verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier2Config {
    /// Whether Tier 2 is enabled.
    pub enabled: bool,
    /// Maximum wall-clock time for the tier (seconds).
    pub max_time_s: f64,
    /// Maximum number of region subdivisions.
    pub max_subdivisions: usize,
    /// Minimum region volume before giving up subdivision.
    pub min_region_volume: f64,
    /// Whether to only run Tier 2 on regions not covered by Tier 1.
    pub residual_only: bool,
    /// Maximum linearization error to accept a proof.
    pub max_linearization_error: f64,
}

impl Default for Tier2Config {
    fn default() -> Self {
        Self {
            enabled: true,
            max_time_s: 300.0,
            max_subdivisions: 100,
            min_region_volume: 1e-8,
            residual_only: true,
            max_linearization_error: 0.01,
        }
    }
}

impl Tier2Config {
    /// Validate this config.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.max_time_s <= 0.0 {
            errors.push("tier2.max_time_s must be positive".into());
        }
        if self.max_subdivisions == 0 {
            errors.push("tier2.max_subdivisions must be > 0".into());
        }
        if self.min_region_volume <= 0.0 {
            errors.push("tier2.min_region_volume must be positive".into());
        }
        if self.max_linearization_error <= 0.0 {
            errors.push("tier2.max_linearization_error must be positive".into());
        }
        errors
    }
}

// ---------------------------------------------------------------------------
// Sampling config
// ---------------------------------------------------------------------------

/// Sampling parameters for Tier 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Number of samples for uniform (non-stratified) sampling.
    pub num_samples: usize,
    /// Number of strata per dimension for stratified sampling.
    pub strata_per_dim: usize,
    /// Confidence parameter δ (failure probability bound).
    pub confidence_delta: f64,
    /// Whether to use stratified sampling.
    pub use_stratified: bool,
    /// Whether to use Latin Hypercube sampling.
    pub use_latin_hypercube: bool,
    /// Random seed (0 = random).
    pub seed: u64,
    /// Maximum samples per stratum in adaptive mode.
    pub max_samples_per_stratum: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            strata_per_dim: 5,
            confidence_delta: 0.05,
            use_stratified: true,
            use_latin_hypercube: false,
            seed: 0,
            max_samples_per_stratum: 20,
        }
    }
}

impl SamplingConfig {
    /// Validate this config.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.num_samples == 0 {
            errors.push("sampling.num_samples must be > 0".into());
        }
        if self.strata_per_dim == 0 {
            errors.push("sampling.strata_per_dim must be > 0".into());
        }
        if self.confidence_delta <= 0.0 || self.confidence_delta >= 1.0 {
            errors.push("sampling.confidence_delta must be in (0, 1)".into());
        }
        errors
    }
}

// ---------------------------------------------------------------------------
// SMT config
// ---------------------------------------------------------------------------

/// SMT solver parameters for Tier 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtConfig {
    /// Per-query timeout in seconds.
    pub timeout_s: f64,
    /// Linearization error bound δ_lin for Taylor approximations.
    pub linearization_delta: f64,
    /// Maximum refinement iterations.
    pub max_refinements: usize,
    /// SMT logic to use.
    pub logic: String,
    /// Whether to use incremental solving.
    pub incremental: bool,
    /// Whether to produce unsat cores.
    pub produce_unsat_cores: bool,
    /// Solver binary path (empty = default).
    pub solver_path: String,
}

impl Default for SmtConfig {
    fn default() -> Self {
        Self {
            timeout_s: 30.0,
            linearization_delta: 0.001,
            max_refinements: 5,
            logic: "QF_NRA".into(),
            incremental: true,
            produce_unsat_cores: false,
            solver_path: String::new(),
        }
    }
}

impl SmtConfig {
    /// Validate this config.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.timeout_s <= 0.0 {
            errors.push("smt.timeout_s must be positive".into());
        }
        if self.linearization_delta <= 0.0 {
            errors.push("smt.linearization_delta must be positive".into());
        }
        if self.max_refinements == 0 {
            errors.push("smt.max_refinements must be > 0".into());
        }
        errors
    }
}

// ---------------------------------------------------------------------------
// Population config
// ---------------------------------------------------------------------------

/// Population / anthropometric parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationConfig {
    /// Lower percentile bound for the target population.
    pub percentile_low: f64,
    /// Upper percentile bound for the target population.
    pub percentile_high: f64,
    /// Target device names to verify against.
    pub target_devices: Vec<String>,
    /// Target movement modes.
    pub target_movement_modes: Vec<MovementMode>,
    /// Whether to include seated mode in verification.
    pub include_seated: bool,
    /// Whether to include standing mode in verification.
    pub include_standing: bool,
    /// Seat height range (min, max) in meters for seated mode.
    pub seat_height_range: (f64, f64),
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self {
            percentile_low: crate::DEFAULT_COVERAGE_LOW,
            percentile_high: crate::DEFAULT_COVERAGE_HIGH,
            target_devices: vec!["Meta Quest 3".into(), "Apple Vision Pro".into()],
            target_movement_modes: vec![MovementMode::Seated, MovementMode::Standing],
            include_seated: true,
            include_standing: true,
            seat_height_range: (0.40, 0.55),
        }
    }
}

impl PopulationConfig {
    /// Validate this config.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.percentile_low < 0.0 || self.percentile_low > 1.0 {
            errors.push("population.percentile_low must be in [0, 1]".into());
        }
        if self.percentile_high < 0.0 || self.percentile_high > 1.0 {
            errors.push("population.percentile_high must be in [0, 1]".into());
        }
        if self.percentile_low >= self.percentile_high {
            errors.push("population.percentile_low must be < percentile_high".into());
        }
        if self.seat_height_range.0 >= self.seat_height_range.1 {
            errors.push("population.seat_height_range min must be < max".into());
        }
        errors
    }

    /// Percentile range as a fraction of the total population.
    pub fn coverage_fraction(&self) -> f64 {
        self.percentile_high - self.percentile_low
    }
}

// ---------------------------------------------------------------------------
// Output config
// ---------------------------------------------------------------------------

/// Output / reporting settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory for reports.
    pub output_dir: PathBuf,
    /// Whether to generate JSON reports.
    pub json: bool,
    /// Whether to generate plain-text reports.
    pub text: bool,
    /// Whether to generate HTML reports.
    pub html: bool,
    /// Whether to generate SVG diagrams.
    pub svg_diagrams: bool,
    /// Verbosity level (0 = silent, 1 = summary, 2 = detailed, 3 = debug).
    pub verbosity: u8,
    /// Whether to include raw sample data in reports.
    pub include_samples: bool,
    /// Whether to include timing information.
    pub include_timing: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./output"),
            json: true,
            text: true,
            html: false,
            svg_diagrams: false,
            verbosity: 1,
            include_samples: false,
            include_timing: true,
        }
    }
}

// ---------------------------------------------------------------------------
// General config
// ---------------------------------------------------------------------------

/// General verifier flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Maximum interaction depth for multi-step sequences.
    pub max_interaction_depth: usize,
    /// Whether to check all elements or stop at first failure.
    pub check_all_elements: bool,
    /// Whether to run Tier 2 even if Tier 1 achieves full coverage.
    pub force_tier2: bool,
    /// Whether to enable parallel processing.
    pub parallel: bool,
    /// Number of threads (0 = auto).
    pub num_threads: usize,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            max_interaction_depth: crate::MAX_INTERACTION_DEPTH,
            check_all_elements: true,
            force_tier2: false,
            parallel: true,
            num_threads: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for [`VerifierConfig`].
#[derive(Debug, Clone)]
pub struct VerifierConfigBuilder {
    config: VerifierConfig,
}

impl VerifierConfigBuilder {
    /// Set the configuration name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    /// Set the number of samples.
    pub fn num_samples(mut self, n: usize) -> Self {
        self.config.sampling.num_samples = n;
        self
    }

    /// Set the number of strata per dimension.
    pub fn strata_per_dim(mut self, n: usize) -> Self {
        self.config.sampling.strata_per_dim = n;
        self
    }

    /// Set the confidence parameter δ.
    pub fn confidence_delta(mut self, delta: f64) -> Self {
        self.config.sampling.confidence_delta = delta;
        self
    }

    /// Set the SMT timeout.
    pub fn smt_timeout(mut self, timeout_s: f64) -> Self {
        self.config.smt.timeout_s = timeout_s;
        self
    }

    /// Set the linearization delta.
    pub fn linearization_delta(mut self, delta: f64) -> Self {
        self.config.smt.linearization_delta = delta;
        self
    }

    /// Set the population percentile range.
    pub fn percentile_range(mut self, low: f64, high: f64) -> Self {
        self.config.population.percentile_low = low;
        self.config.population.percentile_high = high;
        self
    }

    /// Add a target device name.
    pub fn target_device(mut self, device: impl Into<String>) -> Self {
        self.config.population.target_devices.push(device.into());
        self
    }

    /// Set the output directory.
    pub fn output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.output.output_dir = dir.into();
        self
    }

    /// Enable/disable Tier 1.
    pub fn enable_tier1(mut self, enabled: bool) -> Self {
        self.config.tier1.enabled = enabled;
        self
    }

    /// Enable/disable Tier 2.
    pub fn enable_tier2(mut self, enabled: bool) -> Self {
        self.config.tier2.enabled = enabled;
        self
    }

    /// Set verbosity.
    pub fn verbosity(mut self, v: u8) -> Self {
        self.config.output.verbosity = v;
        self
    }

    /// Enable stratified sampling.
    pub fn stratified(mut self, enabled: bool) -> Self {
        self.config.sampling.use_stratified = enabled;
        self
    }

    /// Enable Latin Hypercube sampling.
    pub fn latin_hypercube(mut self, enabled: bool) -> Self {
        self.config.sampling.use_latin_hypercube = enabled;
        self
    }

    /// Set the random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.sampling.seed = seed;
        self
    }

    /// Set parallel processing.
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.config.general.parallel = enabled;
        self
    }

    /// Build and validate the config.
    pub fn build(self) -> VerifierResult<VerifierConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without validation (for testing).
    pub fn build_unchecked(self) -> VerifierConfig {
        self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validates() {
        let cfg = VerifierConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_quick_preset() {
        let cfg = VerifierConfig::quick();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.sampling.num_samples, 100);
        assert_eq!(cfg.name, "quick");
    }

    #[test]
    fn test_thorough_preset() {
        let cfg = VerifierConfig::thorough();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.sampling.num_samples, 10_000);
        assert!(cfg.sampling.use_latin_hypercube);
    }

    #[test]
    fn test_json_roundtrip() {
        let cfg = VerifierConfig::default();
        let json = cfg.to_json().unwrap();
        let back = VerifierConfig::from_json(&json).unwrap();
        assert_eq!(cfg.name, back.name);
        assert_eq!(cfg.sampling.num_samples, back.sampling.num_samples);
        assert_eq!(cfg.smt.timeout_s, back.smt.timeout_s);
    }

    #[test]
    fn test_builder() {
        let cfg = VerifierConfig::builder()
            .name("test")
            .num_samples(500)
            .confidence_delta(0.01)
            .smt_timeout(60.0)
            .percentile_range(0.05, 0.95)
            .verbosity(2)
            .build()
            .unwrap();

        assert_eq!(cfg.name, "test");
        assert_eq!(cfg.sampling.num_samples, 500);
        assert!((cfg.sampling.confidence_delta - 0.01).abs() < 1e-12);
        assert!((cfg.smt.timeout_s - 60.0).abs() < 1e-12);
    }

    #[test]
    fn test_builder_stratified() {
        let cfg = VerifierConfig::builder()
            .stratified(true)
            .strata_per_dim(4)
            .latin_hypercube(true)
            .build()
            .unwrap();
        assert!(cfg.sampling.use_stratified);
        assert!(cfg.sampling.use_latin_hypercube);
        assert_eq!(cfg.sampling.strata_per_dim, 4);
    }

    #[test]
    fn test_validation_bad_delta() {
        let cfg = VerifierConfig::builder()
            .confidence_delta(0.0)
            .build_unchecked();
        let problems = cfg.validation_problems();
        assert!(!problems.is_empty());
        assert!(problems.iter().any(|p| p.contains("confidence_delta")));
    }

    #[test]
    fn test_validation_bad_percentile() {
        let cfg = VerifierConfig::builder()
            .percentile_range(0.95, 0.05)
            .build_unchecked();
        let problems = cfg.validation_problems();
        assert!(problems.iter().any(|p| p.contains("percentile")));
    }

    #[test]
    fn test_validation_bad_smt_timeout() {
        let mut cfg = VerifierConfig::default();
        cfg.smt.timeout_s = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validation_bad_tier1_time() {
        let mut cfg = VerifierConfig::default();
        cfg.tier1.max_time_s = 0.0;
        let problems = cfg.tier1.validate();
        assert!(!problems.is_empty());
    }

    #[test]
    fn test_validation_bad_tier2() {
        let mut cfg = VerifierConfig::default();
        cfg.tier2.max_subdivisions = 0;
        let problems = cfg.tier2.validate();
        assert!(!problems.is_empty());
    }

    #[test]
    fn test_population_coverage_fraction() {
        let pop = PopulationConfig::default();
        assert!((pop.coverage_fraction() - 0.90).abs() < 1e-12);
    }

    #[test]
    fn test_total_samples_uniform() {
        let mut cfg = VerifierConfig::default();
        cfg.sampling.use_stratified = false;
        cfg.sampling.num_samples = 1234;
        assert_eq!(cfg.total_samples(), 1234);
    }

    #[test]
    fn test_total_samples_stratified() {
        let mut cfg = VerifierConfig::default();
        cfg.sampling.use_stratified = true;
        cfg.sampling.strata_per_dim = 3;
        // 3^5 = 243
        assert_eq!(cfg.total_samples(), 243);
    }

    #[test]
    fn test_output_config_default() {
        let out = OutputConfig::default();
        assert!(out.json);
        assert!(out.text);
        assert!(!out.html);
    }

    #[test]
    fn test_general_config_default() {
        let gen = GeneralConfig::default();
        assert_eq!(gen.max_interaction_depth, crate::MAX_INTERACTION_DEPTH);
        assert!(gen.check_all_elements);
    }

    #[test]
    fn test_smt_config_validate() {
        let smt = SmtConfig {
            timeout_s: 0.0,
            linearization_delta: -1.0,
            max_refinements: 0,
            ..SmtConfig::default()
        };
        let problems = smt.validate();
        assert_eq!(problems.len(), 3);
    }

    #[test]
    fn test_builder_build_unchecked() {
        let cfg = VerifierConfig::builder()
            .name("bad")
            .confidence_delta(5.0)
            .build_unchecked();
        assert_eq!(cfg.name, "bad");
    }

    #[test]
    fn test_builder_seed_and_parallel() {
        let cfg = VerifierConfig::builder()
            .seed(42)
            .parallel(false)
            .build()
            .unwrap();
        assert_eq!(cfg.sampling.seed, 42);
        assert!(!cfg.general.parallel);
    }
}
