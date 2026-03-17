//! Configuration types and validation for the CollusionProof system.
//!
//! Includes global configuration, tiered null hypothesis parameters,
//! bootstrap/Monte Carlo settings, significance levels, and
//! TOML serialization/deserialization.

use crate::types::EvaluationMode;
use serde::{Deserialize, Serialize};
use std::fmt;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Significance and confidence levels
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A significance level α ∈ (0, 1).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SignificanceLevel(f64);

impl SignificanceLevel {
    /// Standard α = 0.05.
    pub const STANDARD: Self = SignificanceLevel(0.05);
    /// Strict α = 0.01.
    pub const STRICT: Self = SignificanceLevel(0.01);
    /// Relaxed α = 0.10.
    pub const RELAXED: Self = SignificanceLevel(0.10);

    pub fn new(alpha: f64) -> Result<Self, String> {
        if alpha <= 0.0 || alpha >= 1.0 {
            Err(format!("significance level must be in (0, 1), got {}", alpha))
        } else {
            Ok(SignificanceLevel(alpha))
        }
    }

    /// Create without validation (for const contexts).
    pub const fn new_unchecked(alpha: f64) -> Self {
        SignificanceLevel(alpha)
    }

    pub fn value(&self) -> f64 { self.0 }

    /// Corresponding confidence level (1 − α).
    pub fn confidence(&self) -> ConfidenceLevel {
        ConfidenceLevel(1.0 - self.0)
    }
}

impl Default for SignificanceLevel {
    fn default() -> Self { Self::STANDARD }
}

impl fmt::Display for SignificanceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "α={:.4}", self.0)
    }
}

/// A confidence level (1 − α) ∈ (0, 1).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ConfidenceLevel(f64);

impl ConfidenceLevel {
    pub const NINETY: Self = ConfidenceLevel(0.90);
    pub const NINETY_FIVE: Self = ConfidenceLevel(0.95);
    pub const NINETY_NINE: Self = ConfidenceLevel(0.99);

    pub fn new(level: f64) -> Result<Self, String> {
        if level <= 0.0 || level >= 1.0 {
            Err(format!("confidence level must be in (0, 1), got {}", level))
        } else {
            Ok(ConfidenceLevel(level))
        }
    }

    pub const fn new_unchecked(level: f64) -> Self {
        ConfidenceLevel(level)
    }

    pub fn value(&self) -> f64 { self.0 }

    pub fn significance(&self) -> SignificanceLevel {
        SignificanceLevel(1.0 - self.0)
    }
}

impl Default for ConfidenceLevel {
    fn default() -> Self { Self::NINETY_FIVE }
}

impl fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}% CI", self.0 * 100.0)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tiered null configuration
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Parameters for a single null-hypothesis tier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NullTierParams {
    pub name: String,
    pub collusion_index_threshold: f64,
    pub significance_level: SignificanceLevel,
    pub min_effect_size: f64,
    pub description: String,
}

/// Tiered null hypothesis configuration: narrow, medium, and broad tiers
/// with progressively stricter evidence requirements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TieredNullConfig {
    pub narrow: NullTierParams,
    pub medium: NullTierParams,
    pub broad: NullTierParams,
}

impl TieredNullConfig {
    /// Standard configuration with three tiers.
    pub fn standard() -> Self {
        TieredNullConfig {
            narrow: NullTierParams {
                name: "narrow".into(),
                collusion_index_threshold: 0.8,
                significance_level: SignificanceLevel::STRICT,
                min_effect_size: 0.5,
                description: "Strong collusion: CI > 0.8".into(),
            },
            medium: NullTierParams {
                name: "medium".into(),
                collusion_index_threshold: 0.5,
                significance_level: SignificanceLevel::STANDARD,
                min_effect_size: 0.3,
                description: "Moderate collusion: CI > 0.5".into(),
            },
            broad: NullTierParams {
                name: "broad".into(),
                collusion_index_threshold: 0.2,
                significance_level: SignificanceLevel::RELAXED,
                min_effect_size: 0.1,
                description: "Weak collusion: CI > 0.2".into(),
            },
        }
    }

    /// Return tiers as a slice for iteration.
    pub fn tiers(&self) -> Vec<&NullTierParams> {
        vec![&self.narrow, &self.medium, &self.broad]
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        for tier in self.tiers() {
            if tier.collusion_index_threshold < 0.0 || tier.collusion_index_threshold > 1.0 {
                errors.push(format!("tier '{}': CI threshold out of [0,1]", tier.name));
            }
            if tier.min_effect_size < 0.0 {
                errors.push(format!("tier '{}': negative effect size", tier.name));
            }
        }
        if self.narrow.collusion_index_threshold <= self.medium.collusion_index_threshold {
            errors.push("narrow CI threshold should exceed medium".into());
        }
        if self.medium.collusion_index_threshold <= self.broad.collusion_index_threshold {
            errors.push("medium CI threshold should exceed broad".into());
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

impl Default for TieredNullConfig {
    fn default() -> Self { Self::standard() }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Bootstrap & Monte Carlo configuration
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Configuration for bootstrap resampling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapConfig {
    pub num_iterations: usize,
    pub confidence_level: ConfidenceLevel,
    pub method: BootstrapMethod,
    pub block_size: Option<usize>,
    pub seed: Option<u64>,
}

/// Bootstrap method variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BootstrapMethod {
    /// Standard i.i.d. resampling.
    Nonparametric,
    /// Block bootstrap for dependent data.
    Block,
    /// Bias-corrected and accelerated bootstrap.
    BCa,
    /// Circular block bootstrap for periodic data.
    CircularBlock,
}

impl BootstrapConfig {
    pub fn new(num_iterations: usize, confidence_level: ConfidenceLevel) -> Self {
        BootstrapConfig {
            num_iterations,
            confidence_level,
            method: BootstrapMethod::Nonparametric,
            block_size: None,
            seed: None,
        }
    }

    pub fn for_mode(mode: EvaluationMode) -> Self {
        let n = mode.bootstrap_iterations();
        BootstrapConfig::new(n, ConfidenceLevel::NINETY_FIVE)
    }

    pub fn with_method(mut self, method: BootstrapMethod) -> Self {
        self.method = method; self
    }

    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = Some(size);
        self.method = BootstrapMethod::Block;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed); self
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.num_iterations == 0 {
            errors.push("bootstrap iterations must be > 0".into());
        }
        if let Some(bs) = self.block_size {
            if bs == 0 {
                errors.push("block size must be > 0".into());
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

impl Default for BootstrapConfig {
    fn default() -> Self { Self::for_mode(EvaluationMode::Standard) }
}

/// Configuration for Monte Carlo simulations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonteCarloConfig {
    pub num_iterations: usize,
    pub burn_in: usize,
    pub thinning: usize,
    pub seed: Option<u64>,
    pub parallel: bool,
}

impl MonteCarloConfig {
    pub fn new(num_iterations: usize) -> Self {
        MonteCarloConfig {
            num_iterations,
            burn_in: num_iterations / 10,
            thinning: 1,
            seed: None,
            parallel: true,
        }
    }

    pub fn for_mode(mode: EvaluationMode) -> Self {
        Self::new(mode.monte_carlo_iterations())
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed); self
    }

    pub fn effective_samples(&self) -> usize {
        if self.thinning == 0 { return 0; }
        self.num_iterations.saturating_sub(self.burn_in) / self.thinning
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.num_iterations == 0 {
            errors.push("MC iterations must be > 0".into());
        }
        if self.burn_in >= self.num_iterations {
            errors.push("burn-in must be < num_iterations".into());
        }
        if self.thinning == 0 {
            errors.push("thinning must be > 0".into());
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

impl Default for MonteCarloConfig {
    fn default() -> Self { Self::for_mode(EvaluationMode::Standard) }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Global configuration
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Top-level configuration for the entire certification pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlobalConfig {
    pub evaluation_mode: EvaluationMode,
    pub tiered_null: TieredNullConfig,
    pub bootstrap: BootstrapConfig,
    pub monte_carlo: MonteCarloConfig,
    pub significance_level: SignificanceLevel,
    pub confidence_level: ConfidenceLevel,
    pub max_parallel_sims: usize,
    pub output_dir: String,
    pub verbose: bool,
    pub random_seed: u64,
}

impl GlobalConfig {
    /// Create a configuration preset for the given evaluation mode.
    pub fn for_mode(mode: EvaluationMode) -> Self {
        GlobalConfig {
            evaluation_mode: mode,
            tiered_null: TieredNullConfig::standard(),
            bootstrap: BootstrapConfig::for_mode(mode),
            monte_carlo: MonteCarloConfig::for_mode(mode),
            significance_level: SignificanceLevel::STANDARD,
            confidence_level: ConfidenceLevel::NINETY_FIVE,
            max_parallel_sims: num_cpus(),
            output_dir: "output".into(),
            verbose: false,
            random_seed: 42,
        }
    }

    pub fn smoke() -> Self { Self::for_mode(EvaluationMode::Smoke) }
    pub fn standard() -> Self { Self::for_mode(EvaluationMode::Standard) }
    pub fn full() -> Self { Self::for_mode(EvaluationMode::Full) }

    /// Validate all nested configurations.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut all_errors = Vec::new();

        if let Err(e) = self.tiered_null.validate() {
            all_errors.extend(e.into_iter().map(|s| format!("tiered_null: {}", s)));
        }
        if let Err(e) = self.bootstrap.validate() {
            all_errors.extend(e.into_iter().map(|s| format!("bootstrap: {}", s)));
        }
        if let Err(e) = self.monte_carlo.validate() {
            all_errors.extend(e.into_iter().map(|s| format!("monte_carlo: {}", s)));
        }
        if self.max_parallel_sims == 0 {
            all_errors.push("max_parallel_sims must be > 0".into());
        }

        if all_errors.is_empty() { Ok(()) } else { Err(all_errors) }
    }

    /// Serialize to TOML string.
    pub fn to_toml(&self) -> Result<String, String> {
        toml::to_string_pretty(self).map_err(|e| format!("TOML serialization failed: {}", e))
    }

    /// Deserialize from TOML string.
    pub fn from_toml(s: &str) -> Result<Self, String> {
        toml::from_str(s).map_err(|e| format!("TOML parse failed: {}", e))
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("JSON serialization failed: {}", e))
    }

    /// Deserialize from JSON string.
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| format!("JSON parse failed: {}", e))
    }
}

impl Default for GlobalConfig {
    fn default() -> Self { Self::standard() }
}

impl fmt::Display for GlobalConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GlobalConfig(mode={}, α={}, bootstrap={}, mc={})",
            self.evaluation_mode, self.significance_level,
            self.bootstrap.num_iterations, self.monte_carlo.num_iterations)
    }
}

/// Estimate number of CPUs (fallback to 4).
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_significance_level_valid() {
        assert!(SignificanceLevel::new(0.05).is_ok());
        assert!(SignificanceLevel::new(0.0).is_err());
        assert!(SignificanceLevel::new(1.0).is_err());
        assert!(SignificanceLevel::new(-0.1).is_err());
    }

    #[test]
    fn test_confidence_significance_round_trip() {
        let alpha = SignificanceLevel::STANDARD;
        let conf = alpha.confidence();
        assert!((conf.value() - 0.95).abs() < 1e-10);
        let alpha2 = conf.significance();
        assert!((alpha2.value() - alpha.value()).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_level_valid() {
        assert!(ConfidenceLevel::new(0.95).is_ok());
        assert!(ConfidenceLevel::new(0.0).is_err());
        assert!(ConfidenceLevel::new(1.0).is_err());
    }

    #[test]
    fn test_tiered_null_standard() {
        let tn = TieredNullConfig::standard();
        assert!(tn.validate().is_ok());
        assert_eq!(tn.tiers().len(), 3);
    }

    #[test]
    fn test_tiered_null_invalid() {
        let mut tn = TieredNullConfig::standard();
        tn.narrow.collusion_index_threshold = 0.1; // less than medium
        assert!(tn.validate().is_err());
    }

    #[test]
    fn test_bootstrap_config_for_mode() {
        let smoke = BootstrapConfig::for_mode(EvaluationMode::Smoke);
        let full = BootstrapConfig::for_mode(EvaluationMode::Full);
        assert!(smoke.num_iterations < full.num_iterations);
    }

    #[test]
    fn test_bootstrap_validation() {
        let mut bc = BootstrapConfig::default();
        assert!(bc.validate().is_ok());
        bc.num_iterations = 0;
        assert!(bc.validate().is_err());
    }

    #[test]
    fn test_monte_carlo_effective_samples() {
        let mc = MonteCarloConfig { num_iterations: 1000, burn_in: 100, thinning: 2, seed: None, parallel: true };
        assert_eq!(mc.effective_samples(), 450);
    }

    #[test]
    fn test_monte_carlo_validation() {
        let mut mc = MonteCarloConfig::default();
        assert!(mc.validate().is_ok());
        mc.thinning = 0;
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_global_config_smoke() {
        let cfg = GlobalConfig::smoke();
        assert_eq!(cfg.evaluation_mode, EvaluationMode::Smoke);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_global_config_standard() {
        let cfg = GlobalConfig::standard();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_global_config_full() {
        let cfg = GlobalConfig::full();
        assert_eq!(cfg.evaluation_mode, EvaluationMode::Full);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_global_config_toml_roundtrip() {
        let cfg = GlobalConfig::standard();
        let toml_str = cfg.to_toml().unwrap();
        let cfg2 = GlobalConfig::from_toml(&toml_str).unwrap();
        assert_eq!(cfg, cfg2);
    }

    #[test]
    fn test_global_config_json_roundtrip() {
        let cfg = GlobalConfig::smoke();
        let json = cfg.to_json().unwrap();
        let cfg2 = GlobalConfig::from_json(&json).unwrap();
        assert_eq!(cfg, cfg2);
    }

    #[test]
    fn test_global_config_display() {
        let cfg = GlobalConfig::standard();
        let s = format!("{}", cfg);
        assert!(s.contains("Standard"));
    }

    #[test]
    fn test_bootstrap_with_block() {
        let bc = BootstrapConfig::default().with_block_size(10);
        assert_eq!(bc.method, BootstrapMethod::Block);
        assert_eq!(bc.block_size, Some(10));
    }
}
