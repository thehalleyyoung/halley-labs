//! Configuration types for the conservation analysis pipeline.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level configuration for a ConservationLint analysis run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Input source file or directory.
    pub input_path: PathBuf,
    /// Output directory for reports.
    pub output_dir: PathBuf,
    /// Which conservation laws to check.
    pub laws_to_check: Vec<LawConfig>,
    /// Maximum BCH expansion order.
    pub max_bch_order: u32,
    /// Tolerance settings.
    pub tolerance: ToleranceConfig,
    /// Whether to perform causal localization.
    pub enable_localization: bool,
    /// Whether to compute obstruction certificates.
    pub enable_obstruction: bool,
    /// Parallelism settings.
    pub parallelism: ParallelismConfig,
    /// Output format settings.
    pub output: OutputConfig,
    /// Logging verbosity level.
    pub verbosity: Verbosity,
    /// Tier selection strategy.
    pub tier_strategy: TierStrategy,
}

/// Configuration for a specific conservation law to check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LawConfig {
    pub kind: LawKind,
    pub tolerance: Option<f64>,
    pub enabled: bool,
    pub custom_expression: Option<String>,
}

/// Kinds of conservation laws the tool can check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LawKind {
    Energy,
    LinearMomentum,
    AngularMomentum,
    Mass,
    Charge,
    CenterOfMass,
    SymplecticStructure,
    Vorticity,
    Entropy,
    Baryon,
    Lepton,
    Custom,
}

impl LawKind {
    /// Return all standard conservation law kinds.
    pub fn all_standard() -> Vec<Self> {
        vec![
            Self::Energy,
            Self::LinearMomentum,
            Self::AngularMomentum,
            Self::Mass,
            Self::Charge,
        ]
    }

    /// Human-readable name.
    pub fn display_name(&self) -> &str {
        match self {
            Self::Energy => "Energy",
            Self::LinearMomentum => "Linear Momentum",
            Self::AngularMomentum => "Angular Momentum",
            Self::Mass => "Mass",
            Self::Charge => "Charge",
            Self::CenterOfMass => "Center of Mass",
            Self::SymplecticStructure => "Symplectic Structure",
            Self::Vorticity => "Vorticity",
            Self::Entropy => "Entropy",
            Self::Baryon => "Baryon Number",
            Self::Lepton => "Lepton Number",
            Self::Custom => "Custom",
        }
    }

    /// The associated Noether symmetry generator.
    pub fn associated_symmetry(&self) -> &str {
        match self {
            Self::Energy => "Time translation",
            Self::LinearMomentum => "Spatial translation",
            Self::AngularMomentum => "Rotation",
            Self::Mass => "Phase symmetry",
            Self::Charge => "U(1) gauge symmetry",
            Self::CenterOfMass => "Galilean boost",
            Self::SymplecticStructure => "Canonical transformation",
            Self::Vorticity => "Particle relabeling",
            Self::Entropy => "Adiabatic invariance",
            Self::Baryon => "U(1)_B symmetry",
            Self::Lepton => "U(1)_L symmetry",
            Self::Custom => "User-defined",
        }
    }
}

impl std::fmt::Display for LawKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Tolerance configuration for numerical checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceConfig {
    /// Absolute tolerance for conservation violation detection.
    pub absolute: f64,
    /// Relative tolerance for conservation violation detection.
    pub relative: f64,
    /// Tolerance for drift rate detection (per time step).
    pub drift_rate: f64,
    /// Number of standard deviations for statistical significance.
    pub significance_sigma: f64,
    /// Minimum number of timesteps for statistical tests.
    pub min_timesteps: usize,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            absolute: 1e-10,
            relative: 1e-8,
            drift_rate: 1e-12,
            significance_sigma: 3.0,
            min_timesteps: 100,
        }
    }
}

/// Parallelism configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelismConfig {
    /// Number of threads for parallel analysis.
    pub num_threads: usize,
    /// Whether to use SIMD for vector operations.
    pub enable_simd: bool,
    /// Chunk size for parallel work distribution.
    pub chunk_size: usize,
}

impl Default for ParallelismConfig {
    fn default() -> Self {
        Self {
            num_threads: 0, // 0 = auto-detect
            enable_simd: true,
            chunk_size: 1024,
        }
    }
}

/// Output format configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output format.
    pub format: OutputFormat,
    /// Whether to include source snippets in diagnostics.
    pub include_source: bool,
    /// Whether to include mathematical derivations.
    pub include_derivations: bool,
    /// Maximum number of violations to report.
    pub max_violations: usize,
    /// Whether to generate SARIF output.
    pub sarif: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Text,
            include_source: true,
            include_derivations: false,
            max_violations: 100,
            sarif: false,
        }
    }
}

/// Output format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Text,
    Json,
    Sarif,
    Html,
    Markdown,
    Csv,
}

/// Verbosity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verbosity {
    Quiet,
    Normal,
    Verbose,
    Debug,
    Trace,
}

impl Default for Verbosity {
    fn default() -> Self {
        Self::Normal
    }
}

/// Tier selection strategy for analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TierStrategy {
    /// Always try Tier 1 (static) first, fall back to Tier 2 (dynamic).
    StaticFirst,
    /// Always use Tier 2 (dynamic) analysis.
    DynamicOnly,
    /// Always use Tier 1 (static) analysis; fail if extraction fails.
    StaticOnly,
    /// Let the system choose based on code complexity.
    Auto,
}

impl Default for TierStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            input_path: PathBuf::from("."),
            output_dir: PathBuf::from("./conservation-report"),
            laws_to_check: LawKind::all_standard()
                .into_iter()
                .map(|kind| LawConfig {
                    kind,
                    tolerance: None,
                    enabled: true,
                    custom_expression: None,
                })
                .collect(),
            max_bch_order: 3,
            tolerance: ToleranceConfig::default(),
            enable_localization: true,
            enable_obstruction: true,
            parallelism: ParallelismConfig::default(),
            output: OutputConfig::default(),
            verbosity: Verbosity::Normal,
            tier_strategy: TierStrategy::Auto,
        }
    }
}

impl AnalysisConfig {
    /// Load configuration from a TOML file.
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }

    /// Serialize configuration to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.tolerance.absolute <= 0.0 {
            errors.push("Absolute tolerance must be positive".to_string());
        }
        if self.tolerance.relative <= 0.0 {
            errors.push("Relative tolerance must be positive".to_string());
        }
        if self.max_bch_order == 0 || self.max_bch_order > 8 {
            errors.push("BCH order must be between 1 and 8".to_string());
        }
        if self.laws_to_check.is_empty() {
            errors.push("At least one conservation law must be specified".to_string());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AnalysisConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.laws_to_check.len(), 5);
    }

    #[test]
    fn test_law_kind_display() {
        assert_eq!(LawKind::Energy.display_name(), "Energy");
        assert_eq!(LawKind::AngularMomentum.display_name(), "Angular Momentum");
    }

    #[test]
    fn test_config_serialization() {
        let config = AnalysisConfig::default();
        let json = config.to_json().unwrap();
        let restored: AnalysisConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.laws_to_check.len(), config.laws_to_check.len());
    }

    #[test]
    fn test_invalid_config() {
        let mut config = AnalysisConfig::default();
        config.tolerance.absolute = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_law_kind_symmetry() {
        assert_eq!(LawKind::Energy.associated_symmetry(), "Time translation");
        assert_eq!(LawKind::LinearMomentum.associated_symmetry(), "Spatial translation");
    }
}
