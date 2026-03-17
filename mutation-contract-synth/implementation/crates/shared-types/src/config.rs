//! Configuration types for the MutSpec system.
//!
//! All settings are loaded from TOML and exposed through [`MutSpecConfig`].
//! Sub-configs cover mutation, synthesis, SMT, analysis, coverage, and output.

use std::collections::HashSet;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::errors::MutSpecError;
use crate::operators::MutationOperator;

// ---------------------------------------------------------------------------
// MutationConfig
// ---------------------------------------------------------------------------

/// Configuration for the mutation phase.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MutationConfig {
    /// Which mutation operators to enable.
    pub operators: Vec<String>,
    /// Maximum number of mutants generated per mutation site.
    pub max_mutants_per_site: usize,
    /// Timeout (in seconds) for generating mutants for a single function.
    pub generation_timeout_secs: u64,
    /// Whether to apply higher-order mutations (multiple operators per mutant).
    pub higher_order: bool,
    /// Maximum order for higher-order mutations.
    pub max_order: u32,
    /// Skip equivalent mutant candidates using trivial checks.
    pub skip_trivial_equivalents: bool,
}

impl MutationConfig {
    pub fn enabled_operators(&self) -> Vec<MutationOperator> {
        self.operators
            .iter()
            .filter_map(|s| MutationOperator::from_mnemonic(s))
            .collect()
    }

    pub fn is_operator_enabled(&self, op: &MutationOperator) -> bool {
        self.operators
            .iter()
            .any(|s| MutationOperator::from_mnemonic(s) == Some(*op))
    }

    pub fn validate(&self) -> Result<(), MutSpecError> {
        if self.operators.is_empty() {
            return Err(MutSpecError::config("no mutation operators enabled"));
        }
        for op_name in &self.operators {
            if MutationOperator::from_mnemonic(op_name).is_none() {
                return Err(MutSpecError::config_key(
                    format!("unknown mutation operator: {op_name}"),
                    "mutation.operators",
                ));
            }
        }
        if self.max_mutants_per_site == 0 {
            return Err(MutSpecError::config_key(
                "max_mutants_per_site must be > 0",
                "mutation.max_mutants_per_site",
            ));
        }
        Ok(())
    }
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            operators: MutationOperator::standard_set()
                .iter()
                .map(|op| op.mnemonic().to_string())
                .collect(),
            max_mutants_per_site: 10,
            generation_timeout_secs: 30,
            higher_order: false,
            max_order: 1,
            skip_trivial_equivalents: true,
        }
    }
}

// ---------------------------------------------------------------------------
// SynthesisConfig
// ---------------------------------------------------------------------------

/// Configuration for contract synthesis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynthesisConfig {
    /// Which tiers to attempt (1, 2, 3).
    pub enabled_tiers: Vec<u8>,
    /// Maximum time per tier in seconds.
    pub tier_timeout_secs: u64,
    /// Maximum total synthesis time in seconds.
    pub total_timeout_secs: u64,
    /// Lattice walk: maximum number of refinement steps.
    pub lattice_max_steps: u32,
    /// Lattice walk: widening threshold.
    pub lattice_widening_threshold: u32,
    /// Template: maximum number of template variables.
    pub template_max_vars: u32,
    /// Template: maximum constant magnitude in templates.
    pub template_max_const: i64,
    /// Whether to attempt to minimise contracts.
    pub minimise_contracts: bool,
    /// Whether to verify synthesised contracts with an independent solver call.
    pub verify_contracts: bool,
}

impl SynthesisConfig {
    pub fn is_tier_enabled(&self, tier: u8) -> bool {
        self.enabled_tiers.contains(&tier)
    }

    pub fn validate(&self) -> Result<(), MutSpecError> {
        if self.enabled_tiers.is_empty() {
            return Err(MutSpecError::config("no synthesis tiers enabled"));
        }
        let valid: HashSet<u8> = [1, 2, 3].iter().copied().collect();
        for t in &self.enabled_tiers {
            if !valid.contains(t) {
                return Err(MutSpecError::config_key(
                    format!("invalid synthesis tier: {t}"),
                    "synthesis.enabled_tiers",
                ));
            }
        }
        if self.tier_timeout_secs == 0 || self.total_timeout_secs == 0 {
            return Err(MutSpecError::config("synthesis timeouts must be > 0"));
        }
        Ok(())
    }
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            enabled_tiers: vec![1, 2, 3],
            tier_timeout_secs: 60,
            total_timeout_secs: 300,
            lattice_max_steps: 100,
            lattice_widening_threshold: 10,
            template_max_vars: 4,
            template_max_const: 100,
            minimise_contracts: true,
            verify_contracts: true,
        }
    }
}

// ---------------------------------------------------------------------------
// SmtConfig
// ---------------------------------------------------------------------------

/// Configuration for SMT solver interaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmtConfig {
    /// Path to the SMT solver binary.
    pub solver_path: PathBuf,
    /// Per-query timeout in seconds.
    pub timeout_secs: u64,
    /// Use incremental solving (push/pop) when possible.
    pub incremental: bool,
    /// Memory limit for the solver in MB.
    pub memory_limit_mb: u64,
    /// SMT-LIB logic to use.
    pub logic: String,
    /// Additional solver flags.
    pub extra_flags: Vec<String>,
    /// Whether to dump SMT queries for debugging.
    pub dump_queries: bool,
    /// Directory for dumped queries.
    pub dump_dir: Option<PathBuf>,
}

impl SmtConfig {
    pub fn validate(&self) -> Result<(), MutSpecError> {
        if self.timeout_secs == 0 {
            return Err(MutSpecError::config_key(
                "solver timeout must be > 0",
                "smt.timeout_secs",
            ));
        }
        if self.memory_limit_mb == 0 {
            return Err(MutSpecError::config_key(
                "solver memory limit must be > 0",
                "smt.memory_limit_mb",
            ));
        }
        Ok(())
    }

    /// Build the command-line arguments for the solver.
    pub fn solver_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        if self.incremental {
            args.push("-in".to_string());
        }
        args.push(format!("-T:{}", self.timeout_secs));
        args.push(format!("-memory:{}", self.memory_limit_mb));
        args.extend(self.extra_flags.clone());
        args
    }
}

impl Default for SmtConfig {
    fn default() -> Self {
        Self {
            solver_path: PathBuf::from("z3"),
            timeout_secs: 30,
            incremental: true,
            memory_limit_mb: 4096,
            logic: "QF_LIA".to_string(),
            extra_flags: Vec::new(),
            dump_queries: false,
            dump_dir: None,
        }
    }
}

// ---------------------------------------------------------------------------
// AnalysisConfig
// ---------------------------------------------------------------------------

/// Configuration for program analysis (WP computation, SSA, etc.).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Maximum expression depth before simplification is forced.
    pub max_expr_depth: u32,
    /// Whether to use SSA form.
    pub use_ssa: bool,
    /// Whether to simplify weakest-precondition formulas.
    pub simplify_wp: bool,
    /// Maximum number of basic blocks in a function before aborting.
    pub max_blocks: u32,
    /// Whether to compute strongest postconditions as well.
    pub compute_sp: bool,
    /// Whether to perform constant propagation.
    pub constant_propagation: bool,
    /// Whether to perform dead code elimination.
    pub dead_code_elimination: bool,
}

impl AnalysisConfig {
    pub fn validate(&self) -> Result<(), MutSpecError> {
        if self.max_expr_depth == 0 {
            return Err(MutSpecError::config_key(
                "max_expr_depth must be > 0",
                "analysis.max_expr_depth",
            ));
        }
        Ok(())
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            max_expr_depth: 50,
            use_ssa: true,
            simplify_wp: true,
            max_blocks: 1000,
            compute_sp: false,
            constant_propagation: true,
            dead_code_elimination: true,
        }
    }
}

// ---------------------------------------------------------------------------
// CoverageConfig
// ---------------------------------------------------------------------------

/// Configuration for mutant coverage analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoverageConfig {
    /// Enable mutant subsumption analysis.
    pub subsumption: bool,
    /// Enable dominator analysis for redundant mutant detection.
    pub dominator_analysis: bool,
    /// Weight for operator diversity in scoring.
    pub operator_diversity_weight: f64,
    /// Weight for location coverage in scoring.
    pub location_coverage_weight: f64,
    /// Minimum mutation score to consider adequate.
    pub adequate_score_threshold: f64,
    /// Whether to compute minimal mutant sets.
    pub compute_minimal_set: bool,
}

impl CoverageConfig {
    pub fn validate(&self) -> Result<(), MutSpecError> {
        if self.adequate_score_threshold < 0.0 || self.adequate_score_threshold > 1.0 {
            return Err(MutSpecError::config_key(
                "adequate_score_threshold must be in [0, 1]",
                "coverage.adequate_score_threshold",
            ));
        }
        Ok(())
    }
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            subsumption: true,
            dominator_analysis: false,
            operator_diversity_weight: 0.3,
            location_coverage_weight: 0.7,
            adequate_score_threshold: 0.8,
            compute_minimal_set: false,
        }
    }
}

// ---------------------------------------------------------------------------
// OutputConfig
// ---------------------------------------------------------------------------

/// Configuration for output generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output format.
    pub format: OutputFormat,
    /// Verbosity level (0 = quiet, 1 = normal, 2 = verbose, 3 = debug).
    pub verbosity: u8,
    /// Output directory.
    pub output_dir: PathBuf,
    /// Whether to generate SARIF output.
    pub sarif: bool,
    /// Whether to generate JML annotations.
    pub jml: bool,
    /// Whether to include provenance information.
    pub include_provenance: bool,
    /// Whether to include the full SMT encoding in output.
    pub include_smt_encoding: bool,
    /// Maximum line width for formatted output.
    pub max_line_width: usize,
}

/// Output format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputFormat {
    Text,
    Json,
    Sarif,
    Jml,
}

impl OutputFormat {
    pub fn file_extension(&self) -> &'static str {
        match self {
            OutputFormat::Text => "txt",
            OutputFormat::Json => "json",
            OutputFormat::Sarif => "sarif",
            OutputFormat::Jml => "jml",
        }
    }
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Text
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Text,
            verbosity: 1,
            output_dir: PathBuf::from("mutspec-output"),
            sarif: false,
            jml: true,
            include_provenance: true,
            include_smt_encoding: false,
            max_line_width: 120,
        }
    }
}

// ---------------------------------------------------------------------------
// MutSpecConfig
// ---------------------------------------------------------------------------

/// Top-level configuration for the MutSpec system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MutSpecConfig {
    pub mutation: MutationConfig,
    pub synthesis: SynthesisConfig,
    pub smt: SmtConfig,
    pub analysis: AnalysisConfig,
    pub coverage: CoverageConfig,
    pub output: OutputConfig,
    /// Optional source file paths.
    pub source_files: Vec<PathBuf>,
}

impl MutSpecConfig {
    /// Validate all sub-configurations.
    pub fn validate(&self) -> Result<(), MutSpecError> {
        self.mutation.validate()?;
        self.synthesis.validate()?;
        self.smt.validate()?;
        self.analysis.validate()?;
        self.coverage.validate()?;
        Ok(())
    }

    /// Load configuration from a TOML string.
    pub fn from_toml(s: &str) -> Result<Self, MutSpecError> {
        let config: MutSpecConfig = toml::from_str(s)
            .map_err(|e| MutSpecError::config(format!("TOML parse error: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    /// Serialize to TOML string.
    pub fn to_toml(&self) -> Result<String, MutSpecError> {
        toml::to_string_pretty(self)
            .map_err(|e| MutSpecError::config(format!("TOML serialization error: {e}")))
    }

    /// Merge overlay values from another config, overriding only set fields.
    /// For simplicity this replaces each sub-config entirely if the overlay
    /// differs from the default.
    pub fn merge_overlay(&mut self, overlay: &MutSpecConfig) {
        if overlay.mutation != MutationConfig::default() {
            self.mutation = overlay.mutation.clone();
        }
        if overlay.synthesis != SynthesisConfig::default() {
            self.synthesis = overlay.synthesis.clone();
        }
        if overlay.smt != SmtConfig::default() {
            self.smt = overlay.smt.clone();
        }
        if overlay.analysis != AnalysisConfig::default() {
            self.analysis = overlay.analysis.clone();
        }
        if overlay.coverage != CoverageConfig::default() {
            self.coverage = overlay.coverage.clone();
        }
        if overlay.output != OutputConfig::default() {
            self.output = overlay.output.clone();
        }
        if !overlay.source_files.is_empty() {
            self.source_files = overlay.source_files.clone();
        }
    }

    /// Create a minimal config for testing.
    pub fn test_config() -> Self {
        let mut config = Self::default();
        config.smt.timeout_secs = 5;
        config.synthesis.total_timeout_secs = 10;
        config.synthesis.tier_timeout_secs = 5;
        config.output.verbosity = 0;
        config
    }

    /// Returns true if the output format is JSON.
    pub fn is_json_output(&self) -> bool {
        matches!(self.output.format, OutputFormat::Json)
    }

    /// Convenience: is verbose mode enabled?
    pub fn is_verbose(&self) -> bool {
        self.output.verbosity >= 2
    }

    /// Convenience: is debug mode enabled?
    pub fn is_debug(&self) -> bool {
        self.output.verbosity >= 3
    }
}

impl Default for MutSpecConfig {
    fn default() -> Self {
        Self {
            mutation: MutationConfig::default(),
            synthesis: SynthesisConfig::default(),
            smt: SmtConfig::default(),
            analysis: AnalysisConfig::default(),
            coverage: CoverageConfig::default(),
            output: OutputConfig::default(),
            source_files: Vec::new(),
        }
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
        let config = MutSpecConfig::default();
        config.validate().unwrap();
    }

    #[test]
    fn test_mutation_config_default() {
        let mc = MutationConfig::default();
        assert!(!mc.operators.is_empty());
        assert_eq!(mc.max_mutants_per_site, 10);
        assert!(!mc.higher_order);
    }

    #[test]
    fn test_mutation_config_enabled_operators() {
        let mc = MutationConfig::default();
        let ops = mc.enabled_operators();
        assert!(ops.contains(&MutationOperator::Aor));
    }

    #[test]
    fn test_mutation_config_is_operator_enabled() {
        let mc = MutationConfig::default();
        assert!(mc.is_operator_enabled(&MutationOperator::Aor));
    }

    #[test]
    fn test_mutation_config_validate_empty_ops() {
        let mut mc = MutationConfig::default();
        mc.operators.clear();
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_mutation_config_validate_bad_op() {
        let mut mc = MutationConfig::default();
        mc.operators.push("BOGUS".to_string());
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_mutation_config_validate_zero_per_site() {
        let mut mc = MutationConfig::default();
        mc.max_mutants_per_site = 0;
        assert!(mc.validate().is_err());
    }

    #[test]
    fn test_synthesis_config_default() {
        let sc = SynthesisConfig::default();
        assert_eq!(sc.enabled_tiers, vec![1, 2, 3]);
        assert!(sc.minimise_contracts);
    }

    #[test]
    fn test_synthesis_config_is_tier_enabled() {
        let sc = SynthesisConfig::default();
        assert!(sc.is_tier_enabled(1));
        assert!(!sc.is_tier_enabled(4));
    }

    #[test]
    fn test_synthesis_config_validate_empty_tiers() {
        let mut sc = SynthesisConfig::default();
        sc.enabled_tiers.clear();
        assert!(sc.validate().is_err());
    }

    #[test]
    fn test_synthesis_config_validate_bad_tier() {
        let mut sc = SynthesisConfig::default();
        sc.enabled_tiers.push(5);
        assert!(sc.validate().is_err());
    }

    #[test]
    fn test_synthesis_config_validate_zero_timeout() {
        let mut sc = SynthesisConfig::default();
        sc.tier_timeout_secs = 0;
        assert!(sc.validate().is_err());
    }

    #[test]
    fn test_smt_config_default() {
        let sc = SmtConfig::default();
        assert_eq!(sc.solver_path, PathBuf::from("z3"));
        assert!(sc.incremental);
        assert_eq!(sc.logic, "QF_LIA");
    }

    #[test]
    fn test_smt_config_validate_zero_timeout() {
        let mut sc = SmtConfig::default();
        sc.timeout_secs = 0;
        assert!(sc.validate().is_err());
    }

    #[test]
    fn test_smt_config_validate_zero_memory() {
        let mut sc = SmtConfig::default();
        sc.memory_limit_mb = 0;
        assert!(sc.validate().is_err());
    }

    #[test]
    fn test_smt_config_solver_args() {
        let sc = SmtConfig::default();
        let args = sc.solver_args();
        assert!(args.iter().any(|a| a.starts_with("-T:")));
        assert!(args.iter().any(|a| a.starts_with("-memory:")));
    }

    #[test]
    fn test_analysis_config_default() {
        let ac = AnalysisConfig::default();
        assert!(ac.use_ssa);
        assert!(ac.simplify_wp);
    }

    #[test]
    fn test_analysis_config_validate_zero_depth() {
        let mut ac = AnalysisConfig::default();
        ac.max_expr_depth = 0;
        assert!(ac.validate().is_err());
    }

    #[test]
    fn test_coverage_config_default() {
        let cc = CoverageConfig::default();
        assert!(cc.subsumption);
        assert!((cc.adequate_score_threshold - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_coverage_config_validate_bad_threshold() {
        let mut cc = CoverageConfig::default();
        cc.adequate_score_threshold = 1.5;
        assert!(cc.validate().is_err());
    }

    #[test]
    fn test_output_config_default() {
        let oc = OutputConfig::default();
        assert_eq!(oc.format, OutputFormat::Text);
        assert_eq!(oc.verbosity, 1);
        assert!(oc.jml);
    }

    #[test]
    fn test_output_format_extension() {
        assert_eq!(OutputFormat::Text.file_extension(), "txt");
        assert_eq!(OutputFormat::Json.file_extension(), "json");
        assert_eq!(OutputFormat::Sarif.file_extension(), "sarif");
        assert_eq!(OutputFormat::Jml.file_extension(), "jml");
    }

    #[test]
    fn test_config_test_config() {
        let tc = MutSpecConfig::test_config();
        assert_eq!(tc.smt.timeout_secs, 5);
        assert_eq!(tc.output.verbosity, 0);
        tc.validate().unwrap();
    }

    #[test]
    fn test_config_is_verbose_debug() {
        let mut c = MutSpecConfig::default();
        assert!(!c.is_verbose());
        assert!(!c.is_debug());
        c.output.verbosity = 2;
        assert!(c.is_verbose());
        assert!(!c.is_debug());
        c.output.verbosity = 3;
        assert!(c.is_debug());
    }

    #[test]
    fn test_config_is_json() {
        let mut c = MutSpecConfig::default();
        assert!(!c.is_json_output());
        c.output.format = OutputFormat::Json;
        assert!(c.is_json_output());
    }

    #[test]
    fn test_config_toml_roundtrip() {
        let config = MutSpecConfig::default();
        let toml_str = config.to_toml().unwrap();
        let config2 = MutSpecConfig::from_toml(&toml_str).unwrap();
        assert_eq!(config, config2);
    }

    #[test]
    fn test_config_merge_overlay() {
        let mut base = MutSpecConfig::default();
        let mut overlay = MutSpecConfig::default();
        overlay.smt.timeout_secs = 999;
        base.merge_overlay(&overlay);
        assert_eq!(base.smt.timeout_secs, 999);
    }

    #[test]
    fn test_config_merge_overlay_source_files() {
        let mut base = MutSpecConfig::default();
        let mut overlay = MutSpecConfig::default();
        overlay.source_files = vec![PathBuf::from("test.ms")];
        base.merge_overlay(&overlay);
        assert_eq!(base.source_files.len(), 1);
    }

    #[test]
    fn test_config_serialization_json() {
        let config = MutSpecConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let config2: MutSpecConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, config2);
    }

    #[test]
    fn test_config_from_toml_invalid() {
        let result = MutSpecConfig::from_toml("invalid { toml [");
        assert!(result.is_err());
    }
}
