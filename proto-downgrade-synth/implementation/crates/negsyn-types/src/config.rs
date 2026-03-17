//! Configuration types for the NegSynth analysis pipeline.
//!
//! Provides typed configuration with defaults, validation, and builder
//! patterns for all pipeline phases.

use crate::adversary::AdversaryBudget;
use crate::error::{ConfigError, NegSynthError};
use crate::protocol::{ProtocolFamily, SecurityLevel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

// ── Top-level analysis config ────────────────────────────────────────────

/// Master configuration for the NegSynth analysis pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    pub slicer: SlicerConfig,
    pub merge: MergeConfig,
    pub extraction: ExtractionConfig,
    pub encoding: EncodingConfig,
    pub concretizer: ConcretizerConfig,
    pub protocol: ProtocolConfig,
    pub output_dir: PathBuf,
    pub verbose: bool,
    pub timeout_secs: u64,
    pub max_memory_mb: u64,
    pub parallel_workers: u32,
}

impl AnalysisConfig {
    pub fn validate(&self) -> Result<(), NegSynthError> {
        self.slicer.validate()?;
        self.merge.validate()?;
        self.extraction.validate()?;
        self.encoding.validate()?;
        self.concretizer.validate()?;
        self.protocol.validate()?;

        if self.timeout_secs == 0 {
            return Err(ConfigError::invalid_field("timeout_secs", "must be positive").into());
        }
        if self.parallel_workers == 0 {
            return Err(
                ConfigError::invalid_field("parallel_workers", "must be at least 1").into(),
            );
        }
        Ok(())
    }

    pub fn builder() -> AnalysisConfigBuilder {
        AnalysisConfigBuilder::new()
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        AnalysisConfig {
            slicer: SlicerConfig::default(),
            merge: MergeConfig::default(),
            extraction: ExtractionConfig::default(),
            encoding: EncodingConfig::default(),
            concretizer: ConcretizerConfig::default(),
            protocol: ProtocolConfig::default(),
            output_dir: PathBuf::from("./output"),
            verbose: false,
            timeout_secs: 3600,
            max_memory_mb: 8192,
            parallel_workers: 4,
        }
    }
}

// ── Builder ──────────────────────────────────────────────────────────────

pub struct AnalysisConfigBuilder {
    config: AnalysisConfig,
}

impl AnalysisConfigBuilder {
    pub fn new() -> Self {
        AnalysisConfigBuilder {
            config: AnalysisConfig::default(),
        }
    }

    pub fn slicer(mut self, slicer: SlicerConfig) -> Self {
        self.config.slicer = slicer;
        self
    }

    pub fn merge(mut self, merge: MergeConfig) -> Self {
        self.config.merge = merge;
        self
    }

    pub fn extraction(mut self, extraction: ExtractionConfig) -> Self {
        self.config.extraction = extraction;
        self
    }

    pub fn encoding(mut self, encoding: EncodingConfig) -> Self {
        self.config.encoding = encoding;
        self
    }

    pub fn concretizer(mut self, concretizer: ConcretizerConfig) -> Self {
        self.config.concretizer = concretizer;
        self
    }

    pub fn protocol(mut self, protocol: ProtocolConfig) -> Self {
        self.config.protocol = protocol;
        self
    }

    pub fn output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.output_dir = dir.into();
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.config.timeout_secs = secs;
        self
    }

    pub fn max_memory_mb(mut self, mb: u64) -> Self {
        self.config.max_memory_mb = mb;
        self
    }

    pub fn parallel_workers(mut self, workers: u32) -> Self {
        self.config.parallel_workers = workers;
        self
    }

    pub fn build(self) -> Result<AnalysisConfig, NegSynthError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for AnalysisConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Slicer Config ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlicerConfig {
    /// Entry point function names to start slicing from.
    pub entry_points: Vec<String>,
    /// Maximum slice depth (call chain length).
    pub max_depth: u32,
    /// Whether to include indirect calls.
    pub follow_indirect_calls: bool,
    /// Functions to exclude from the slice.
    pub excluded_functions: Vec<String>,
    /// Maximum IR instructions in a slice.
    pub max_instructions: u64,
    /// Whether to track data dependencies.
    pub track_data_deps: bool,
    /// Whether to track control dependencies.
    pub track_control_deps: bool,
}

impl SlicerConfig {
    pub fn validate(&self) -> Result<(), NegSynthError> {
        if self.entry_points.is_empty() {
            return Err(
                ConfigError::missing_field("slicer.entry_points").into(),
            );
        }
        if self.max_depth == 0 {
            return Err(
                ConfigError::invalid_field("slicer.max_depth", "must be positive").into(),
            );
        }
        if self.max_instructions == 0 {
            return Err(
                ConfigError::invalid_field("slicer.max_instructions", "must be positive").into(),
            );
        }
        Ok(())
    }
}

impl Default for SlicerConfig {
    fn default() -> Self {
        SlicerConfig {
            entry_points: vec![
                "SSL_do_handshake".into(),
                "tls13_client_hello".into(),
                "ssl3_get_server_hello".into(),
            ],
            max_depth: 20,
            follow_indirect_calls: true,
            excluded_functions: vec!["malloc".into(), "free".into(), "memcpy".into()],
            max_instructions: 100_000,
            track_data_deps: true,
            track_control_deps: true,
        }
    }
}

// ── Merge Config ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConfig {
    /// Maximum number of states before forcing merges.
    pub max_states: u32,
    /// Merge strategy to use.
    pub strategy: MergeStrategy,
    /// Maximum merges per merge point.
    pub max_merges_per_point: u32,
    /// Widening threshold: widen after this many iterations.
    pub widening_threshold: u32,
    /// Whether to use protocol-aware merging.
    pub protocol_aware: bool,
    /// Similarity threshold for merge compatibility (0.0-1.0).
    pub similarity_threshold: f64,
    /// Maximum cipher suite outcomes per merged state.
    pub max_cipher_outcomes: u32,
    /// Maximum version outcomes per merged state.
    pub max_version_outcomes: u32,
    /// Maximum extension outcomes per merged state.
    pub max_extension_outcomes: u32,
    /// Maximum ITE nesting depth in merged expressions.
    pub max_ite_depth: u32,
    /// Maximum merged constraints before widening.
    pub max_merged_constraints: u32,
    /// Enable caching of merge results.
    pub enable_caching: bool,
    /// Cache capacity (number of entries).
    pub cache_capacity: usize,
    /// Enable constraint simplification during merge.
    pub enable_constraint_simplification: bool,
    /// Enable FIPS-only mode (restrict to FIPS-approved suites).
    pub fips_mode: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Merge all compatible states at each program point.
    Aggressive,
    /// Only merge states with high similarity.
    Conservative,
    /// Protocol-aware merge that preserves negotiation semantics.
    ProtocolAware,
    /// No merging (full symbolic execution).
    None,
}

impl MergeConfig {
    pub fn validate(&self) -> Result<(), NegSynthError> {
        if self.max_states == 0 {
            return Err(
                ConfigError::invalid_field("merge.max_states", "must be positive").into(),
            );
        }
        if !(0.0..=1.0).contains(&self.similarity_threshold) {
            return Err(ConfigError::InvalidField {
                field: "merge.similarity_threshold".into(),
                reason: "must be between 0.0 and 1.0".into(),
            }.into());
        }
        Ok(())
    }
}

impl Default for MergeConfig {
    fn default() -> Self {
        MergeConfig {
            max_states: 10_000,
            strategy: MergeStrategy::ProtocolAware,
            max_merges_per_point: 100,
            widening_threshold: 5,
            protocol_aware: true,
            similarity_threshold: 0.8,
            max_cipher_outcomes: 16,
            max_version_outcomes: 8,
            max_extension_outcomes: 32,
            max_ite_depth: 64,
            max_merged_constraints: 256,
            enable_caching: true,
            cache_capacity: 4096,
            enable_constraint_simplification: true,
            fips_mode: false,
        }
    }
}

// ── Extraction Config ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Patterns to recognize cipher suite arrays.
    pub cipher_patterns: Vec<String>,
    /// Patterns to recognize version negotiation.
    pub version_patterns: Vec<String>,
    /// Whether to extract extensions.
    pub extract_extensions: bool,
    /// Whether to infer types from usage.
    pub type_inference: bool,
    /// Confidence threshold for extraction (0.0-1.0).
    pub confidence_threshold: f64,
}

impl ExtractionConfig {
    pub fn validate(&self) -> Result<(), NegSynthError> {
        if !(0.0..=1.0).contains(&self.confidence_threshold) {
            return Err(ConfigError::InvalidField {
                field: "extraction.confidence_threshold".into(),
                reason: "must be between 0.0 and 1.0".into(),
            }.into());
        }
        Ok(())
    }
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        ExtractionConfig {
            cipher_patterns: vec![
                "cipher_suites".into(),
                "offered_ciphers".into(),
                "supported_ciphers".into(),
            ],
            version_patterns: vec![
                "client_version".into(),
                "server_version".into(),
                "protocol_version".into(),
            ],
            extract_extensions: true,
            type_inference: true,
            confidence_threshold: 0.7,
        }
    }
}

// ── Encoding Config ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingConfig {
    /// SMT logic to use.
    pub logic: String,
    /// Maximum expression depth.
    pub max_expr_depth: u32,
    /// Maximum formula nodes.
    pub max_formula_nodes: u64,
    /// Bitvector width for addresses.
    pub address_width: u32,
    /// Bitvector width for data values.
    pub data_width: u32,
    /// Whether to use array theory for memory.
    pub use_array_theory: bool,
    /// Solver timeout in milliseconds.
    pub solver_timeout_ms: u64,
    /// Adversary budget for the encoding.
    pub adversary_budget: AdversaryBudget,
}

impl EncodingConfig {
    pub fn validate(&self) -> Result<(), NegSynthError> {
        if self.logic.is_empty() {
            return Err(
                ConfigError::missing_field("encoding.logic").into(),
            );
        }
        if self.max_expr_depth == 0 {
            return Err(
                ConfigError::invalid_field("encoding.max_expr_depth", "must be positive").into(),
            );
        }
        if self.address_width == 0 || self.data_width == 0 {
            return Err(
                ConfigError::invalid_field("encoding.address/data_width", "must be positive").into(),
            );
        }
        Ok(())
    }
}

impl Default for EncodingConfig {
    fn default() -> Self {
        EncodingConfig {
            logic: "QF_AUFBV".into(),
            max_expr_depth: 100,
            max_formula_nodes: 1_000_000,
            address_width: 64,
            data_width: 8,
            use_array_theory: true,
            solver_timeout_ms: 60_000,
            adversary_budget: AdversaryBudget::standard(),
        }
    }
}

// ── Concretizer Config ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcretizerConfig {
    /// Maximum replay attempts.
    pub max_replay_attempts: u32,
    /// Whether to validate attacks via network replay.
    pub network_validation: bool,
    /// Target host for network replay.
    pub target_host: Option<String>,
    /// Target port for network replay.
    pub target_port: Option<u16>,
    /// Timeout for network operations (ms).
    pub network_timeout_ms: u64,
    /// Whether to generate pcap files.
    pub generate_pcap: bool,
}

impl ConcretizerConfig {
    pub fn validate(&self) -> Result<(), NegSynthError> {
        if self.max_replay_attempts == 0 {
            return Err(
                ConfigError::invalid_field("concretizer.max_replay_attempts", "must be positive")
                    .into(),
            );
        }
        if self.network_validation && self.target_host.is_none() {
            return Err(ConfigError::InvalidField {
                field: "concretizer.target_host".into(),
                reason: "required when network_validation is true".into(),
            }.into());
        }
        Ok(())
    }
}

impl Default for ConcretizerConfig {
    fn default() -> Self {
        ConcretizerConfig {
            max_replay_attempts: 3,
            network_validation: false,
            target_host: None,
            target_port: None,
            network_timeout_ms: 5000,
            generate_pcap: false,
        }
    }
}

// ── Protocol Config ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolConfig {
    /// Protocol family being analyzed.
    pub family: ProtocolFamily,
    /// Minimum acceptable security level.
    pub min_security_level: SecurityLevel,
    /// Whether to check for version downgrade.
    pub check_version_downgrade: bool,
    /// Whether to check for cipher suite downgrade.
    pub check_cipher_downgrade: bool,
    /// Whether to check for forward secrecy loss.
    pub check_forward_secrecy: bool,
    /// Whether to check for extension stripping.
    pub check_extension_stripping: bool,
    /// Specific cipher suite IDs to consider as targets.
    pub target_cipher_ids: Vec<u16>,
    /// Minimum version to consider acceptable.
    pub min_version: Option<(u8, u8)>,
    /// Extra custom properties to check.
    pub custom_properties: HashMap<String, String>,
}

impl ProtocolConfig {
    pub fn validate(&self) -> Result<(), NegSynthError> {
        if !self.check_version_downgrade
            && !self.check_cipher_downgrade
            && !self.check_forward_secrecy
            && !self.check_extension_stripping
        {
            return Err(ConfigError::InvalidField {
                field: "protocol".into(),
                reason: "at least one check must be enabled".into(),
            }.into());
        }
        Ok(())
    }

    pub fn tls_default() -> Self {
        ProtocolConfig {
            family: ProtocolFamily::TLS,
            min_security_level: SecurityLevel::Standard,
            check_version_downgrade: true,
            check_cipher_downgrade: true,
            check_forward_secrecy: true,
            check_extension_stripping: true,
            target_cipher_ids: Vec::new(),
            min_version: Some((3, 3)), // TLS 1.2
            custom_properties: HashMap::new(),
        }
    }

    pub fn ssh_default() -> Self {
        ProtocolConfig {
            family: ProtocolFamily::SSH,
            min_security_level: SecurityLevel::Standard,
            check_version_downgrade: true,
            check_cipher_downgrade: true,
            check_forward_secrecy: true,
            check_extension_stripping: false,
            target_cipher_ids: Vec::new(),
            min_version: Some((2, 0)),
            custom_properties: HashMap::new(),
        }
    }
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        Self::tls_default()
    }
}

impl fmt::Display for AnalysisConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "AnalysisConfig:")?;
        writeln!(f, "  output: {}", self.output_dir.display())?;
        writeln!(f, "  timeout: {}s", self.timeout_secs)?;
        writeln!(f, "  workers: {}", self.parallel_workers)?;
        writeln!(f, "  protocol: {:?}", self.protocol.family)?;
        writeln!(f, "  merge strategy: {:?}", self.merge.strategy)?;
        writeln!(f, "  encoding logic: {}", self.encoding.logic)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let config = AnalysisConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder() {
        let config = AnalysisConfig::builder()
            .verbose(true)
            .timeout_secs(7200)
            .parallel_workers(8)
            .build();
        assert!(config.is_ok());
        let c = config.unwrap();
        assert!(c.verbose);
        assert_eq!(c.timeout_secs, 7200);
    }

    #[test]
    fn test_invalid_timeout() {
        let config = AnalysisConfig::builder().timeout_secs(0).build();
        assert!(config.is_err());
    }

    #[test]
    fn test_invalid_workers() {
        let config = AnalysisConfig::builder().parallel_workers(0).build();
        assert!(config.is_err());
    }

    #[test]
    fn test_slicer_config_validation() {
        let mut sc = SlicerConfig::default();
        assert!(sc.validate().is_ok());

        sc.entry_points.clear();
        assert!(sc.validate().is_err());
    }

    #[test]
    fn test_merge_config_validation() {
        let mc = MergeConfig::default();
        assert!(mc.validate().is_ok());

        let mut mc2 = mc.clone();
        mc2.similarity_threshold = 1.5;
        assert!(mc2.validate().is_err());
    }

    #[test]
    fn test_encoding_config_validation() {
        let ec = EncodingConfig::default();
        assert!(ec.validate().is_ok());
    }

    #[test]
    fn test_protocol_config() {
        let tls = ProtocolConfig::tls_default();
        assert!(tls.validate().is_ok());
        assert_eq!(tls.family, ProtocolFamily::TLS);

        let ssh = ProtocolConfig::ssh_default();
        assert!(ssh.validate().is_ok());
        assert_eq!(ssh.family, ProtocolFamily::SSH);
    }

    #[test]
    fn test_no_checks_invalid() {
        let mut pc = ProtocolConfig::default();
        pc.check_version_downgrade = false;
        pc.check_cipher_downgrade = false;
        pc.check_forward_secrecy = false;
        pc.check_extension_stripping = false;
        assert!(pc.validate().is_err());
    }

    #[test]
    fn test_concretizer_network_validation() {
        let mut cc = ConcretizerConfig::default();
        cc.network_validation = true;
        assert!(cc.validate().is_err()); // missing target_host

        cc.target_host = Some("localhost".into());
        assert!(cc.validate().is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let config = AnalysisConfig::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        let deserialized: AnalysisConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.timeout_secs, config.timeout_secs);
    }

    #[test]
    fn test_merge_strategies() {
        let strategies = [
            MergeStrategy::Aggressive,
            MergeStrategy::Conservative,
            MergeStrategy::ProtocolAware,
            MergeStrategy::None,
        ];
        for s in strategies {
            let mut mc = MergeConfig::default();
            mc.strategy = s;
            assert!(mc.validate().is_ok());
        }
    }

    #[test]
    fn test_extraction_config() {
        let ec = ExtractionConfig::default();
        assert!(ec.validate().is_ok());
        assert!(ec.extract_extensions);
        assert!(ec.type_inference);
    }

    #[test]
    fn test_config_display() {
        let config = AnalysisConfig::default();
        let display = format!("{}", config);
        assert!(display.contains("AnalysisConfig"));
        assert!(display.contains("QF_AUFBV"));
    }
}
