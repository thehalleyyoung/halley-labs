//! Configuration management for the CollusionProof CLI.
//!
//! Handles loading from files, environment variables, CLI argument merging,
//! validation, and default configuration generation in TOML format.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use shared_types::{
    BootstrapConfig, ConfidenceLevel, GlobalConfig, MonteCarloConfig, SignificanceLevel,
    TieredNullConfig, EvaluationMode,
};

use crate::commands::{ConfigProfileArg, RunArgs};

// ── Config profiles ─────────────────────────────────────────────────────────

/// Preset configuration profiles that override default parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfigProfile {
    Smoke,
    Standard,
    Full,
}

impl ConfigProfile {
    pub fn apply(self, config: &mut CliConfig) {
        match self {
            ConfigProfile::Smoke => {
                config.num_rounds = 100;
                config.global.evaluation_mode = EvaluationMode::Smoke;
                config.global.bootstrap = BootstrapConfig::for_mode(EvaluationMode::Smoke);
                config.global.monte_carlo = MonteCarloConfig::for_mode(EvaluationMode::Smoke);
            }
            ConfigProfile::Standard => {
                config.num_rounds = 1000;
                config.global.evaluation_mode = EvaluationMode::Standard;
                config.global.bootstrap = BootstrapConfig::for_mode(EvaluationMode::Standard);
                config.global.monte_carlo = MonteCarloConfig::for_mode(EvaluationMode::Standard);
            }
            ConfigProfile::Full => {
                config.num_rounds = 10000;
                config.global.evaluation_mode = EvaluationMode::Full;
                config.global.bootstrap = BootstrapConfig::for_mode(EvaluationMode::Full);
                config.global.monte_carlo = MonteCarloConfig::for_mode(EvaluationMode::Full);
            }
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ConfigProfile::Smoke => "smoke",
            ConfigProfile::Standard => "standard",
            ConfigProfile::Full => "full",
        }
    }
}

impl From<ConfigProfileArg> for ConfigProfile {
    fn from(arg: ConfigProfileArg) -> Self {
        match arg {
            ConfigProfileArg::Smoke => ConfigProfile::Smoke,
            ConfigProfileArg::Standard => ConfigProfile::Standard,
            ConfigProfileArg::Full => ConfigProfile::Full,
        }
    }
}

// ── Extended config types ───────────────────────────────────────────────────

/// Extended CLI configuration wrapping GlobalConfig with CLI-specific fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    #[serde(flatten)]
    pub global: GlobalConfig,

    #[serde(default = "default_num_rounds")]
    pub num_rounds: usize,

    #[serde(default = "default_num_players")]
    pub num_players: usize,

    #[serde(default = "default_output_dir")]
    pub output_dir: String,

    #[serde(default = "default_true")]
    pub color: bool,

    #[serde(default)]
    pub jobs: usize,

    #[serde(default)]
    pub checkpoint: bool,

    #[serde(default = "default_checkpoint_dir")]
    pub checkpoint_dir: String,

    #[serde(default)]
    pub save_intermediates: bool,
}

fn default_num_rounds() -> usize {
    1000
}

fn default_num_players() -> usize {
    2
}

fn default_output_dir() -> String {
    "./output".to_string()
}

fn default_checkpoint_dir() -> String {
    "./.checkpoints".to_string()
}

fn default_true() -> bool {
    true
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            global: GlobalConfig::default(),
            num_rounds: default_num_rounds(),
            num_players: default_num_players(),
            output_dir: default_output_dir(),
            color: true,
            jobs: 0,
            checkpoint: false,
            checkpoint_dir: default_checkpoint_dir(),
            save_intermediates: false,
        }
    }
}

// ── Config loading ──────────────────────────────────────────────────────────

const CONFIG_SEARCH_NAMES: &[&str] = &[
    "collusion-proof.toml",
    ".collusion-proof.toml",
    "collusion-proof.json",
];

/// Load configuration from file, environment, and defaults.
pub fn load_config(path: Option<&str>) -> Result<CliConfig> {
    let mut config = CliConfig::default();

    if let Some(p) = path {
        let content = std::fs::read_to_string(p)
            .with_context(|| format!("Failed to read config file: {}", p))?;
        config = parse_config_content(&content, p)?;
        log::info!("Loaded config from: {}", p);
    } else if let Some(found) = find_config_file() {
        let content = std::fs::read_to_string(&found)
            .with_context(|| format!("Failed to read config: {}", found.display()))?;
        config = parse_config_content(&content, &found.to_string_lossy())?;
        log::info!("Loaded config from: {}", found.display());
    } else {
        log::debug!("No config file found, using defaults");
    }

    apply_env_overrides(&mut config);
    Ok(config)
}

fn find_config_file() -> Option<PathBuf> {
    for name in CONFIG_SEARCH_NAMES {
        let path = PathBuf::from(name);
        if path.exists() {
            return Some(path);
        }
    }
    if let Some(home) = dirs_home() {
        for name in CONFIG_SEARCH_NAMES {
            let path = home.join(name);
            if path.exists() {
                return Some(path);
            }
        }
    }
    None
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| std::env::var("USERPROFILE").ok().map(PathBuf::from))
}

fn parse_config_content(content: &str, path: &str) -> Result<CliConfig> {
    if path.ends_with(".json") {
        serde_json::from_str(content)
            .with_context(|| format!("Failed to parse JSON config: {}", path))
    } else {
        toml::from_str(content)
            .with_context(|| format!("Failed to parse TOML config: {}", path))
    }
}

fn apply_env_overrides(config: &mut CliConfig) {
    if let Ok(val) = std::env::var("COLLUSION_PROOF_ALPHA") {
        if let Ok(alpha) = val.parse::<f64>() {
            if alpha > 0.0 && alpha < 1.0 {
                if let Ok(sl) = SignificanceLevel::new(alpha) {
                    config.global.significance_level = sl;
                    log::debug!("Override from env: alpha = {}", alpha);
                }
            }
        }
    }
    if let Ok(val) = std::env::var("COLLUSION_PROOF_ROUNDS") {
        if let Ok(rounds) = val.parse::<usize>() {
            config.num_rounds = rounds;
            log::debug!("Override from env: rounds = {}", rounds);
        }
    }
    if let Ok(val) = std::env::var("COLLUSION_PROOF_PLAYERS") {
        if let Ok(players) = val.parse::<usize>() {
            config.num_players = players;
            log::debug!("Override from env: players = {}", players);
        }
    }
    if let Ok(val) = std::env::var("COLLUSION_PROOF_BOOTSTRAP") {
        if let Ok(n) = val.parse::<usize>() {
            config.global.bootstrap.num_iterations = n;
            log::debug!("Override from env: bootstrap_iterations = {}", n);
        }
    }
    if let Ok(val) = std::env::var("COLLUSION_PROOF_JOBS") {
        if let Ok(j) = val.parse::<usize>() {
            config.jobs = j;
            log::debug!("Override from env: jobs = {}", j);
        }
    }
    if let Ok(val) = std::env::var("COLLUSION_PROOF_OUTPUT_DIR") {
        config.output_dir = val.clone();
        log::debug!("Override from env: output_dir = {}", val);
    }
    if std::env::var("COLLUSION_PROOF_CHECKPOINT").is_ok() {
        config.checkpoint = true;
        log::debug!("Override from env: checkpoint = true");
    }
}

// ── Merge CLI args ──────────────────────────────────────────────────────────

/// Merge CLI arguments into the config, overriding file/env defaults.
pub fn merge_cli_args(config: &mut CliConfig, args: &RunArgs) {
    if let Ok(sl) = SignificanceLevel::new(args.alpha) {
        config.global.significance_level = sl;
    }
    config.num_rounds = args.rounds;
    config.global.bootstrap.num_iterations = args.bootstrap_resamples;
    config.global.bootstrap.seed = Some(args.seed);
    config.global.monte_carlo.seed = Some(args.seed);
    config.output_dir = args.output_dir.to_string_lossy().to_string();
    config.jobs = args.jobs;
    config.checkpoint = args.checkpoint;
    config.checkpoint_dir = args.checkpoint_dir.to_string_lossy().to_string();
    config.save_intermediates = args.save_intermediates;
}

// ── Config builder ──────────────────────────────────────────────────────────

/// Fluent builder for constructing a CliConfig.
#[derive(Debug, Default)]
pub struct ConfigBuilder {
    config: CliConfig,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn significance_level(mut self, alpha: f64) -> Self {
        if let Ok(sl) = SignificanceLevel::new(alpha) {
            self.config.global.significance_level = sl;
        }
        self
    }

    pub fn confidence_level(mut self, level: f64) -> Self {
        if let Ok(cl) = ConfidenceLevel::new(level) {
            self.config.global.confidence_level = cl;
        }
        self
    }

    pub fn num_rounds(mut self, rounds: usize) -> Self {
        self.config.num_rounds = rounds;
        self
    }

    pub fn num_players(mut self, players: usize) -> Self {
        self.config.num_players = players;
        self
    }

    pub fn bootstrap_resamples(mut self, n: usize) -> Self {
        self.config.global.bootstrap.num_iterations = n;
        self
    }

    pub fn monte_carlo_simulations(mut self, n: usize) -> Self {
        self.config.global.monte_carlo.num_iterations = n;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.config.global.bootstrap.seed = Some(seed);
        self.config.global.monte_carlo.seed = Some(seed);
        self
    }

    pub fn output_dir(mut self, dir: impl Into<String>) -> Self {
        self.config.output_dir = dir.into();
        self
    }

    pub fn jobs(mut self, j: usize) -> Self {
        self.config.jobs = j;
        self
    }

    pub fn checkpoint(mut self, enable: bool) -> Self {
        self.config.checkpoint = enable;
        self
    }

    pub fn checkpoint_dir(mut self, dir: impl Into<String>) -> Self {
        self.config.checkpoint_dir = dir.into();
        self
    }

    pub fn save_intermediates(mut self, enable: bool) -> Self {
        self.config.save_intermediates = enable;
        self
    }

    pub fn profile(mut self, profile: ConfigProfile) -> Self {
        profile.apply(&mut self.config);
        self
    }

    pub fn build(self) -> CliConfig {
        self.config
    }
}

// ── Config validation ───────────────────────────────────────────────────────

/// Validate a configuration for internal consistency.
pub fn validate_config(config: &CliConfig) -> Result<()> {
    let alpha = config.global.significance_level.value();
    if alpha <= 0.0 || alpha >= 1.0 {
        bail!("significance_level must be in (0, 1), got {}", alpha);
    }
    let cl = config.global.confidence_level.value();
    if cl <= 0.0 || cl >= 1.0 {
        bail!("confidence_level must be in (0, 1), got {}", cl);
    }
    if config.num_rounds < 10 {
        bail!("num_rounds must be >= 10, got {}", config.num_rounds);
    }
    if config.num_players < 2 {
        bail!("num_players must be >= 2, got {}", config.num_players);
    }
    if config.global.bootstrap.num_iterations < 10 {
        bail!(
            "bootstrap.num_iterations must be >= 10, got {}",
            config.global.bootstrap.num_iterations
        );
    }
    if config.global.monte_carlo.num_iterations < 10 {
        bail!(
            "monte_carlo.num_iterations must be >= 10, got {}",
            config.global.monte_carlo.num_iterations
        );
    }
    if alpha + cl > 1.5 {
        log::warn!(
            "Unusual configuration: alpha={} + confidence={} > 1.5",
            alpha, cl
        );
    }
    Ok(())
}

// ── Default config generation ───────────────────────────────────────────────

/// Generate a default configuration file in TOML format.
pub fn generate_default_config(profile: Option<ConfigProfile>) -> String {
    let mut config = CliConfig::default();
    if let Some(p) = profile {
        p.apply(&mut config);
    }
    let profile_name = profile.map(|p| p.name()).unwrap_or("standard");

    let mut s = String::new();
    s.push_str(&format!("# CollusionProof Configuration\n"));
    s.push_str(&format!("# Profile: {}\n", profile_name));
    s.push_str("# Generated automatically - edit as needed.\n\n");
    s.push_str(&format!("significance_level = {}\n", config.global.significance_level.value()));
    s.push_str(&format!("confidence_level = {}\n", config.global.confidence_level.value()));
    s.push_str(&format!("num_rounds = {}\n", config.num_rounds));
    s.push_str(&format!("num_players = {}\n\n", config.num_players));
    s.push_str("[bootstrap]\n");
    s.push_str(&format!("num_iterations = {}\n", config.global.bootstrap.num_iterations));
    s.push_str(&format!("confidence_level = {}\n\n", config.global.bootstrap.confidence_level.value()));
    s.push_str("[monte_carlo]\n");
    s.push_str(&format!("num_iterations = {}\n\n", config.global.monte_carlo.num_iterations));
    s.push_str("output_dir = \"./output\"\n");
    s.push_str("color = true\n");
    s.push_str("jobs = 0\n");
    s.push_str("checkpoint = false\n");
    s.push_str("checkpoint_dir = \"./.checkpoints\"\n");
    s.push_str("save_intermediates = false\n");
    s
}

/// Write a default config file to the given path.
pub fn write_default_config(path: &Path, profile: Option<ConfigProfile>) -> Result<()> {
    let content = generate_default_config(profile);
    std::fs::write(path, &content)
        .with_context(|| format!("Failed to write config to: {}", path.display()))?;
    log::info!("Wrote default config to: {}", path.display());
    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CliConfig::default();
        assert_eq!(config.num_rounds, 1000);
        assert_eq!(config.num_players, 2);
        assert!(!config.checkpoint);
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .significance_level(0.01)
            .num_rounds(5000)
            .num_players(3)
            .bootstrap_resamples(2000)
            .jobs(4)
            .output_dir("/tmp/output")
            .build();
        assert!((config.global.significance_level.value() - 0.01).abs() < 1e-9);
        assert_eq!(config.num_rounds, 5000);
        assert_eq!(config.num_players, 3);
        assert_eq!(config.global.bootstrap.num_iterations, 2000);
        assert_eq!(config.jobs, 4);
        assert_eq!(config.output_dir, "/tmp/output");
    }

    #[test]
    fn test_config_builder_with_profile() {
        let config = ConfigBuilder::new().profile(ConfigProfile::Smoke).build();
        assert_eq!(config.num_rounds, 100);
    }

    #[test]
    fn test_config_builder_with_full_profile() {
        let config = ConfigBuilder::new().profile(ConfigProfile::Full).build();
        assert_eq!(config.num_rounds, 10000);
    }

    #[test]
    fn test_config_profile_apply() {
        let mut config = CliConfig::default();
        ConfigProfile::Standard.apply(&mut config);
        assert_eq!(config.num_rounds, 1000);
    }

    #[test]
    fn test_config_profile_names() {
        assert_eq!(ConfigProfile::Smoke.name(), "smoke");
        assert_eq!(ConfigProfile::Standard.name(), "standard");
        assert_eq!(ConfigProfile::Full.name(), "full");
    }

    #[test]
    fn test_validate_config_valid() {
        let config = CliConfig::default();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_bad_rounds() {
        let mut config = CliConfig::default();
        config.num_rounds = 5;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_bad_players() {
        let mut config = CliConfig::default();
        config.num_players = 1;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_bad_bootstrap() {
        let mut config = CliConfig::default();
        config.global.bootstrap.num_iterations = 5;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_generate_default_config() {
        let toml_str = generate_default_config(None);
        assert!(toml_str.contains("significance_level"));
        assert!(toml_str.contains("num_rounds"));
        assert!(toml_str.contains("[bootstrap]"));
        assert!(toml_str.contains("[monte_carlo]"));
    }

    #[test]
    fn test_generate_smoke_config() {
        let toml_str = generate_default_config(Some(ConfigProfile::Smoke));
        assert!(toml_str.contains("smoke"));
        assert!(toml_str.contains("num_rounds = 100"));
    }

    #[test]
    fn test_generate_full_config() {
        let toml_str = generate_default_config(Some(ConfigProfile::Full));
        assert!(toml_str.contains("full"));
        assert!(toml_str.contains("num_rounds = 10000"));
    }

    #[test]
    fn test_load_config_no_file() {
        let config = load_config(None).unwrap();
        assert_eq!(config.num_players, 2);
    }

    #[test]
    fn test_write_and_load_config() {
        let tmp = std::env::temp_dir().join("cp_test_config.toml");
        write_default_config(&tmp, Some(ConfigProfile::Standard)).unwrap();
        let content = std::fs::read_to_string(&tmp).unwrap();
        assert!(content.contains("num_rounds = 1000"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_config_profile_from_arg() {
        let profile: ConfigProfile = ConfigProfileArg::Smoke.into();
        assert_eq!(profile, ConfigProfile::Smoke);
    }

    #[test]
    fn test_config_builder_seed() {
        let config = ConfigBuilder::new().seed(123).build();
        assert_eq!(config.global.bootstrap.seed, Some(123));
        assert_eq!(config.global.monte_carlo.seed, Some(123));
    }

    #[test]
    fn test_config_builder_checkpoint() {
        let config = ConfigBuilder::new()
            .checkpoint(true)
            .checkpoint_dir("/tmp/ckpt")
            .build();
        assert!(config.checkpoint);
        assert_eq!(config.checkpoint_dir, "/tmp/ckpt");
    }
}