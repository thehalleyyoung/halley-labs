//! CLI configuration: loading, validation, and defaults.
//!
//! Configuration can come from a TOML or JSON file, be overridden from the
//! command line, and is validated before use.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ── Main configuration struct ───────────────────────────────────────────────

/// Complete CLI configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CliConfig {
    // Audio output
    pub output_format: OutputFormat,
    pub sample_rate: u32,
    pub bit_depth: u16,
    pub channels: u16,
    pub buffer_size: u32,

    // Compiler
    pub optimization_level: u8,
    pub cognitive_load_budget: u8,
    pub hearing_profile: HearingProfile,

    // Flags
    pub verbose: bool,
    pub quiet: bool,

    // Paths
    pub output_dir: Option<PathBuf>,
    pub stdlib_path: Option<PathBuf>,

    // Rendering
    pub max_render_duration: Option<f64>,
    pub normalize_output: bool,
    pub dither: bool,

    // WCET
    pub wcet_budget_ms: f64,
    pub skip_wcet: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            output_format: OutputFormat::Wav,
            sample_rate: 44100,
            bit_depth: 16,
            channels: 2,
            buffer_size: 512,
            optimization_level: 2,
            cognitive_load_budget: 4,
            hearing_profile: HearingProfile::default(),
            verbose: false,
            quiet: false,
            output_dir: None,
            stdlib_path: None,
            max_render_duration: None,
            normalize_output: true,
            dither: false,
            wcet_budget_ms: 10.0,
            skip_wcet: false,
        }
    }
}

impl CliConfig {
    // ── I/O ────────────────────────────────────────────────────

    /// Load configuration from a file (TOML or JSON, detected by extension).
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let contents =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("json");

        match ext {
            "toml" => Self::from_toml(&contents),
            "json" => Self::from_json(&contents),
            other => bail!("Unsupported config format: .{other}"),
        }
    }

    /// Discover a config file by searching standard locations.
    pub fn discover() -> Option<Self> {
        let candidates = [
            PathBuf::from("sonitype.toml"),
            PathBuf::from("sonitype.json"),
            PathBuf::from(".sonitype.toml"),
            PathBuf::from(".sonitype.json"),
        ];

        for path in &candidates {
            if path.is_file() {
                if let Ok(cfg) = Self::load_from_file(path) {
                    log::info!("Loaded config from {}", path.display());
                    return Some(cfg);
                }
            }
        }

        // Try home directory.
        if let Some(home) = home_dir() {
            let home_cfg = home.join(".config").join("sonitype").join("config.toml");
            if home_cfg.is_file() {
                if let Ok(cfg) = Self::load_from_file(&home_cfg) {
                    log::info!("Loaded config from {}", home_cfg.display());
                    return Some(cfg);
                }
            }
        }

        None
    }

    /// Serialise to TOML string (best-effort using JSON round-trip since
    /// we don't depend on the `toml` crate at runtime).
    pub fn to_toml_string(&self) -> Result<String> {
        // Emit a simplified TOML manually for the fields we care about.
        let mut out = String::new();
        out.push_str("# SoniType configuration\n\n");
        out.push_str(&format!(
            "output_format = \"{}\"\n",
            serde_json::to_string(&self.output_format)?.trim_matches('"')
        ));
        out.push_str(&format!("sample_rate = {}\n", self.sample_rate));
        out.push_str(&format!("bit_depth = {}\n", self.bit_depth));
        out.push_str(&format!("channels = {}\n", self.channels));
        out.push_str(&format!("buffer_size = {}\n", self.buffer_size));
        out.push_str(&format!("optimization_level = {}\n", self.optimization_level));
        out.push_str(&format!(
            "cognitive_load_budget = {}\n",
            self.cognitive_load_budget
        ));
        out.push_str(&format!("verbose = {}\n", self.verbose));
        out.push_str(&format!("quiet = {}\n", self.quiet));
        out.push_str(&format!("normalize_output = {}\n", self.normalize_output));
        out.push_str(&format!("dither = {}\n", self.dither));
        out.push_str(&format!("wcet_budget_ms = {}\n", self.wcet_budget_ms));
        out.push_str(&format!("skip_wcet = {}\n", self.skip_wcet));
        Ok(out)
    }

    /// Serialise to JSON.
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("serialising config to JSON")
    }

    // ── Parsing helpers ───────────────────────────────────────

    fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).context("parsing JSON config")
    }

    /// Minimal TOML parser — handles flat `key = value` lines.
    fn from_toml(s: &str) -> Result<Self> {
        // Convert simple TOML to JSON then deserialise.
        let mut map = serde_json::Map::new();
        for line in s.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((key, val)) = line.split_once('=') {
                let key = key.trim().to_string();
                let val = val.trim();
                let json_val = if val == "true" {
                    serde_json::Value::Bool(true)
                } else if val == "false" {
                    serde_json::Value::Bool(false)
                } else if let Ok(n) = val.parse::<i64>() {
                    serde_json::Value::Number(n.into())
                } else if let Ok(n) = val.parse::<f64>() {
                    serde_json::json!(n)
                } else {
                    // Strip quotes.
                    let trimmed = val.trim_matches('"').trim_matches('\'');
                    serde_json::Value::String(trimmed.to_string())
                };
                map.insert(key, json_val);
            }
        }
        let json = serde_json::Value::Object(map);
        serde_json::from_value(json).context("parsing TOML config")
    }

    // ── Validation ─────────────────────────────────────────────

    /// Validate all configuration fields, returning an error describing the
    /// first invalid value found.
    pub fn validate(&self) -> Result<()> {
        // Sample rate
        if ![8000, 11025, 16000, 22050, 44100, 48000, 88200, 96000, 192000]
            .contains(&self.sample_rate)
        {
            bail!(
                "Unsupported sample rate {}. Use one of: 8000, 11025, 16000, 22050, 44100, 48000, 88200, 96000, 192000",
                self.sample_rate
            );
        }

        // Bit depth
        if ![16, 24, 32].contains(&self.bit_depth) {
            bail!(
                "Unsupported bit depth {}. Use 16, 24, or 32.",
                self.bit_depth
            );
        }

        // Channels
        if self.channels == 0 || self.channels > 8 {
            bail!("Channels must be 1–8, got {}", self.channels);
        }

        // Buffer size (must be power of 2)
        if self.buffer_size == 0 || (self.buffer_size & (self.buffer_size - 1)) != 0 {
            bail!(
                "Buffer size must be a power of 2, got {}",
                self.buffer_size
            );
        }
        if !(64..=4096).contains(&self.buffer_size) {
            bail!("Buffer size must be 64–4096, got {}", self.buffer_size);
        }

        // Optimisation level
        if self.optimization_level > 3 {
            bail!(
                "Optimisation level must be 0–3, got {}",
                self.optimization_level
            );
        }

        // Cognitive load budget (Miller's 7±2 → 1–9, practical 1–7)
        if self.cognitive_load_budget == 0 || self.cognitive_load_budget > 7 {
            bail!(
                "Cognitive load budget must be 1–7, got {}",
                self.cognitive_load_budget
            );
        }

        // WCET
        if self.wcet_budget_ms <= 0.0 {
            bail!("WCET budget must be > 0, got {}", self.wcet_budget_ms);
        }

        // Verbose + quiet
        if self.verbose && self.quiet {
            bail!("Cannot set both --verbose and --quiet");
        }

        Ok(())
    }
}

// ── Output format ───────────────────────────────────────────────────────────

/// Supported audio output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Wav,
    Raw,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Wav
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wav => write!(f, "WAV"),
            Self::Raw => write!(f, "raw PCM"),
        }
    }
}

// ── Hearing profile ─────────────────────────────────────────────────────────

/// Hearing profile: either a named preset or inline audiogram data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HearingProfile {
    Named(String),
    Custom(CustomAudiogram),
}

impl Default for HearingProfile {
    fn default() -> Self {
        HearingProfile::Named("normal".into())
    }
}

/// Inline audiogram: per-frequency hearing threshold offsets in dB HL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomAudiogram {
    /// Frequencies in Hz.
    pub frequencies: Vec<f64>,
    /// Thresholds in dB HL (positive = hearing loss).
    pub thresholds_db_hl: Vec<f64>,
}

// ── Config Loader ───────────────────────────────────────────────────────────

/// Utility for searching standard locations for configuration files.
pub struct ConfigLoader;

impl ConfigLoader {
    /// Return the list of candidate paths, in priority order.
    pub fn candidate_paths() -> Vec<PathBuf> {
        let mut paths = vec![
            PathBuf::from("sonitype.toml"),
            PathBuf::from("sonitype.json"),
            PathBuf::from(".sonitype.toml"),
            PathBuf::from(".sonitype.json"),
        ];
        if let Some(home) = home_dir() {
            paths.push(home.join(".config/sonitype/config.toml"));
            paths.push(home.join(".config/sonitype/config.json"));
        }
        paths
    }

    /// Load the first config file found, or return `None`.
    pub fn load_first() -> Option<CliConfig> {
        CliConfig::discover()
    }

    /// Merge a base config with overrides from a second config, keeping
    /// non-default values from `overrides`.
    pub fn merge(base: &CliConfig, overrides: &CliConfig) -> CliConfig {
        let def = CliConfig::default();
        CliConfig {
            output_format: if overrides.output_format != def.output_format {
                overrides.output_format
            } else {
                base.output_format
            },
            sample_rate: if overrides.sample_rate != def.sample_rate {
                overrides.sample_rate
            } else {
                base.sample_rate
            },
            bit_depth: if overrides.bit_depth != def.bit_depth {
                overrides.bit_depth
            } else {
                base.bit_depth
            },
            channels: if overrides.channels != def.channels {
                overrides.channels
            } else {
                base.channels
            },
            buffer_size: if overrides.buffer_size != def.buffer_size {
                overrides.buffer_size
            } else {
                base.buffer_size
            },
            optimization_level: if overrides.optimization_level != def.optimization_level {
                overrides.optimization_level
            } else {
                base.optimization_level
            },
            cognitive_load_budget: if overrides.cognitive_load_budget != def.cognitive_load_budget {
                overrides.cognitive_load_budget
            } else {
                base.cognitive_load_budget
            },
            hearing_profile: base.hearing_profile.clone(),
            verbose: overrides.verbose || base.verbose,
            quiet: overrides.quiet || base.quiet,
            output_dir: overrides.output_dir.clone().or_else(|| base.output_dir.clone()),
            stdlib_path: overrides.stdlib_path.clone().or_else(|| base.stdlib_path.clone()),
            max_render_duration: overrides.max_render_duration.or(base.max_render_duration),
            normalize_output: if overrides.normalize_output != def.normalize_output {
                overrides.normalize_output
            } else {
                base.normalize_output
            },
            dither: overrides.dither || base.dither,
            wcet_budget_ms: if (overrides.wcet_budget_ms - def.wcet_budget_ms).abs() > f64::EPSILON
            {
                overrides.wcet_budget_ms
            } else {
                base.wcet_budget_ms
            },
            skip_wcet: overrides.skip_wcet || base.skip_wcet,
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        CliConfig::default().validate().unwrap();
    }

    #[test]
    fn invalid_sample_rate() {
        let mut cfg = CliConfig::default();
        cfg.sample_rate = 12345;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_bit_depth() {
        let mut cfg = CliConfig::default();
        cfg.bit_depth = 8;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_channels_zero() {
        let mut cfg = CliConfig::default();
        cfg.channels = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_channels_too_many() {
        let mut cfg = CliConfig::default();
        cfg.channels = 10;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_buffer_size_not_power_of_two() {
        let mut cfg = CliConfig::default();
        cfg.buffer_size = 300;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_opt_level() {
        let mut cfg = CliConfig::default();
        cfg.optimization_level = 5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_cognitive_budget() {
        let mut cfg = CliConfig::default();
        cfg.cognitive_load_budget = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn verbose_and_quiet_conflict() {
        let mut cfg = CliConfig::default();
        cfg.verbose = true;
        cfg.quiet = true;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn json_round_trip() {
        let cfg = CliConfig::default();
        let json = cfg.to_json_string().unwrap();
        let parsed: CliConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.sample_rate, cfg.sample_rate);
        assert_eq!(parsed.bit_depth, cfg.bit_depth);
    }

    #[test]
    fn from_json_partial() {
        let json = r#"{ "sample_rate": 48000 }"#;
        let cfg: CliConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.sample_rate, 48000);
        // All other fields should be defaults.
        assert_eq!(cfg.bit_depth, 16);
    }

    #[test]
    fn toml_round_trip() {
        let cfg = CliConfig::default();
        let toml = cfg.to_toml_string().unwrap();
        let parsed = CliConfig::from_toml(&toml).unwrap();
        assert_eq!(parsed.sample_rate, cfg.sample_rate);
    }

    #[test]
    fn toml_parsing_ignores_comments() {
        let toml = "# A comment\nsample_rate = 48000\n";
        let cfg = CliConfig::from_toml(toml).unwrap();
        assert_eq!(cfg.sample_rate, 48000);
    }

    #[test]
    fn merge_overrides() {
        let base = CliConfig::default();
        let mut over = CliConfig::default();
        over.sample_rate = 48000;
        let merged = ConfigLoader::merge(&base, &over);
        assert_eq!(merged.sample_rate, 48000);
        assert_eq!(merged.bit_depth, 16); // kept from base/default
    }

    #[test]
    fn candidate_paths_non_empty() {
        assert!(!ConfigLoader::candidate_paths().is_empty());
    }

    #[test]
    fn output_format_display() {
        assert_eq!(OutputFormat::Wav.to_string(), "WAV");
        assert_eq!(OutputFormat::Raw.to_string(), "raw PCM");
    }

    #[test]
    fn hearing_profile_default_is_normal() {
        match HearingProfile::default() {
            HearingProfile::Named(n) => assert_eq!(n, "normal"),
            _ => panic!("expected named"),
        }
    }

    #[test]
    fn invalid_wcet_budget() {
        let mut cfg = CliConfig::default();
        cfg.wcet_budget_ms = -1.0;
        assert!(cfg.validate().is_err());
    }
}
