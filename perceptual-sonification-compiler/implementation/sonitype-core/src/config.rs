//! Configuration system for the SoniType compiler.
//!
//! Provides builder-pattern configs for the compiler, psychoacoustic model,
//! renderer, optimizer, and accessibility settings. All configs support
//! serde serialization, default values, and validation.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::ConfigError;

// ===========================================================================
// Compiler config
// ===========================================================================

/// Optimization level for the compiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization; fastest compilation.
    O0,
    /// Basic optimization (constant folding, dead code elimination).
    O1,
    /// Full optimization (psychoacoustic cost minimization).
    O2,
    /// Aggressive optimization with longer compile times.
    O3,
}

impl Default for OptimizationLevel {
    fn default() -> Self { Self::O1 }
}

impl fmt::Display for OptimizationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::O0 => write!(f, "O0"),
            Self::O1 => write!(f, "O1"),
            Self::O2 => write!(f, "O2"),
            Self::O3 => write!(f, "O3"),
        }
    }
}

/// Top-level compiler configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompilerConfig {
    pub optimization_level: OptimizationLevel,
    pub target_sample_rate: u32,
    pub buffer_size: usize,
    pub max_streams: usize,
    pub cognitive_load_budget: f64,
    pub debug_mode: bool,
    pub verbose: bool,
    pub emit_ir: bool,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::O1,
            target_sample_rate: 44100,
            buffer_size: 1024,
            max_streams: 8,
            cognitive_load_budget: 4.0,
            debug_mode: false,
            verbose: false,
            emit_ir: false,
        }
    }
}

impl CompilerConfig {
    pub fn builder() -> CompilerConfigBuilder {
        CompilerConfigBuilder(Self::default())
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.target_sample_rate == 0 {
            return Err(ConfigError::invalid_field("target_sample_rate", "must be > 0"));
        }
        if self.buffer_size == 0 {
            return Err(ConfigError::invalid_field("buffer_size", "must be > 0"));
        }
        if self.max_streams == 0 {
            return Err(ConfigError::invalid_field("max_streams", "must be > 0"));
        }
        if self.cognitive_load_budget <= 0.0 {
            return Err(ConfigError::invalid_field("cognitive_load_budget", "must be > 0"));
        }
        Ok(())
    }

    /// Apply overrides from environment variables (SONITYPE_ prefix).
    pub fn apply_env_overrides(&mut self) {
        if let Ok(val) = std::env::var("SONITYPE_SAMPLE_RATE") {
            if let Ok(sr) = val.parse::<u32>() { self.target_sample_rate = sr; }
        }
        if let Ok(val) = std::env::var("SONITYPE_BUFFER_SIZE") {
            if let Ok(bs) = val.parse::<usize>() { self.buffer_size = bs; }
        }
        if let Ok(val) = std::env::var("SONITYPE_MAX_STREAMS") {
            if let Ok(ms) = val.parse::<usize>() { self.max_streams = ms; }
        }
        if let Ok(val) = std::env::var("SONITYPE_DEBUG") {
            self.debug_mode = val == "1" || val.eq_ignore_ascii_case("true");
        }
        if let Ok(val) = std::env::var("SONITYPE_OPT_LEVEL") {
            match val.as_str() {
                "0" | "O0" => self.optimization_level = OptimizationLevel::O0,
                "1" | "O1" => self.optimization_level = OptimizationLevel::O1,
                "2" | "O2" => self.optimization_level = OptimizationLevel::O2,
                "3" | "O3" => self.optimization_level = OptimizationLevel::O3,
                _ => {}
            }
        }
    }
}

/// Builder for CompilerConfig.
pub struct CompilerConfigBuilder(CompilerConfig);

impl CompilerConfigBuilder {
    pub fn optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.0.optimization_level = level; self
    }
    pub fn target_sample_rate(mut self, rate: u32) -> Self {
        self.0.target_sample_rate = rate; self
    }
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.0.buffer_size = size; self
    }
    pub fn max_streams(mut self, n: usize) -> Self {
        self.0.max_streams = n; self
    }
    pub fn cognitive_load_budget(mut self, budget: f64) -> Self {
        self.0.cognitive_load_budget = budget; self
    }
    pub fn debug_mode(mut self, on: bool) -> Self {
        self.0.debug_mode = on; self
    }
    pub fn verbose(mut self, on: bool) -> Self {
        self.0.verbose = on; self
    }
    pub fn emit_ir(mut self, on: bool) -> Self {
        self.0.emit_ir = on; self
    }
    pub fn build(self) -> Result<CompilerConfig, ConfigError> {
        self.0.validate()?;
        Ok(self.0)
    }
}

impl fmt::Display for CompilerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CompilerConfig(opt={}, sr={}, buf={}, streams={})",
            self.optimization_level, self.target_sample_rate,
            self.buffer_size, self.max_streams)
    }
}

// ===========================================================================
// Psychoacoustic config
// ===========================================================================

/// Selection of psychoacoustic masking model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaskingModel {
    Simple,
    Zwicker,
    JohnstonTransform,
    Custom(String),
}

impl Default for MaskingModel {
    fn default() -> Self { Self::Simple }
}

/// Psychoacoustic model configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PsychoacousticConfig {
    pub masking_model: MaskingModel,
    pub jnd_pitch_factor: f64,
    pub jnd_loudness_factor: f64,
    pub jnd_temporal_ms: f64,
    pub min_segregation_score: f64,
    pub max_cognitive_load: f64,
    pub masking_margin_db: f64,
    pub frequency_range_min: f64,
    pub frequency_range_max: f64,
}

impl Default for PsychoacousticConfig {
    fn default() -> Self {
        Self {
            masking_model: MaskingModel::default(),
            jnd_pitch_factor: 1.0,
            jnd_loudness_factor: 1.0,
            jnd_temporal_ms: 2.0,
            min_segregation_score: 0.6,
            max_cognitive_load: 4.0,
            masking_margin_db: 6.0,
            frequency_range_min: 80.0,
            frequency_range_max: 12000.0,
        }
    }
}

impl PsychoacousticConfig {
    pub fn builder() -> PsychoacousticConfigBuilder {
        PsychoacousticConfigBuilder(Self::default())
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.jnd_pitch_factor <= 0.0 {
            return Err(ConfigError::invalid_field("jnd_pitch_factor", "must be > 0"));
        }
        if self.frequency_range_min >= self.frequency_range_max {
            return Err(ConfigError::invalid_field("frequency_range", "min must be < max"));
        }
        if self.max_cognitive_load <= 0.0 {
            return Err(ConfigError::invalid_field("max_cognitive_load", "must be > 0"));
        }
        Ok(())
    }
}

pub struct PsychoacousticConfigBuilder(PsychoacousticConfig);

impl PsychoacousticConfigBuilder {
    pub fn masking_model(mut self, model: MaskingModel) -> Self {
        self.0.masking_model = model; self
    }
    pub fn jnd_pitch_factor(mut self, f: f64) -> Self {
        self.0.jnd_pitch_factor = f; self
    }
    pub fn jnd_loudness_factor(mut self, f: f64) -> Self {
        self.0.jnd_loudness_factor = f; self
    }
    pub fn jnd_temporal_ms(mut self, ms: f64) -> Self {
        self.0.jnd_temporal_ms = ms; self
    }
    pub fn min_segregation_score(mut self, s: f64) -> Self {
        self.0.min_segregation_score = s; self
    }
    pub fn max_cognitive_load(mut self, l: f64) -> Self {
        self.0.max_cognitive_load = l; self
    }
    pub fn masking_margin_db(mut self, db: f64) -> Self {
        self.0.masking_margin_db = db; self
    }
    pub fn frequency_range(mut self, min: f64, max: f64) -> Self {
        self.0.frequency_range_min = min;
        self.0.frequency_range_max = max;
        self
    }
    pub fn build(self) -> Result<PsychoacousticConfig, ConfigError> {
        self.0.validate()?;
        Ok(self.0)
    }
}

impl fmt::Display for PsychoacousticConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PsychoacousticConfig(model={:?}, freq=[{:.0}, {:.0}])",
            self.masking_model, self.frequency_range_min, self.frequency_range_max)
    }
}

// ===========================================================================
// Renderer config
// ===========================================================================

/// Audio output format.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Wav,
    Raw,
    Stream,
}

impl Default for OutputFormat {
    fn default() -> Self { Self::Wav }
}

/// Renderer configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RendererConfig {
    pub output_format: OutputFormat,
    pub channel_count: usize,
    pub bit_depth: u16,
    pub sample_rate: u32,
    pub buffer_size: usize,
    pub real_time: bool,
    pub normalize_output: bool,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            output_format: OutputFormat::default(),
            channel_count: 2,
            bit_depth: 16,
            sample_rate: 44100,
            buffer_size: 1024,
            real_time: false,
            normalize_output: true,
        }
    }
}

impl RendererConfig {
    pub fn builder() -> RendererConfigBuilder {
        RendererConfigBuilder(Self::default())
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.channel_count == 0 {
            return Err(ConfigError::invalid_field("channel_count", "must be > 0"));
        }
        if ![8, 16, 24, 32].contains(&self.bit_depth) {
            return Err(ConfigError::invalid_field("bit_depth", "must be 8, 16, 24, or 32"));
        }
        if self.sample_rate == 0 {
            return Err(ConfigError::invalid_field("sample_rate", "must be > 0"));
        }
        Ok(())
    }

    /// Bytes per sample for the configured bit depth.
    pub fn bytes_per_sample(&self) -> usize {
        (self.bit_depth / 8) as usize
    }

    /// Total frame size in bytes (all channels * bytes_per_sample).
    pub fn frame_size_bytes(&self) -> usize {
        self.channel_count * self.bytes_per_sample()
    }
}

pub struct RendererConfigBuilder(RendererConfig);

impl RendererConfigBuilder {
    pub fn output_format(mut self, fmt: OutputFormat) -> Self {
        self.0.output_format = fmt; self
    }
    pub fn channel_count(mut self, n: usize) -> Self {
        self.0.channel_count = n; self
    }
    pub fn bit_depth(mut self, bits: u16) -> Self {
        self.0.bit_depth = bits; self
    }
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.0.sample_rate = rate; self
    }
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.0.buffer_size = size; self
    }
    pub fn real_time(mut self, rt: bool) -> Self {
        self.0.real_time = rt; self
    }
    pub fn normalize_output(mut self, on: bool) -> Self {
        self.0.normalize_output = on; self
    }
    pub fn build(self) -> Result<RendererConfig, ConfigError> {
        self.0.validate()?;
        Ok(self.0)
    }
}

impl fmt::Display for RendererConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RendererConfig({}ch, {}bit, {} Hz, buf={})",
            self.channel_count, self.bit_depth, self.sample_rate, self.buffer_size)
    }
}

// ===========================================================================
// Optimizer config
// ===========================================================================

/// Search strategy for the optimizer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStrategy {
    GradientDescent,
    SimulatedAnnealing,
    GeneticAlgorithm,
    BayesianOptimization,
    GridSearch,
    RandomSearch,
}

impl Default for SearchStrategy {
    fn default() -> Self { Self::SimulatedAnnealing }
}

/// Optimizer configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub search_strategy: SearchStrategy,
    pub time_budget_ms: u64,
    pub population_size: usize,
    pub temperature_initial: f64,
    pub temperature_decay: f64,
    pub seed: Option<u64>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            search_strategy: SearchStrategy::default(),
            time_budget_ms: 5000,
            population_size: 50,
            temperature_initial: 100.0,
            temperature_decay: 0.995,
            seed: None,
        }
    }
}

impl OptimizerConfig {
    pub fn builder() -> OptimizerConfigBuilder {
        OptimizerConfigBuilder(Self::default())
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_iterations == 0 {
            return Err(ConfigError::invalid_field("max_iterations", "must be > 0"));
        }
        if self.convergence_threshold <= 0.0 {
            return Err(ConfigError::invalid_field("convergence_threshold", "must be > 0"));
        }
        if self.time_budget_ms == 0 {
            return Err(ConfigError::invalid_field("time_budget_ms", "must be > 0"));
        }
        Ok(())
    }
}

pub struct OptimizerConfigBuilder(OptimizerConfig);

impl OptimizerConfigBuilder {
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.0.max_iterations = n; self
    }
    pub fn convergence_threshold(mut self, t: f64) -> Self {
        self.0.convergence_threshold = t; self
    }
    pub fn search_strategy(mut self, s: SearchStrategy) -> Self {
        self.0.search_strategy = s; self
    }
    pub fn time_budget_ms(mut self, ms: u64) -> Self {
        self.0.time_budget_ms = ms; self
    }
    pub fn population_size(mut self, n: usize) -> Self {
        self.0.population_size = n; self
    }
    pub fn seed(mut self, s: u64) -> Self {
        self.0.seed = Some(s); self
    }
    pub fn build(self) -> Result<OptimizerConfig, ConfigError> {
        self.0.validate()?;
        Ok(self.0)
    }
}

impl fmt::Display for OptimizerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OptimizerConfig(strategy={:?}, max_iter={}, budget={}ms)",
            self.search_strategy, self.max_iterations, self.time_budget_ms)
    }
}

// ===========================================================================
// Accessibility config
// ===========================================================================

/// Hearing profile presets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HearingProfile {
    Normal,
    MildLoss,
    ModerateLoss,
    SevereLoss,
    HighFrequencyLoss,
    Custom {
        freq_db_offsets: Vec<(u32, f64)>,
    },
}

impl Default for HearingProfile {
    fn default() -> Self { Self::Normal }
}

/// Accessibility configuration for adapting sonification to listener needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AccessibilityConfig {
    pub hearing_profile: HearingProfile,
    pub frequency_range_min: f64,
    pub frequency_range_max: f64,
    pub amplitude_min: f64,
    pub amplitude_max: f64,
    pub prefer_low_frequencies: bool,
    pub haptic_feedback: bool,
    pub visual_feedback: bool,
}

impl Default for AccessibilityConfig {
    fn default() -> Self {
        Self {
            hearing_profile: HearingProfile::default(),
            frequency_range_min: 200.0,
            frequency_range_max: 8000.0,
            amplitude_min: 0.1,
            amplitude_max: 0.9,
            prefer_low_frequencies: false,
            haptic_feedback: false,
            visual_feedback: true,
        }
    }
}

impl AccessibilityConfig {
    pub fn builder() -> AccessibilityConfigBuilder {
        AccessibilityConfigBuilder(Self::default())
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.frequency_range_min >= self.frequency_range_max {
            return Err(ConfigError::invalid_field("frequency_range", "min must be < max"));
        }
        if self.amplitude_min < 0.0 || self.amplitude_max > 1.0 {
            return Err(ConfigError::invalid_field("amplitude", "must be in [0, 1]"));
        }
        if self.amplitude_min >= self.amplitude_max {
            return Err(ConfigError::invalid_field("amplitude_range", "min must be < max"));
        }
        Ok(())
    }

    /// Get a frequency scaling factor based on the hearing profile.
    pub fn frequency_adjustment(&self, freq: f64) -> f64 {
        match &self.hearing_profile {
            HearingProfile::Normal => freq,
            HearingProfile::HighFrequencyLoss => {
                if freq > 4000.0 { freq * 0.5 } else { freq }
            }
            HearingProfile::MildLoss => freq.clamp(self.frequency_range_min, self.frequency_range_max),
            HearingProfile::ModerateLoss => {
                freq.clamp(self.frequency_range_min * 1.5, self.frequency_range_max * 0.7)
            }
            HearingProfile::SevereLoss => {
                freq.clamp(self.frequency_range_min * 2.0, self.frequency_range_max * 0.5)
            }
            HearingProfile::Custom { freq_db_offsets: _ } => freq,
        }
    }
}

pub struct AccessibilityConfigBuilder(AccessibilityConfig);

impl AccessibilityConfigBuilder {
    pub fn hearing_profile(mut self, p: HearingProfile) -> Self {
        self.0.hearing_profile = p; self
    }
    pub fn frequency_range(mut self, min: f64, max: f64) -> Self {
        self.0.frequency_range_min = min;
        self.0.frequency_range_max = max;
        self
    }
    pub fn amplitude_range(mut self, min: f64, max: f64) -> Self {
        self.0.amplitude_min = min;
        self.0.amplitude_max = max;
        self
    }
    pub fn haptic_feedback(mut self, on: bool) -> Self {
        self.0.haptic_feedback = on; self
    }
    pub fn visual_feedback(mut self, on: bool) -> Self {
        self.0.visual_feedback = on; self
    }
    pub fn build(self) -> Result<AccessibilityConfig, ConfigError> {
        self.0.validate()?;
        Ok(self.0)
    }
}

impl fmt::Display for AccessibilityConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AccessibilityConfig(profile={:?}, freq=[{:.0}, {:.0}])",
            self.hearing_profile, self.frequency_range_min, self.frequency_range_max)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiler_config_default_valid() {
        assert!(CompilerConfig::default().validate().is_ok());
    }

    #[test]
    fn compiler_config_builder() {
        let cfg = CompilerConfig::builder()
            .optimization_level(OptimizationLevel::O2)
            .target_sample_rate(48000)
            .buffer_size(512)
            .build()
            .unwrap();
        assert_eq!(cfg.optimization_level, OptimizationLevel::O2);
        assert_eq!(cfg.target_sample_rate, 48000);
    }

    #[test]
    fn compiler_config_invalid() {
        let result = CompilerConfig::builder()
            .target_sample_rate(0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn psychoacoustic_config_default_valid() {
        assert!(PsychoacousticConfig::default().validate().is_ok());
    }

    #[test]
    fn psychoacoustic_config_builder() {
        let cfg = PsychoacousticConfig::builder()
            .masking_model(MaskingModel::Zwicker)
            .frequency_range(100.0, 10000.0)
            .build()
            .unwrap();
        assert_eq!(cfg.masking_model, MaskingModel::Zwicker);
    }

    #[test]
    fn renderer_config_default_valid() {
        assert!(RendererConfig::default().validate().is_ok());
    }

    #[test]
    fn renderer_config_builder() {
        let cfg = RendererConfig::builder()
            .channel_count(1)
            .bit_depth(24)
            .sample_rate(96000)
            .real_time(true)
            .build()
            .unwrap();
        assert_eq!(cfg.channel_count, 1);
        assert_eq!(cfg.bytes_per_sample(), 3);
    }

    #[test]
    fn renderer_config_invalid_bit_depth() {
        let result = RendererConfig::builder().bit_depth(12).build();
        assert!(result.is_err());
    }

    #[test]
    fn optimizer_config_default_valid() {
        assert!(OptimizerConfig::default().validate().is_ok());
    }

    #[test]
    fn optimizer_config_builder() {
        let cfg = OptimizerConfig::builder()
            .search_strategy(SearchStrategy::GeneticAlgorithm)
            .max_iterations(500)
            .seed(42)
            .build()
            .unwrap();
        assert_eq!(cfg.search_strategy, SearchStrategy::GeneticAlgorithm);
        assert_eq!(cfg.seed, Some(42));
    }

    #[test]
    fn accessibility_config_default_valid() {
        assert!(AccessibilityConfig::default().validate().is_ok());
    }

    #[test]
    fn accessibility_config_builder() {
        let cfg = AccessibilityConfig::builder()
            .hearing_profile(HearingProfile::HighFrequencyLoss)
            .frequency_range(300.0, 6000.0)
            .build()
            .unwrap();
        assert_eq!(cfg.hearing_profile, HearingProfile::HighFrequencyLoss);
    }

    #[test]
    fn accessibility_frequency_adjustment() {
        let cfg = AccessibilityConfig {
            hearing_profile: HearingProfile::HighFrequencyLoss,
            ..Default::default()
        };
        let adjusted = cfg.frequency_adjustment(8000.0);
        assert!(adjusted < 8000.0, "High freq should be lowered");
    }

    #[test]
    fn renderer_frame_size() {
        let cfg = RendererConfig { channel_count: 2, bit_depth: 16, ..Default::default() };
        assert_eq!(cfg.frame_size_bytes(), 4);
    }

    #[test]
    fn serde_roundtrip_compiler_config() {
        let cfg = CompilerConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: CompilerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn serde_roundtrip_renderer_config() {
        let cfg = RendererConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: RendererConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }
}
