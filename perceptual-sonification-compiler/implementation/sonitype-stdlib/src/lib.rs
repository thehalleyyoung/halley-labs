//! # SoniType Standard Library
//!
//! Built-in mappings, scales, timbral palettes, and preset sonification recipes
//! for the SoniType perceptual sonification compiler.
//!
//! ## Modules
//!
//! - [`scales`] — Pitch scale mappings (linear, logarithmic, Bark, Mel, musical, MIDI, microtonal)
//! - [`timbres`] — Timbral palettes (additive, FM, noise band, subtractive, palettes)
//! - [`mappings`] — Data-to-sound mapping presets (pitch, loudness, temporal, timbre, spatial)
//! - [`presets`] — Preset sonification recipes (time series, categorical, scatter, alerts)
//! - [`data_adapters`] — Data source adapters (CSV, JSON, arrays, streaming, normalization)
//! - [`templates`] — Sonification templates with variable substitution and expansion
//! - [`validation`] — Psychoacoustic validation of scales, timbres, and mappings

pub mod scales;
pub mod timbres;
pub mod mappings;
pub mod presets;
pub mod data_adapters;
pub mod templates;
pub mod validation;

// Re-export primary types for ergonomic access
pub use scales::{
    LinearScale, LogarithmicScale, BarkScale, MelScale,
    MusicalScale, MidiScale, MicrotonalScale,
    ScalePreset, ScaleBuilder,
};

pub use timbres::{
    AdditiveTimbre, FMTimbre, NoiseBandTimbre, SubtractiveTimbre,
    TimbrePalette, TimbreInterpolator,
};

pub use mappings::{
    PitchMapping, LoudnessMapping, TemporalMapping, TimbreMapping,
    SpatialMapping, FilterMapping, CompositeMapping, MappingBuilder,
};

pub use presets::{
    TimeSeriesPreset, CategoricalPreset, ScatterplotPreset,
    HistogramPreset, CorrelationPreset, AlertPreset,
    NavigationPreset, PresetRegistry,
};

pub use data_adapters::{
    CsvDataSource, JsonDataSource, ArrayDataSource,
    StreamingDataSource, DataNormalizer,
};

pub use templates::{
    TemplateEngine, TemplateParameter, TemplateLibrary,
};

pub use validation::StdlibValidator;
