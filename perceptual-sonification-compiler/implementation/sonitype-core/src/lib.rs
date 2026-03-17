//! # SoniType Core
//!
//! Foundational crate for the SoniType perceptual sonification compiler.
//! Contains shared types, traits, error handling, configuration, audio primitives,
//! mathematical utilities, and ID management used across all compiler stages.

pub mod audio;
pub mod config;
pub mod error;
pub mod id;
pub mod math;
pub mod traits;
pub mod types;
pub mod units;

pub use audio::{AudioBuffer, AudioFrame, RingBuffer, WindowFunction};
pub use config::{
    AccessibilityConfig, CompilerConfig, OptimizerConfig, PsychoacousticConfig, RendererConfig,
};
pub use error::{
    OptimizationError, ParseError, PsychoacousticError, RuntimeError, SoniTypeError, TypeError,
    ValidationError,
};
pub use id::{GraphId, MappingId, NodeId, StreamId};
pub use types::{
    Amplitude, BarkBand, BufferSize, CognitiveLoadBudget, DataDistribution, DataSchema,
    DataValue, DecibelSpl, Duration, Frequency, MappingParameter, MelBand, Octave, Pan,
    PerceptualQualifier, Phase, PitchClass, SampleRate, SpatialPosition, StreamDescriptor,
};
pub use units::{
    amplitude_to_db_spl, bark_bandwidth, bark_to_hz, cents_to_ratio, db_spl_to_amplitude,
    erb_of_frequency, hz_to_bark, hz_to_erb_rate, hz_to_mel, hz_to_midi, mel_to_hz, midi_to_hz,
    ratio_to_cents, semitones_to_ratio,
};
