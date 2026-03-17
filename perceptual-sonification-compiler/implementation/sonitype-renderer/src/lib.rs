//! # sonitype-renderer
//!
//! Lock-free real-time audio graph executor and renderer for the SoniType
//! perceptual sonification compiler. Provides:
//!
//! - **Audio graph execution** with topological ordering and inter-node buffer management
//! - **Oscillators**: wavetable, band-limited, FM, additive synthesis
//! - **Filters**: biquad, state-variable, comb, cascaded higher-order
//! - **Envelopes**: ADSR, multi-segment, LFO, amplitude follower
//! - **Mixing**: stereo/multichannel mixers, crossfaders, send/return
//! - **Effects**: delay, chorus, reverb, compressor, limiter, distortion
//! - **Output**: WAV writer, metering (peak, RMS, LUFS)
//! - **Parameter management**: lock-free updates, smoothing, interpolation
//! - **Rendering**: offline, real-time, preview modes

pub mod parameter;
pub mod oscillators;
pub mod filters;
pub mod envelope;
pub mod mixer;
pub mod effects;
pub mod output;
pub mod executor;
pub mod render;
pub mod midi_output;
pub mod wav_output;

// Re-export key types for ergonomic access.
pub use executor::{AudioGraphExecutor, ExecutionContext, BufferPool, NodeProcessor};
pub use oscillators::{
    WavetableOscillator, SineOscillator, SawOscillator, SquareOscillator,
    TriangleOscillator, PulseOscillator, NoiseOscillator, FMOscillator,
    AdditiveOscillator,
};
pub use filters::{
    BiquadFilter, OnePoleFilter, StateVariableFilter, CombFilter,
    BiquadCascade, DCBlocker,
};
pub use envelope::{
    AdsrEnvelope, MultiSegmentEnvelope, LfoEnvelope, FollowerEnvelope,
};
pub use mixer::{
    StereoMixer, MultichannelMixer, Crossfader, ChannelSplitter,
    ChannelMerger, SendReturn,
};
pub use effects::{Delay, Chorus, Reverb, Compressor, Limiter, Distortion};
pub use output::{
    WavWriter, RawOutput, NullOutput, BufferedOutput, OutputFormat, Metering,
};
pub use parameter::{
    ParameterManager, Parameter, ParameterChange, ParameterInterpolator,
};
pub use render::{
    OfflineRenderer, RealTimeRenderer, PreviewRenderer, RenderSession,
};
pub use midi_output::{
    MidiOutputConfig, SoniMidiEvent, SoniMidiEventKind, MidiOutputError,
    hz_to_midi_note, midi_note_to_hz, loudness_to_velocity, pan_to_cc,
    write_midi_file,
};
pub use wav_output::{
    WavOutputConfig, WavBitDepth, WavOutputError,
    write_wav_file, read_wav_file,
};

/// Lightweight audio buffer used throughout the renderer.
/// Stored as interleaved f32 samples with a known channel count.
#[derive(Debug, Clone)]
pub struct AudioBuf {
    pub data: Vec<f32>,
    pub channels: usize,
    pub sample_rate: u32,
}

impl AudioBuf {
    pub fn new(frames: usize, channels: usize, sample_rate: u32) -> Self {
        Self {
            data: vec![0.0; frames * channels],
            channels,
            sample_rate,
        }
    }

    #[inline]
    pub fn frames(&self) -> usize {
        if self.channels == 0 { 0 } else { self.data.len() / self.channels }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn zero(&mut self) {
        for s in self.data.iter_mut() {
            *s = 0.0;
        }
    }

    /// Get a single sample (frame, channel).
    #[inline]
    pub fn get(&self, frame: usize, channel: usize) -> f32 {
        self.data[frame * self.channels + channel]
    }

    /// Set a single sample (frame, channel).
    #[inline]
    pub fn set(&mut self, frame: usize, channel: usize, value: f32) {
        self.data[frame * self.channels + channel] = value;
    }

    /// Mix (add) another buffer into this one.
    pub fn mix_from(&mut self, other: &AudioBuf) {
        let n = self.data.len().min(other.data.len());
        for i in 0..n {
            self.data[i] += other.data[i];
        }
    }

    /// Scale all samples by a constant gain.
    pub fn apply_gain(&mut self, gain: f32) {
        for s in self.data.iter_mut() {
            *s *= gain;
        }
    }

    /// Copy contents from another buffer.
    pub fn copy_from_buf(&mut self, other: &AudioBuf) {
        let n = self.data.len().min(other.data.len());
        self.data[..n].copy_from_slice(&other.data[..n]);
    }
}

/// Convenience alias used across the crate.
pub type RendererResult<T> = Result<T, RendererError>;

/// Errors originating from the renderer.
#[derive(Debug, Clone)]
pub enum RendererError {
    BufferUnderrun,
    InvalidGraph(String),
    NodeNotFound(u64),
    ParameterNotFound(String),
    OutputError(String),
    WcetExceeded { actual_us: f64, budget_us: f64 },
    InvalidConfiguration(String),
}

impl std::fmt::Display for RendererError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferUnderrun => write!(f, "audio buffer underrun"),
            Self::InvalidGraph(msg) => write!(f, "invalid graph: {msg}"),
            Self::NodeNotFound(id) => write!(f, "node {id} not found"),
            Self::ParameterNotFound(name) => write!(f, "parameter '{name}' not found"),
            Self::OutputError(msg) => write!(f, "output error: {msg}"),
            Self::WcetExceeded { actual_us, budget_us } => {
                write!(f, "WCET exceeded: {actual_us:.1}µs > {budget_us:.1}µs")
            }
            Self::InvalidConfiguration(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for RendererError {}
