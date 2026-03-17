//! Core traits for the SoniType perceptual sonification compiler.
//!
//! Defines the key abstractions: psychoacoustic models, segregation predicates,
//! cost functions, optimizers, audio nodes, renderers, data sources,
//! mapping strategies, type checkers, code generators, and stream processors.

use crate::audio::AudioBuffer;
use crate::error::SoniTypeError;
use crate::types::{
    Amplitude, BarkBand, DataSchema, DataValue, Duration, Frequency, MappingParameter, Pan,
    StreamDescriptor,
};

// ===========================================================================
// Psychoacoustic model
// ===========================================================================

/// Provides psychoacoustic measurements used throughout the compiler.
pub trait PsychoacousticModel {
    /// Masking threshold in dB at the given frequency when a masker is present.
    fn masking_threshold(&self, masker_freq: Frequency, masker_level_db: f64, probe_freq: Frequency) -> f64;

    /// Just-noticeable difference in pitch (Hz) at the given frequency and level.
    fn jnd_pitch(&self, freq: Frequency, level_db: f64) -> f64;

    /// Just-noticeable difference in loudness (dB) at the given level.
    fn jnd_loudness(&self, level_db: f64) -> f64;

    /// Just-noticeable difference in temporal onset (seconds).
    fn jnd_temporal(&self) -> f64 {
        0.002 // ~2 ms default
    }

    /// Return the Bark band index for a given frequency.
    fn bark_band_for_frequency(&self, freq: Frequency) -> BarkBand {
        let bark = crate::units::hz_to_bark(freq.hz());
        let idx = (bark.round() as u8).min(23);
        BarkBand(idx)
    }

    /// Specific loudness at a given frequency and level (sone-like units).
    fn specific_loudness(&self, freq: Frequency, level_db: f64) -> f64 {
        let _ = freq;
        // Rough Stevens power-law approximation
        if level_db < 0.0 { return 0.0; }
        (level_db / 40.0).powf(0.6)
    }
}

// ===========================================================================
// Segregation predicate
// ===========================================================================

/// Predicate that checks whether two auditory streams will be perceptually
/// segregated (heard as distinct) by the listener.
pub trait SegregationPredicate {
    /// Check whether two streams have sufficient onset asynchrony to be segregated.
    fn check_onset_synchrony(&self, onset_diff: Duration) -> bool {
        onset_diff.seconds().abs() > 0.03 // 30 ms threshold
    }

    /// Check whether two streams have sufficiently different harmonic structure.
    fn check_harmonicity(&self, stream_a: &StreamDescriptor, stream_b: &StreamDescriptor) -> bool {
        let fa = stream_a.frequency.hz();
        let fb = stream_b.frequency.hz();
        if fa <= 0.0 || fb <= 0.0 { return true; }
        // Always take ratio >= 1 so we compare against integer harmonics
        let ratio = if fa >= fb { fa / fb } else { fb / fa };
        let nearest_harmonic = ratio.round();
        if nearest_harmonic < 1.0 { return true; }
        (ratio - nearest_harmonic).abs() > 0.05
    }

    /// Check whether two streams are in different critical bands.
    fn check_spectral_proximity(&self, freq_a: Frequency, freq_b: Frequency) -> bool {
        let bark_a = crate::units::hz_to_bark(freq_a.hz());
        let bark_b = crate::units::hz_to_bark(freq_b.hz());
        (bark_a - bark_b).abs() > 1.0
    }

    /// Check common-fate grouping: streams with correlated modulation group together.
    /// Returns true if they do NOT share common fate (i.e., are segregated).
    fn check_common_fate(&self, modulation_correlation: f64) -> bool {
        modulation_correlation.abs() < 0.5
    }

    /// Evaluate all segregation checks and return a summary score [0, 1].
    /// 1.0 = fully segregated, 0.0 = fully fused.
    fn evaluate_all(
        &self,
        stream_a: &StreamDescriptor,
        stream_b: &StreamDescriptor,
        onset_diff: Duration,
        modulation_correlation: f64,
    ) -> f64 {
        let mut score = 0.0;
        let checks = 4.0;
        if self.check_onset_synchrony(onset_diff) { score += 1.0; }
        if self.check_harmonicity(stream_a, stream_b) { score += 1.0; }
        if self.check_spectral_proximity(stream_a.frequency, stream_b.frequency) { score += 1.0; }
        if self.check_common_fate(modulation_correlation) { score += 1.0; }
        score / checks
    }
}

// ===========================================================================
// Cost function
// ===========================================================================

/// Evaluates the perceptual cost of a mapping configuration.
pub trait CostFunction {
    /// Compute total cost for a set of mapping parameters and stream descriptors.
    fn evaluate(&self, params: &[MappingParameter], streams: &[StreamDescriptor]) -> f64;

    /// Name of this cost function for logging.
    fn name(&self) -> &str;
}

// ===========================================================================
// Optimizer
// ===========================================================================

/// Result of an optimization run.
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    pub parameters: Vec<MappingParameter>,
    pub cost: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Optimizes mapping parameters subject to psychoacoustic constraints.
pub trait Optimizer {
    /// Run the optimizer and return the best configuration found.
    fn optimize(
        &mut self,
        initial: &[MappingParameter],
        streams: &[StreamDescriptor],
        cost_fn: &dyn CostFunction,
    ) -> Result<OptimizationResult, SoniTypeError>;

    /// Name/type of the optimizer.
    fn optimizer_name(&self) -> &str;
}

// ===========================================================================
// Audio node
// ===========================================================================

/// Type tag for audio processing nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioNodeType {
    Source,
    Effect,
    Mixer,
    Output,
    Custom(String),
}

/// A node in the audio render graph.
pub trait AudioNode: Send {
    /// Process a buffer of audio in-place. Returns the number of samples written.
    fn process(&mut self, buffer: &mut AudioBuffer) -> usize;

    /// Worst-case execution time bound in microseconds for processing one buffer.
    fn wcet_bound_us(&self) -> u64;

    /// The type of this node.
    fn node_type(&self) -> AudioNodeType;

    /// Human-readable name for debugging.
    fn node_name(&self) -> &str { "unnamed" }

    /// Number of input channels expected.
    fn input_channels(&self) -> usize { 0 }

    /// Number of output channels produced.
    fn output_channels(&self) -> usize { 1 }

    /// Reset internal state (e.g., clear delay lines).
    fn reset(&mut self) {}
}

// ===========================================================================
// Renderer
// ===========================================================================

/// Renders an audio graph to a sample buffer.
pub trait Renderer {
    /// Render one buffer-worth of audio from the graph.
    fn render(&mut self, output: &mut AudioBuffer) -> Result<(), SoniTypeError>;

    /// The sample rate this renderer is configured for.
    fn sample_rate(&self) -> u32;

    /// The buffer size this renderer uses.
    fn buffer_size(&self) -> usize;

    /// Total latency in samples from input to output.
    fn latency_samples(&self) -> usize { self.buffer_size() }
}

// ===========================================================================
// Data source
// ===========================================================================

/// Provides data values for sonification.
pub trait DataSource {
    /// Schema of the data provided.
    fn schema(&self) -> &DataSchema;

    /// Read the next row of data values. Returns None at end of stream.
    fn next_row(&mut self) -> Option<Vec<DataValue>>;

    /// Peek at the next row without consuming it.
    fn peek(&self) -> Option<&[DataValue]> { None }

    /// Reset to the beginning (if supported).
    fn reset(&mut self) -> Result<(), SoniTypeError> {
        Err(SoniTypeError::Io("reset not supported".into()))
    }

    /// Total number of rows if known.
    fn row_count_hint(&self) -> Option<usize> { None }
}

// ===========================================================================
// Sonification mapping
// ===========================================================================

/// Maps data values to audio parameters.
pub trait SonificationMapping {
    /// Map a data value to a frequency.
    fn map_to_frequency(&self, value: &DataValue) -> Option<Frequency>;

    /// Map a data value to an amplitude.
    fn map_to_amplitude(&self, value: &DataValue) -> Option<Amplitude>;

    /// Map a data value to a pan position.
    fn map_to_pan(&self, value: &DataValue) -> Option<Pan>;

    /// Map a data value to a complete stream descriptor.
    fn map_to_stream(&self, values: &[DataValue]) -> Result<StreamDescriptor, SoniTypeError>;

    /// The mapping parameters this implementation uses.
    fn parameters(&self) -> &[MappingParameter];
}

// ===========================================================================
// Type checker
// ===========================================================================

/// Checks perceptual type constraints on mapping configurations.
pub trait TypeChecker {
    /// Check all constraints for a set of streams. Returns a list of violations.
    fn check(&self, streams: &[StreamDescriptor]) -> Vec<String>;

    /// Check whether two specific streams can coexist without perceptual confusion.
    fn check_pair(&self, a: &StreamDescriptor, b: &StreamDescriptor) -> Vec<String>;

    /// Returns true if the configuration passes all checks.
    fn is_valid(&self, streams: &[StreamDescriptor]) -> bool {
        self.check(streams).is_empty()
    }
}

// ===========================================================================
// Code generator
// ===========================================================================

/// Generates renderer code from an intermediate representation.
pub trait CodeGenerator {
    /// The IR type this generator consumes.
    type IR;
    /// The output code representation.
    type Output;

    /// Generate code from the IR.
    fn generate(&self, ir: &Self::IR) -> Result<Self::Output, SoniTypeError>;

    /// Name of the target platform / backend.
    fn target_name(&self) -> &str;
}

// ===========================================================================
// Stream processor
// ===========================================================================

/// Processes an audio stream with effects (filter, reverb, etc.).
pub trait StreamProcessor: Send {
    /// Process a buffer in-place.
    fn process(&mut self, buffer: &mut AudioBuffer);

    /// Latency introduced by this processor in samples.
    fn latency_samples(&self) -> usize { 0 }

    /// Reset processor state.
    fn reset(&mut self);

    /// Human-readable name.
    fn name(&self) -> &str;

    /// Tail length in samples (e.g., reverb tail).
    fn tail_samples(&self) -> usize { 0 }
}

// ===========================================================================
// Serializable
// ===========================================================================

/// Trait for binary serialization/deserialization.
pub trait Serializable: Sized {
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Result<Self, SoniTypeError>;
}

// ===========================================================================
// Validate
// ===========================================================================

/// Trait for entities that can be validated.
pub trait Validate {
    type Error;
    fn validate(&self) -> Result<(), Self::Error>;
}

// ===========================================================================
// Named
// ===========================================================================

/// Trait for entities that carry a human-readable label.
pub trait Named {
    fn name(&self) -> &str;
}

// ===========================================================================
// Default implementations
// ===========================================================================

/// A simple psychoacoustic model using standard approximations.
pub struct SimplePsychoacousticModel;

impl PsychoacousticModel for SimplePsychoacousticModel {
    fn masking_threshold(&self, masker_freq: Frequency, masker_level_db: f64, probe_freq: Frequency) -> f64 {
        let bark_diff = crate::units::hz_to_bark(probe_freq.hz()) - crate::units::hz_to_bark(masker_freq.hz());
        let spread = if bark_diff >= 0.0 {
            // Upward spread of masking (steeper)
            -27.0 * bark_diff
        } else {
            // Downward spread (more gradual)
            (-6.0 - 0.4 * masker_level_db) * bark_diff
        };
        masker_level_db + spread
    }

    fn jnd_pitch(&self, freq: Frequency, _level_db: f64) -> f64 {
        // Weber fraction ~0.3% for frequencies above 500 Hz, larger below
        let f = freq.hz();
        if f < 500.0 {
            1.0 + 0.003 * (500.0 - f)
        } else {
            f * 0.003
        }
    }

    fn jnd_loudness(&self, level_db: f64) -> f64 {
        // ~1 dB near threshold, ~0.3 dB at higher levels
        if level_db < 20.0 { 3.0 } else if level_db < 40.0 { 1.5 } else { 0.5 }
    }
}

/// A default segregation predicate using standard thresholds.
pub struct DefaultSegregationPredicate;

impl SegregationPredicate for DefaultSegregationPredicate {}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stream(freq: f64, amp: f64) -> StreamDescriptor {
        StreamDescriptor::new(Frequency::new(freq), Amplitude::new(amp))
    }

    #[test]
    fn simple_model_masking_threshold() {
        let model = SimplePsychoacousticModel;
        let thresh = model.masking_threshold(
            Frequency::new(1000.0), 60.0, Frequency::new(1200.0),
        );
        assert!(thresh < 60.0, "Probe should be below masker level");
    }

    #[test]
    fn simple_model_jnd_pitch() {
        let model = SimplePsychoacousticModel;
        let jnd_low = model.jnd_pitch(Frequency::new(200.0), 60.0);
        let jnd_high = model.jnd_pitch(Frequency::new(2000.0), 60.0);
        assert!(jnd_high > jnd_low, "JND should scale with frequency");
    }

    #[test]
    fn simple_model_jnd_loudness() {
        let model = SimplePsychoacousticModel;
        let jnd_quiet = model.jnd_loudness(10.0);
        let jnd_loud = model.jnd_loudness(60.0);
        assert!(jnd_quiet > jnd_loud, "JND should decrease with level");
    }

    #[test]
    fn simple_model_bark_band() {
        let model = SimplePsychoacousticModel;
        let band = model.bark_band_for_frequency(Frequency::new(1000.0));
        assert!(band.value() > 5 && band.value() < 12);
    }

    #[test]
    fn segregation_onset_synchrony() {
        let pred = DefaultSegregationPredicate;
        assert!(pred.check_onset_synchrony(Duration::new(0.05)));
        assert!(!pred.check_onset_synchrony(Duration::new(0.01)));
    }

    #[test]
    fn segregation_spectral_proximity() {
        let pred = DefaultSegregationPredicate;
        assert!(pred.check_spectral_proximity(Frequency::new(200.0), Frequency::new(2000.0)));
        assert!(!pred.check_spectral_proximity(Frequency::new(1000.0), Frequency::new(1050.0)));
    }

    #[test]
    fn segregation_common_fate() {
        let pred = DefaultSegregationPredicate;
        assert!(pred.check_common_fate(0.1));
        assert!(!pred.check_common_fate(0.8));
    }

    #[test]
    fn segregation_evaluate_all() {
        let pred = DefaultSegregationPredicate;
        let a = make_stream(200.0, 0.5);
        let b = make_stream(4000.0, 0.5);
        let score = pred.evaluate_all(&a, &b, Duration::new(0.05), 0.1);
        assert!(score > 0.5, "Widely separated streams should segregate well: {score}");
    }

    #[test]
    fn optimization_result_structure() {
        let res = OptimizationResult {
            parameters: vec![MappingParameter::PitchRange(200.0, 2000.0)],
            cost: 0.5,
            iterations: 100,
            converged: true,
        };
        assert!(res.converged);
        assert_eq!(res.parameters.len(), 1);
    }

    #[test]
    fn audio_node_type_equality() {
        assert_eq!(AudioNodeType::Source, AudioNodeType::Source);
        assert_ne!(AudioNodeType::Source, AudioNodeType::Effect);
        assert_eq!(
            AudioNodeType::Custom("reverb".into()),
            AudioNodeType::Custom("reverb".into()),
        );
    }

    #[test]
    fn segregation_harmonicity() {
        let pred = DefaultSegregationPredicate;
        // Octave relationship (harmonic) -> should NOT be segregated
        let a = make_stream(440.0, 0.5);
        let b = make_stream(880.0, 0.5);
        let harmonic = pred.check_harmonicity(&a, &b);
        assert!(!harmonic, "Octave (2:1) should be detected as harmonic");

        // Non-harmonic relationship -> should be segregated
        let c = make_stream(440.0, 0.5);
        let d = make_stream(630.0, 0.5);
        let non_harmonic = pred.check_harmonicity(&c, &d);
        assert!(non_harmonic, "Non-integer ratio should be detected as non-harmonic");
    }

    #[test]
    fn specific_loudness_default() {
        let model = SimplePsychoacousticModel;
        let loud = model.specific_loudness(Frequency::new(1000.0), 60.0);
        let quiet = model.specific_loudness(Frequency::new(1000.0), 20.0);
        assert!(loud > quiet);
    }
}
