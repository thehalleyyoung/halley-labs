//! Standard library validation for SoniType.
//!
//! Validates scales, timbres, mappings, and presets against psychoacoustic
//! limits to ensure perceptual quality and discriminability.

use crate::scales::{LinearScale, MusicalScale, MidiScale, MicrotonalScale};
use crate::timbres::{TimbrePalette, TimbreDescriptor, AdditiveTimbre};
use crate::mappings::{
    PitchMapping, LoudnessMapping, CompositeMapping, AudioParameter,
};

// ---------------------------------------------------------------------------
// Validation result types
// ---------------------------------------------------------------------------

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationFinding {
    pub severity: Severity,
    pub category: String,
    pub message: String,
}

impl ValidationFinding {
    pub fn info(category: impl Into<String>, message: impl Into<String>) -> Self {
        Self { severity: Severity::Info, category: category.into(), message: message.into() }
    }

    pub fn warning(category: impl Into<String>, message: impl Into<String>) -> Self {
        Self { severity: Severity::Warning, category: category.into(), message: message.into() }
    }

    pub fn error(category: impl Into<String>, message: impl Into<String>) -> Self {
        Self { severity: Severity::Error, category: category.into(), message: message.into() }
    }

    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }

    pub fn is_warning(&self) -> bool {
        self.severity == Severity::Warning
    }
}

impl std::fmt::Display for ValidationFinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sev = match self.severity {
            Severity::Info => "INFO",
            Severity::Warning => "WARN",
            Severity::Error => "ERROR",
        };
        write!(f, "[{}] {}: {}", sev, self.category, self.message)
    }
}

/// Aggregated validation report.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<ValidationFinding>,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self { findings: Vec::new() }
    }

    pub fn add(&mut self, finding: ValidationFinding) {
        self.findings.push(finding);
    }

    pub fn merge(&mut self, other: ValidationReport) {
        self.findings.extend(other.findings);
    }

    pub fn has_errors(&self) -> bool {
        self.findings.iter().any(|f| f.is_error())
    }

    pub fn has_warnings(&self) -> bool {
        self.findings.iter().any(|f| f.is_warning())
    }

    pub fn errors(&self) -> Vec<&ValidationFinding> {
        self.findings.iter().filter(|f| f.is_error()).collect()
    }

    pub fn warnings(&self) -> Vec<&ValidationFinding> {
        self.findings.iter().filter(|f| f.is_warning()).collect()
    }

    pub fn is_valid(&self) -> bool {
        !self.has_errors()
    }

    pub fn summary(&self) -> String {
        let errors = self.errors().len();
        let warnings = self.warnings().len();
        let infos = self.findings.len() - errors - warnings;
        format!("{} error(s), {} warning(s), {} info(s)", errors, warnings, infos)
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Psychoacoustic constants
// ---------------------------------------------------------------------------

/// Minimum audible frequency in Hz.
const MIN_AUDIBLE_HZ: f64 = 20.0;
/// Maximum audible frequency in Hz.
const MAX_AUDIBLE_HZ: f64 = 20000.0;
/// Recommended minimum for sonification (avoids sub-bass rumble).
const MIN_SONI_HZ: f64 = 100.0;
/// Recommended maximum for sonification (avoids painful high frequencies).
const MAX_SONI_HZ: f64 = 8000.0;
/// Pitch JND in cents (just-noticeable difference).
const PITCH_JND_CENTS: f64 = 5.0;
/// Minimum recommended loudness range in dB.
const MIN_LOUDNESS_RANGE_DB: f64 = 10.0;
/// Maximum recommended simultaneous streams for cognitive load.
const MAX_COGNITIVE_STREAMS: usize = 6;
/// Minimum timbre discriminability distance in perceptual space.
const MIN_TIMBRE_DISTANCE: f64 = 0.15;

// ---------------------------------------------------------------------------
// Scale validation
// ---------------------------------------------------------------------------

/// Validate that a frequency range is within psychoacoustic limits.
pub fn validate_frequency_range(f_min: f64, f_max: f64) -> ValidationReport {
    let mut report = ValidationReport::new();
    if f_min < MIN_AUDIBLE_HZ {
        report.add(ValidationFinding::error(
            "scale",
            format!("f_min ({:.1} Hz) is below audible range ({} Hz)", f_min, MIN_AUDIBLE_HZ),
        ));
    }
    if f_max > MAX_AUDIBLE_HZ {
        report.add(ValidationFinding::error(
            "scale",
            format!("f_max ({:.1} Hz) exceeds audible range ({} Hz)", f_max, MAX_AUDIBLE_HZ),
        ));
    }
    if f_min < MIN_SONI_HZ {
        report.add(ValidationFinding::warning(
            "scale",
            format!("f_min ({:.1} Hz) is below recommended sonification range ({} Hz)", f_min, MIN_SONI_HZ),
        ));
    }
    if f_max > MAX_SONI_HZ {
        report.add(ValidationFinding::warning(
            "scale",
            format!("f_max ({:.1} Hz) exceeds recommended sonification range ({} Hz)", f_max, MAX_SONI_HZ),
        ));
    }
    if f_min >= f_max {
        report.add(ValidationFinding::error(
            "scale",
            "f_min must be less than f_max".to_string(),
        ));
    }
    // Check pitch resolution (enough JND steps)
    if f_min > 0.0 && f_max > f_min {
        let cents_range = 1200.0 * (f_max / f_min).log2();
        let jnd_steps = cents_range / PITCH_JND_CENTS;
        if jnd_steps < 20.0 {
            report.add(ValidationFinding::warning(
                "scale",
                format!("Frequency range spans only {:.0} JND steps (recommend ≥ 20)", jnd_steps),
            ));
        } else {
            report.add(ValidationFinding::info(
                "scale",
                format!("Frequency range spans {:.0} JND steps", jnd_steps),
            ));
        }
    }
    report
}

/// Validate a musical scale's pitches.
pub fn validate_musical_scale(scale: &MusicalScale) -> ValidationReport {
    let mut report = validate_frequency_range(
        scale.pitches().first().copied().unwrap_or(0.0),
        scale.pitches().last().copied().unwrap_or(0.0),
    );
    if scale.pitches().len() < 3 {
        report.add(ValidationFinding::warning(
            "scale",
            "Musical scale has fewer than 3 pitches; limited expressiveness".to_string(),
        ));
    }
    // Check for sufficient pitch separation
    let pitches = scale.pitches();
    for window in pitches.windows(2) {
        let cents = 1200.0 * (window[1] / window[0]).log2();
        if cents < PITCH_JND_CENTS * 2.0 {
            report.add(ValidationFinding::warning(
                "scale",
                format!(
                    "Adjacent pitches {:.1} Hz and {:.1} Hz are only {:.1} cents apart (JND = {} cents)",
                    window[0], window[1], cents, PITCH_JND_CENTS,
                ),
            ));
            break;
        }
    }
    report
}

/// Validate a microtonal scale.
pub fn validate_microtonal_scale(scale: &MicrotonalScale) -> ValidationReport {
    let pitches = scale.pitches();
    let f_min = pitches.first().copied().unwrap_or(0.0);
    let f_max = pitches.last().copied().unwrap_or(0.0);
    let mut report = validate_frequency_range(f_min, f_max);
    if pitches.is_empty() {
        report.add(ValidationFinding::error("scale", "Microtonal scale has no pitches".to_string()));
    }
    report
}

// ---------------------------------------------------------------------------
// Timbre validation
// ---------------------------------------------------------------------------

/// Validate a timbre palette for perceptual discriminability.
pub fn validate_timbre_palette(palette: &TimbrePalette) -> ValidationReport {
    let mut report = ValidationReport::new();
    if palette.is_empty() {
        report.add(ValidationFinding::error("timbre", "Palette is empty".to_string()));
        return report;
    }
    let min_dist = palette.min_distance();
    if min_dist < MIN_TIMBRE_DISTANCE {
        report.add(ValidationFinding::warning(
            "timbre",
            format!(
                "Minimum pairwise timbre distance {:.3} is below threshold {:.3}; timbres may be confusable",
                min_dist, MIN_TIMBRE_DISTANCE,
            ),
        ));
    } else {
        report.add(ValidationFinding::info(
            "timbre",
            format!("Minimum pairwise timbre distance: {:.3} (threshold: {:.3})", min_dist, MIN_TIMBRE_DISTANCE),
        ));
    }
    // Check for extreme brightness values
    for d in &palette.descriptors {
        if d.brightness > 0.95 {
            report.add(ValidationFinding::warning(
                "timbre",
                format!("Timbre '{}' has very high brightness ({:.2}); may cause listener fatigue", d.name, d.brightness),
            ));
        }
    }
    report
}

/// Validate an additive timbre for reasonable partial count and brightness.
pub fn validate_additive_timbre(timbre: &AdditiveTimbre) -> ValidationReport {
    let mut report = ValidationReport::new();
    if timbre.partials.is_empty() {
        report.add(ValidationFinding::error("timbre", "Additive timbre has no partials".to_string()));
        return report;
    }
    let max_harmonic = timbre.partials.iter().map(|p| p.harmonic).max().unwrap_or(0);
    if max_harmonic > 32 {
        report.add(ValidationFinding::warning(
            "timbre",
            format!("Timbre '{}' uses harmonics up to {}; high harmonics may alias at low sample rates", timbre.name, max_harmonic),
        ));
    }
    let brightness = timbre.brightness();
    if brightness > 10.0 {
        report.add(ValidationFinding::warning(
            "timbre",
            format!("Timbre '{}' has very high spectral centroid proxy ({:.1})", timbre.name, brightness),
        ));
    }
    report
}

// ---------------------------------------------------------------------------
// Mapping validation
// ---------------------------------------------------------------------------

/// Validate a pitch mapping for JND compliance.
pub fn validate_pitch_mapping(mapping: &PitchMapping) -> ValidationReport {
    validate_frequency_range(mapping.f_min, mapping.f_max)
}

/// Validate a loudness mapping for adequate dynamic range.
pub fn validate_loudness_mapping(mapping: &LoudnessMapping) -> ValidationReport {
    let mut report = ValidationReport::new();
    let range = (mapping.range_max - mapping.range_min).abs();
    if range < MIN_LOUDNESS_RANGE_DB {
        report.add(ValidationFinding::warning(
            "loudness",
            format!("Loudness range {:.1} dB is narrow (recommend ≥ {} dB)", range, MIN_LOUDNESS_RANGE_DB),
        ));
    }
    if mapping.range_max > 0.0 {
        report.add(ValidationFinding::warning(
            "loudness",
            "Maximum loudness > 0 dBFS; clipping may occur".to_string(),
        ));
    }
    report
}

/// Validate a composite mapping for cognitive load limits.
pub fn validate_composite_mapping(mapping: &CompositeMapping) -> ValidationReport {
    let mut report = ValidationReport::new();
    let n_streams = mapping.layers.len();
    if n_streams > MAX_COGNITIVE_STREAMS {
        report.add(ValidationFinding::warning(
            "cognitive_load",
            format!(
                "Composite mapping has {} layers; exceeds recommended maximum of {} simultaneous streams",
                n_streams, MAX_COGNITIVE_STREAMS,
            ),
        ));
    }
    // Check for duplicate targets.
    let mut targets: Vec<AudioParameter> = mapping.layers.iter().map(|l| l.target).collect();
    targets.sort_by_key(|t| format!("{:?}", t));
    let before = targets.len();
    targets.dedup();
    if targets.len() < before {
        report.add(ValidationFinding::warning(
            "mapping",
            "Composite mapping has duplicate target parameters; later mappings may override earlier ones".to_string(),
        ));
    }
    // Validate sub-mappings if present.
    if let Some(ref pm) = mapping.pitch {
        report.merge(validate_pitch_mapping(pm));
    }
    if let Some(ref lm) = mapping.loudness {
        report.merge(validate_loudness_mapping(lm));
    }
    report
}

// ---------------------------------------------------------------------------
// StdlibValidator (run all checks)
// ---------------------------------------------------------------------------

/// Run all standard-library validation checks.
#[derive(Debug, Clone)]
pub struct StdlibValidator;

impl StdlibValidator {
    /// Validate a frequency range.
    pub fn validate_frequency_range(f_min: f64, f_max: f64) -> ValidationReport {
        validate_frequency_range(f_min, f_max)
    }

    /// Validate a musical scale.
    pub fn validate_musical_scale(scale: &MusicalScale) -> ValidationReport {
        validate_musical_scale(scale)
    }

    /// Validate a microtonal scale.
    pub fn validate_microtonal_scale(scale: &MicrotonalScale) -> ValidationReport {
        validate_microtonal_scale(scale)
    }

    /// Validate a timbre palette.
    pub fn validate_timbre_palette(palette: &TimbrePalette) -> ValidationReport {
        validate_timbre_palette(palette)
    }

    /// Validate an additive timbre.
    pub fn validate_additive_timbre(timbre: &AdditiveTimbre) -> ValidationReport {
        validate_additive_timbre(timbre)
    }

    /// Validate a pitch mapping.
    pub fn validate_pitch_mapping(mapping: &PitchMapping) -> ValidationReport {
        validate_pitch_mapping(mapping)
    }

    /// Validate a loudness mapping.
    pub fn validate_loudness_mapping(mapping: &LoudnessMapping) -> ValidationReport {
        validate_loudness_mapping(mapping)
    }

    /// Validate a composite mapping.
    pub fn validate_composite_mapping(mapping: &CompositeMapping) -> ValidationReport {
        validate_composite_mapping(mapping)
    }

    /// Run a comprehensive validation suite across all components.
    pub fn validate_all(
        freq_range: Option<(f64, f64)>,
        musical_scale: Option<&MusicalScale>,
        microtonal_scale: Option<&MicrotonalScale>,
        palette: Option<&TimbrePalette>,
        additive_timbre: Option<&AdditiveTimbre>,
        pitch_mapping: Option<&PitchMapping>,
        loudness_mapping: Option<&LoudnessMapping>,
        composite_mapping: Option<&CompositeMapping>,
    ) -> ValidationReport {
        let mut report = ValidationReport::new();
        if let Some((f_min, f_max)) = freq_range {
            report.merge(validate_frequency_range(f_min, f_max));
        }
        if let Some(s) = musical_scale {
            report.merge(validate_musical_scale(s));
        }
        if let Some(s) = microtonal_scale {
            report.merge(validate_microtonal_scale(s));
        }
        if let Some(p) = palette {
            report.merge(validate_timbre_palette(p));
        }
        if let Some(t) = additive_timbre {
            report.merge(validate_additive_timbre(t));
        }
        if let Some(pm) = pitch_mapping {
            report.merge(validate_pitch_mapping(pm));
        }
        if let Some(lm) = loudness_mapping {
            report.merge(validate_loudness_mapping(lm));
        }
        if let Some(cm) = composite_mapping {
            report.merge(validate_composite_mapping(cm));
        }
        report
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scales::{NoteName, ScaleType};

    #[test]
    fn test_valid_frequency_range() {
        let report = validate_frequency_range(200.0, 4000.0);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_below_audible_range() {
        let report = validate_frequency_range(10.0, 4000.0);
        assert!(report.has_errors());
    }

    #[test]
    fn test_above_audible_range() {
        let report = validate_frequency_range(200.0, 25000.0);
        assert!(report.has_errors());
    }

    #[test]
    fn test_below_recommended_range() {
        let report = validate_frequency_range(50.0, 4000.0);
        assert!(report.has_warnings());
    }

    #[test]
    fn test_inverted_range() {
        let report = validate_frequency_range(4000.0, 200.0);
        assert!(report.has_errors());
    }

    #[test]
    fn test_narrow_range_jnd() {
        // Very narrow range: few JND steps
        let report = validate_frequency_range(440.0, 460.0);
        assert!(report.has_warnings());
    }

    #[test]
    fn test_musical_scale_valid() {
        let scale = MusicalScale::new(NoteName::C, ScaleType::Major, 3, 6);
        let report = validate_musical_scale(&scale);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_musical_scale_too_few_pitches() {
        let scale = MusicalScale::new(NoteName::C, ScaleType::Custom(vec![0, 1]), 4, 4);
        let report = validate_musical_scale(&scale);
        // Only 2 pitches → warning
        assert!(report.has_warnings());
    }

    #[test]
    fn test_palette_8_valid() {
        let palette = TimbrePalette::palette_8();
        let report = validate_timbre_palette(&palette);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_palette_empty() {
        let palette = TimbrePalette::new("empty", vec![]);
        let report = validate_timbre_palette(&palette);
        assert!(report.has_errors());
    }

    #[test]
    fn test_palette_low_discriminability() {
        let palette = TimbrePalette::new("similar", vec![
            TimbreDescriptor::new("a", 0.5, 0.5, 0.5),
            TimbreDescriptor::new("b", 0.51, 0.51, 0.51),
        ]);
        let report = validate_timbre_palette(&palette);
        assert!(report.has_warnings());
    }

    #[test]
    fn test_additive_timbre_valid() {
        let t = AdditiveTimbre::flute();
        let report = validate_additive_timbre(&t);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_additive_timbre_empty() {
        let t = AdditiveTimbre::new("empty", vec![]);
        let report = validate_additive_timbre(&t);
        assert!(report.has_errors());
    }

    #[test]
    fn test_pitch_mapping_validation() {
        let pm = PitchMapping::continuous(200.0, 4000.0, (0.0, 1.0));
        let report = validate_pitch_mapping(&pm);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_loudness_mapping_narrow_range() {
        let lm = LoudnessMapping::linear_db(-5.0, 0.0, (0.0, 1.0));
        let report = validate_loudness_mapping(&lm);
        assert!(report.has_warnings());
    }

    #[test]
    fn test_loudness_mapping_clipping() {
        let lm = LoudnessMapping::linear_db(-20.0, 6.0, (0.0, 1.0));
        let report = validate_loudness_mapping(&lm);
        assert!(report.has_warnings()); // max > 0 dBFS
    }

    #[test]
    fn test_composite_cognitive_overload() {
        let mut cm = CompositeMapping::new();
        for i in 0..8 {
            cm = cm.add_pitch(
                &format!("field_{}", i),
                PitchMapping::continuous(200.0, 4000.0, (0.0, 1.0)),
                10,
            );
        }
        let report = validate_composite_mapping(&cm);
        assert!(report.has_warnings());
    }

    #[test]
    fn test_validation_report_summary() {
        let mut report = ValidationReport::new();
        report.add(ValidationFinding::info("test", "An info"));
        report.add(ValidationFinding::warning("test", "A warning"));
        report.add(ValidationFinding::error("test", "An error"));
        let summary = report.summary();
        assert!(summary.contains("1 error(s)"));
        assert!(summary.contains("1 warning(s)"));
        assert!(summary.contains("1 info(s)"));
    }

    #[test]
    fn test_stdlib_validator_all() {
        let scale = MusicalScale::new(NoteName::C, ScaleType::Major, 3, 6);
        let palette = TimbrePalette::palette_8();
        let pm = PitchMapping::continuous(200.0, 4000.0, (0.0, 1.0));
        let report = StdlibValidator::validate_all(
            Some((200.0, 4000.0)),
            Some(&scale),
            None,
            Some(&palette),
            None,
            Some(&pm),
            None,
            None,
        );
        assert!(report.is_valid());
    }

    #[test]
    fn test_finding_display() {
        let f = ValidationFinding::error("scale", "Bad range");
        let s = format!("{}", f);
        assert!(s.contains("[ERROR]"));
        assert!(s.contains("scale"));
        assert!(s.contains("Bad range"));
    }
}
