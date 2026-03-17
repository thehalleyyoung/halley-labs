//! Just Noticeable Difference (JND) Models
//!
//! Psychoacoustic models for the smallest perceptible change in pitch,
//! loudness, timing, timbre, and spatial position.  Based on Weber-fraction
//! models, the Wier et al. (1977) frequency DL, and the minimum audible
//! angle (MAA) literature.

use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Local type aliases
// ---------------------------------------------------------------------------

pub type Frequency = f64;
pub type DecibelSpl = f64;

// ---------------------------------------------------------------------------
// PitchJnd – frequency discrimination
// ---------------------------------------------------------------------------

/// Weber-fraction model for frequency discrimination.
///
/// The frequency difference limen (DL) is approximately 0.35 % in the
/// 500 Hz – 5 kHz range, rising below and above that range (Wier et al. 1977).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchJnd {
    /// Weber fraction in the optimal region (500 Hz – 5 kHz).
    pub optimal_fraction: f64,
}

impl PitchJnd {
    pub fn new() -> Self {
        Self {
            optimal_fraction: 0.0035,
        }
    }

    /// Frequency-dependent Weber fraction Δf / f.
    ///
    /// * 500 – 5000 Hz: ≈ 0.0035
    /// * Below 500 Hz: rises as √(500/f) · 0.0035
    /// * Above 5 kHz: rises as √(f/5000) · 0.0035
    pub fn weber_fraction(&self, f: f64) -> f64 {
        let f = f.abs().max(20.0);
        if f < 500.0 {
            self.optimal_fraction * (500.0 / f).sqrt()
        } else if f > 5000.0 {
            self.optimal_fraction * (f / 5000.0).sqrt()
        } else {
            self.optimal_fraction
        }
    }

    /// JND in Hz at frequency `f`.
    pub fn jnd_frequency(&self, f: f64) -> f64 {
        f.abs().max(20.0) * self.weber_fraction(f)
    }

    /// Are two frequencies discriminable?
    pub fn is_discriminable(&self, f1: f64, f2: f64) -> bool {
        let lower = f1.abs().min(f2.abs()).max(20.0);
        (f1 - f2).abs() > self.jnd_frequency(lower)
    }

    /// Discriminability ratio: |f1 − f2| / JND(min(f1, f2)).
    pub fn discriminability_ratio(&self, f1: f64, f2: f64) -> f64 {
        let lower = f1.abs().min(f2.abs()).max(20.0);
        let jnd = self.jnd_frequency(lower);
        if jnd <= 0.0 {
            return f64::INFINITY;
        }
        (f1 - f2).abs() / jnd
    }
}

impl Default for PitchJnd {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LoudnessJnd – intensity discrimination
// ---------------------------------------------------------------------------

/// Intensity JND based on the near-miss to Weber's law.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoudnessJnd {
    /// JND in dB at moderate levels (40 – 80 dB SPL).
    pub moderate_jnd_db: f64,
}

impl LoudnessJnd {
    pub fn new() -> Self {
        Self {
            moderate_jnd_db: 1.0,
        }
    }

    /// JND in dB at a given SPL level.
    ///
    /// * Moderate levels (20 – 80 dB): ≈ 1 dB
    /// * Low levels (< 20 dB): rises (near-miss to Weber's law)
    /// * High levels (> 80 dB): slightly smaller
    pub fn jnd_intensity(&self, level_db: f64) -> f64 {
        if level_db < 20.0 {
            self.moderate_jnd_db + 0.5 * (20.0 - level_db) / 20.0
        } else if level_db > 80.0 {
            (self.moderate_jnd_db - 0.1 * (level_db - 80.0) / 40.0).max(0.5)
        } else {
            self.moderate_jnd_db
        }
    }

    /// Weber fraction for intensity: ΔI / I in linear terms.
    pub fn weber_fraction_intensity(&self, level_db: f64) -> f64 {
        let jnd_db = self.jnd_intensity(level_db);
        10.0_f64.powf(jnd_db / 10.0) - 1.0
    }

    /// Are two levels discriminable?
    pub fn is_discriminable(&self, l1_db: f64, l2_db: f64) -> bool {
        let lower = l1_db.min(l2_db);
        (l1_db - l2_db).abs() > self.jnd_intensity(lower)
    }

    /// Discriminability ratio: |l1 − l2| / JND(min).
    pub fn discriminability_ratio(&self, l1_db: f64, l2_db: f64) -> f64 {
        let lower = l1_db.min(l2_db);
        let jnd = self.jnd_intensity(lower);
        if jnd <= 0.0 {
            return f64::INFINITY;
        }
        (l1_db - l2_db).abs() / jnd
    }
}

impl Default for LoudnessJnd {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TemporalJnd – temporal resolution
// ---------------------------------------------------------------------------

/// Temporal JNDs for duration, onset asynchrony, and gap detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalJnd {
    /// Minimum gap-detection threshold (ms) for broadband stimuli.
    pub min_gap_ms: f64,
    /// Baseline onset asynchrony detection (ms).
    pub baseline_onset_ms: f64,
}

impl TemporalJnd {
    pub fn new() -> Self {
        Self {
            min_gap_ms: 2.5,
            baseline_onset_ms: 20.0,
        }
    }

    /// JND for duration changes (ms).
    ///
    /// Weber fraction ≈ 10 % for durations above ~100 ms, with a floor of ~10 ms.
    pub fn jnd_duration(&self, duration_ms: f64) -> f64 {
        let weber = 0.10 * duration_ms;
        weber.max(10.0)
    }

    /// JND for onset asynchrony (ms).
    ///
    /// Baseline ≈ 20 ms; scales slightly with reference duration.
    pub fn jnd_onset(&self, reference_ms: f64) -> f64 {
        let base = self.baseline_onset_ms;
        base + 0.02 * (reference_ms - 200.0).max(0.0)
    }

    /// Gap-detection threshold (ms).
    ///
    /// ~2–3 ms for broadband noise, ~8–10 ms for narrow-band or tonal stimuli.
    pub fn jnd_gap(&self, _gap_ms: f64) -> f64 {
        self.min_gap_ms
    }

    /// JND for gap detection of tonal stimuli (longer).
    pub fn jnd_gap_tonal(&self) -> f64 {
        10.0
    }

    /// Are two onset times discriminable?
    pub fn is_onset_discriminable(&self, onset1_ms: f64, onset2_ms: f64) -> bool {
        let ref_ms = onset1_ms.min(onset2_ms);
        (onset1_ms - onset2_ms).abs() > self.jnd_onset(ref_ms)
    }

    /// Discriminability ratio for onsets.
    pub fn onset_discriminability_ratio(&self, onset1_ms: f64, onset2_ms: f64) -> f64 {
        let ref_ms = onset1_ms.min(onset2_ms);
        let jnd = self.jnd_onset(ref_ms);
        if jnd <= 0.0 {
            return f64::INFINITY;
        }
        (onset1_ms - onset2_ms).abs() / jnd
    }
}

impl Default for TemporalJnd {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TimbreJnd – spectral shape discrimination
// ---------------------------------------------------------------------------

/// JND models for timbral attributes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimbreJnd {
    pub centroid_fraction: f64,
    pub spread_fraction: f64,
    pub flux_fraction: f64,
    pub attack_fraction: f64,
}

impl TimbreJnd {
    pub fn new() -> Self {
        Self {
            centroid_fraction: 0.05,
            spread_fraction: 0.10,
            flux_fraction: 0.15,
            attack_fraction: 0.25,
        }
    }

    /// JND for spectral centroid (Hz).  ≈ 5 % of centroid value.
    pub fn jnd_spectral_centroid(&self, centroid_hz: f64) -> f64 {
        self.centroid_fraction * centroid_hz.abs().max(1.0)
    }

    /// JND for spectral spread. ≈ 10 %.
    pub fn jnd_spectral_spread(&self, spread: f64) -> f64 {
        self.spread_fraction * spread.abs().max(1.0)
    }

    /// JND for spectral flux. ≈ 15 %.
    pub fn jnd_spectral_flux(&self, flux: f64) -> f64 {
        self.flux_fraction * flux.abs().max(0.001)
    }

    /// JND for attack time. ≈ 20–30 % (uses 25 %).
    pub fn jnd_attack_time(&self, attack_ms: f64) -> f64 {
        self.attack_fraction * attack_ms.abs().max(1.0)
    }

    /// Are two spectral centroids discriminable?
    pub fn is_centroid_discriminable(&self, c1: f64, c2: f64) -> bool {
        let lower = c1.abs().min(c2.abs()).max(1.0);
        (c1 - c2).abs() > self.jnd_spectral_centroid(lower)
    }
}

impl Default for TimbreJnd {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SpatialJnd – spatial discrimination
// ---------------------------------------------------------------------------

/// Minimum Audible Angle (MAA) and spatial JND models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialJnd {
    /// MAA at 0° azimuth (straight ahead), in degrees.
    pub maa_frontal: f64,
}

impl SpatialJnd {
    pub fn new() -> Self {
        Self { maa_frontal: 1.0 }
    }

    /// Minimum audible angle (degrees) as a function of azimuth.
    ///
    /// MAA ≈ 1° at 0° (straight ahead), increasing toward the sides:
    ///   MAA(θ) = 1.0 + 0.004 · θ², capped at 40°.
    pub fn jnd_azimuth(&self, azimuth_deg: f64) -> f64 {
        let theta = azimuth_deg.abs();
        (self.maa_frontal + 0.004 * theta * theta).min(40.0)
    }

    /// Elevation JND (degrees).  ≈ 4° at 0° elevation, increasing off-axis.
    pub fn jnd_elevation(&self, elevation_deg: f64) -> f64 {
        let e = elevation_deg.abs();
        (4.0 + 0.05 * e).min(30.0)
    }

    /// Distance JND (Weber fraction ≈ 5–20 % depending on conditions).
    pub fn jnd_distance(&self, distance_m: f64) -> f64 {
        let frac = if distance_m < 1.0 {
            0.20
        } else {
            0.05 + 0.01 * distance_m.min(15.0)
        };
        frac * distance_m.abs().max(0.01)
    }

    /// Are two azimuths discriminable?
    pub fn is_azimuth_discriminable(&self, az1: f64, az2: f64) -> bool {
        let ref_az = az1.abs().min(az2.abs());
        (az1 - az2).abs() > self.jnd_azimuth(ref_az)
    }

    /// Azimuth discriminability ratio.
    pub fn azimuth_discriminability_ratio(&self, az1: f64, az2: f64) -> f64 {
        let ref_az = az1.abs().min(az2.abs());
        let jnd = self.jnd_azimuth(ref_az);
        if jnd <= 0.0 {
            return f64::INFINITY;
        }
        (az1 - az2).abs() / jnd
    }
}

impl Default for SpatialJnd {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// JndDimension enum
// ---------------------------------------------------------------------------

/// Perceptual dimension for JND evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JndDimension {
    Pitch,
    Loudness,
    Duration,
    Onset,
    Gap,
    SpectralCentroid,
    SpectralSpread,
    SpectralFlux,
    AttackTime,
    Azimuth,
    Elevation,
    Distance,
}

impl std::fmt::Display for JndDimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// PerceptualParams & ValidationReport
// ---------------------------------------------------------------------------

/// A snapshot of perceptual parameters for one auditory stream / event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualParams {
    pub frequency: f64,
    pub level_db: f64,
    pub onset_ms: f64,
    pub duration_ms: f64,
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub spectral_centroid: f64,
    pub spectral_spread: f64,
    pub spectral_flux: f64,
    pub attack_time_ms: f64,
}

impl PerceptualParams {
    /// Convenience constructor with the most common parameters.
    pub fn basic(frequency: f64, level_db: f64, onset_ms: f64, azimuth_deg: f64) -> Self {
        Self {
            frequency,
            level_db,
            onset_ms,
            duration_ms: 500.0,
            azimuth_deg,
            elevation_deg: 0.0,
            spectral_centroid: frequency,
            spectral_spread: 100.0,
            spectral_flux: 0.0,
            attack_time_ms: 10.0,
        }
    }
}

/// Result of evaluating one perceptual dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionResult {
    pub dimension: JndDimension,
    pub param1: f64,
    pub param2: f64,
    pub jnd: f64,
    pub difference: f64,
    pub margin: f64,
    pub passed: bool,
}

/// Aggregate validation report across all dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub all_passed: bool,
    pub any_passed: bool,
    pub results: Vec<DimensionResult>,
    pub min_margin: f64,
    pub max_margin: f64,
}

impl ValidationReport {
    pub fn failures(&self) -> Vec<&DimensionResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }

    pub fn successes(&self) -> Vec<&DimensionResult> {
        self.results.iter().filter(|r| r.passed).collect()
    }
}

// ---------------------------------------------------------------------------
// JndValidator
// ---------------------------------------------------------------------------

/// Validates whether parameter differences exceed JND thresholds.
#[derive(Debug, Clone)]
pub struct JndValidator {
    pub pitch: PitchJnd,
    pub loudness: LoudnessJnd,
    pub temporal: TemporalJnd,
    pub timbre: TimbreJnd,
    pub spatial: SpatialJnd,
}

impl JndValidator {
    pub fn new() -> Self {
        Self {
            pitch: PitchJnd::new(),
            loudness: LoudnessJnd::new(),
            temporal: TemporalJnd::new(),
            timbre: TimbreJnd::new(),
            spatial: SpatialJnd::new(),
        }
    }

    pub fn validate_pitch_discriminability(&self, f1: f64, f2: f64) -> bool {
        self.pitch.is_discriminable(f1, f2)
    }

    pub fn validate_loudness_discriminability(&self, l1: f64, l2: f64) -> bool {
        self.loudness.is_discriminable(l1, l2)
    }

    pub fn validate_temporal_discriminability(&self, t1: f64, t2: f64) -> bool {
        self.temporal.is_onset_discriminable(t1, t2)
    }

    pub fn validate_spatial_discriminability(&self, az1: f64, az2: f64) -> bool {
        self.spatial.is_azimuth_discriminable(az1, az2)
    }

    /// Compute discriminability margin for a given dimension.
    ///
    /// Returns how many JNDs apart the two values are (≥ 1.0 means discriminable).
    pub fn compute_discriminability_margin(
        &self,
        param1: f64,
        param2: f64,
        dimension: JndDimension,
    ) -> f64 {
        let diff = (param1 - param2).abs();
        let jnd = match dimension {
            JndDimension::Pitch => self.pitch.jnd_frequency(param1.min(param2).max(20.0)),
            JndDimension::Loudness => self.loudness.jnd_intensity(param1.min(param2)),
            JndDimension::Duration => self.temporal.jnd_duration(param1.min(param2).max(1.0)),
            JndDimension::Onset => self.temporal.jnd_onset(param1.min(param2)),
            JndDimension::Gap => self.temporal.jnd_gap(param1.min(param2)),
            JndDimension::SpectralCentroid => {
                self.timbre.jnd_spectral_centroid(param1.min(param2).max(1.0))
            }
            JndDimension::SpectralSpread => {
                self.timbre.jnd_spectral_spread(param1.min(param2).max(1.0))
            }
            JndDimension::SpectralFlux => {
                self.timbre.jnd_spectral_flux(param1.min(param2).max(0.001))
            }
            JndDimension::AttackTime => {
                self.timbre.jnd_attack_time(param1.min(param2).max(1.0))
            }
            JndDimension::Azimuth => self.spatial.jnd_azimuth(param1.abs().min(param2.abs())),
            JndDimension::Elevation => {
                self.spatial.jnd_elevation(param1.abs().min(param2.abs()))
            }
            JndDimension::Distance => self.spatial.jnd_distance(param1.min(param2).max(0.01)),
        };
        if jnd <= 0.0 {
            return f64::INFINITY;
        }
        diff / jnd
    }

    /// Evaluate a single dimension and return a structured result.
    fn evaluate_dimension(
        &self,
        param1: f64,
        param2: f64,
        dimension: JndDimension,
    ) -> DimensionResult {
        let margin = self.compute_discriminability_margin(param1, param2, dimension);
        let diff = (param1 - param2).abs();
        let jnd = if margin > 0.0 && margin.is_finite() {
            diff / margin
        } else {
            0.0
        };
        DimensionResult {
            dimension,
            param1,
            param2,
            jnd,
            difference: diff,
            margin,
            passed: margin >= 1.0,
        }
    }

    /// Validate all dimensions at once.
    pub fn all_dimensions_discriminable(
        &self,
        p1: &PerceptualParams,
        p2: &PerceptualParams,
    ) -> ValidationReport {
        let results = vec![
            self.evaluate_dimension(p1.frequency, p2.frequency, JndDimension::Pitch),
            self.evaluate_dimension(p1.level_db, p2.level_db, JndDimension::Loudness),
            self.evaluate_dimension(p1.onset_ms, p2.onset_ms, JndDimension::Onset),
            self.evaluate_dimension(p1.duration_ms, p2.duration_ms, JndDimension::Duration),
            self.evaluate_dimension(p1.azimuth_deg, p2.azimuth_deg, JndDimension::Azimuth),
            self.evaluate_dimension(
                p1.elevation_deg,
                p2.elevation_deg,
                JndDimension::Elevation,
            ),
            self.evaluate_dimension(
                p1.spectral_centroid,
                p2.spectral_centroid,
                JndDimension::SpectralCentroid,
            ),
            self.evaluate_dimension(
                p1.spectral_spread,
                p2.spectral_spread,
                JndDimension::SpectralSpread,
            ),
            self.evaluate_dimension(
                p1.spectral_flux,
                p2.spectral_flux,
                JndDimension::SpectralFlux,
            ),
            self.evaluate_dimension(
                p1.attack_time_ms,
                p2.attack_time_ms,
                JndDimension::AttackTime,
            ),
        ];

        let all_passed = results.iter().all(|r| r.passed);
        let any_passed = results.iter().any(|r| r.passed);
        let min_margin = results
            .iter()
            .map(|r| r.margin)
            .fold(f64::INFINITY, f64::min);
        let max_margin = results
            .iter()
            .map(|r| r.margin)
            .fold(f64::NEG_INFINITY, f64::max);

        ValidationReport {
            all_passed,
            any_passed,
            results,
            min_margin,
            max_margin,
        }
    }

    /// Quick check: is at least one dimension discriminable?
    pub fn any_dimension_discriminable(
        &self,
        p1: &PerceptualParams,
        p2: &PerceptualParams,
    ) -> bool {
        self.all_dimensions_discriminable(p1, p2).any_passed
    }

    /// Suggest the easiest dimension to modify to achieve discrimination.
    pub fn suggest_easiest_discrimination(
        &self,
        p1: &PerceptualParams,
        p2: &PerceptualParams,
    ) -> Option<JndDimension> {
        let report = self.all_dimensions_discriminable(p1, p2);
        report
            .results
            .iter()
            .filter(|r| !r.passed)
            .max_by(|a, b| {
                a.margin
                    .partial_cmp(&b.margin)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.dimension)
    }
}

impl Default for JndValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Composite JND: d-prime model-predicted discriminability
// ---------------------------------------------------------------------------

/// Signal-detection-theory d' from multi-dimensional JND margins.
///
/// d' = sqrt( sum (margin_i)^2 )  where margin_i = |delta_x_i| / JND_i.
pub fn d_prime_from_margins(margins: &[f64]) -> f64 {
    let sum_sq: f64 = margins.iter().map(|m| m * m).sum();
    sum_sq.sqrt()
}

/// Probability of correct discrimination from d' (assuming Gaussian noise).
///
/// P(correct) ~ Phi(d' / sqrt(2)) approximated via logistic function.
pub fn p_correct_from_d_prime(d_prime: f64) -> f64 {
    let x = d_prime / std::f64::consts::SQRT_2;
    1.0 / (1.0 + (-1.7 * x).exp())
}

/// Compute d' for two perceptual parameter sets across all dimensions.
pub fn multi_dimensional_d_prime(p1: &PerceptualParams, p2: &PerceptualParams) -> f64 {
    let validator = JndValidator::new();
    let report = validator.all_dimensions_discriminable(p1, p2);
    let margins: Vec<f64> = report.results.iter().map(|r| r.margin).collect();
    d_prime_from_margins(&margins)
}

// ---------------------------------------------------------------------------
// Batch JND analysis
// ---------------------------------------------------------------------------

/// Pairwise JND validation matrix for a set of streams.
#[derive(Debug, Clone)]
pub struct JndMatrix {
    pub n: usize,
    reports: Vec<Vec<Option<ValidationReport>>>,
}

impl JndMatrix {
    /// Build pairwise JND matrix for a slice of perceptual parameter sets.
    pub fn compute(params: &[PerceptualParams]) -> Self {
        let n = params.len();
        let validator = JndValidator::new();
        let mut reports = vec![vec![None; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let report = validator.all_dimensions_discriminable(&params[i], &params[j]);
                reports[i][j] = Some(report.clone());
                reports[j][i] = Some(report);
            }
        }
        Self { n, reports }
    }

    /// Get the validation report between items i and j.
    pub fn get(&self, i: usize, j: usize) -> Option<&ValidationReport> {
        if i < self.n && j < self.n {
            self.reports[i][j].as_ref()
        } else {
            None
        }
    }

    /// Pairs that are NOT discriminable on any dimension.
    pub fn indiscriminable_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if let Some(report) = &self.reports[i][j] {
                    if !report.any_passed {
                        pairs.push((i, j));
                    }
                }
            }
        }
        pairs
    }

    /// Pairs where all dimensions are discriminable.
    pub fn fully_discriminable_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if let Some(report) = &self.reports[i][j] {
                    if report.all_passed {
                        pairs.push((i, j));
                    }
                }
            }
        }
        pairs
    }

    /// Are all pairs discriminable on at least one dimension?
    pub fn all_pairs_discriminable(&self) -> bool {
        self.indiscriminable_pairs().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_jnd_at_1000hz() {
        let model = PitchJnd::new();
        let jnd = model.jnd_frequency(1000.0);
        assert!(
            (jnd - 3.5).abs() < 1.0,
            "JND at 1000 Hz should be ~3.5 Hz, got {jnd}"
        );
    }

    #[test]
    fn test_pitch_jnd_increases_below_500() {
        let model = PitchJnd::new();
        let jnd_200 = model.weber_fraction(200.0);
        let jnd_1000 = model.weber_fraction(1000.0);
        assert!(
            jnd_200 > jnd_1000,
            "Weber fraction at 200 Hz ({jnd_200}) should exceed 1000 Hz ({jnd_1000})"
        );
    }

    #[test]
    fn test_pitch_jnd_increases_above_5khz() {
        let model = PitchJnd::new();
        let jnd_8k = model.weber_fraction(8000.0);
        let jnd_2k = model.weber_fraction(2000.0);
        assert!(
            jnd_8k > jnd_2k,
            "Weber fraction at 8 kHz ({jnd_8k}) should exceed 2 kHz ({jnd_2k})"
        );
    }

    #[test]
    fn test_loudness_jnd_moderate() {
        let model = LoudnessJnd::new();
        let jnd = model.jnd_intensity(60.0);
        assert!(
            (jnd - 1.0).abs() < 0.1,
            "JND at 60 dB should be ~1 dB, got {jnd}"
        );
    }

    #[test]
    fn test_loudness_jnd_low_level() {
        let model = LoudnessJnd::new();
        let jnd_low = model.jnd_intensity(5.0);
        let jnd_mod = model.jnd_intensity(50.0);
        assert!(
            jnd_low > jnd_mod,
            "JND at 5 dB ({jnd_low}) should exceed 50 dB ({jnd_mod})"
        );
    }

    #[test]
    fn test_temporal_jnd_short_duration() {
        let model = TemporalJnd::new();
        let jnd = model.jnd_duration(50.0);
        assert!(jnd >= 10.0, "Short-duration JND >= 10 ms, got {jnd}");
    }

    #[test]
    fn test_onset_asynchrony_baseline() {
        let model = TemporalJnd::new();
        let jnd = model.jnd_onset(200.0);
        assert!(
            (jnd - 20.0).abs() < 5.0,
            "Onset JND should be near 20 ms, got {jnd}"
        );
    }

    #[test]
    fn test_maa_at_zero() {
        let model = SpatialJnd::new();
        let maa = model.jnd_azimuth(0.0);
        assert!(
            (maa - 1.0).abs() < 0.5,
            "MAA at 0° should be ~1°, got {maa}"
        );
    }

    #[test]
    fn test_maa_increases_at_90() {
        let model = SpatialJnd::new();
        let maa_0 = model.jnd_azimuth(0.0);
        let maa_90 = model.jnd_azimuth(90.0);
        assert!(
            maa_90 > maa_0 * 3.0,
            "MAA at 90° ({maa_90}) >> MAA at 0° ({maa_0})"
        );
    }

    #[test]
    fn test_validator_discriminable() {
        let v = JndValidator::new();
        assert!(v.validate_pitch_discriminability(1000.0, 1050.0));
        assert!(!v.validate_pitch_discriminability(1000.0, 1001.0));
    }

    #[test]
    fn test_validator_non_discriminable() {
        let v = JndValidator::new();
        let p1 = PerceptualParams::basic(1000.0, 60.0, 100.0, 0.0);
        let p2 = PerceptualParams::basic(1001.0, 60.1, 100.5, 0.1);
        let report = v.all_dimensions_discriminable(&p1, &p2);
        assert!(
            !report.all_passed,
            "Nearly identical params should not all pass"
        );
    }

    #[test]
    fn test_discriminability_margin() {
        let v = JndValidator::new();
        let margin =
            v.compute_discriminability_margin(1000.0, 1050.0, JndDimension::Pitch);
        assert!(margin > 1.0, "50 Hz apart at 1 kHz: margin > 1 JND, got {margin}");
    }

    #[test]
    fn test_d_prime_from_margins() {
        let margins = vec![2.0, 3.0, 0.0];
        let dp = d_prime_from_margins(&margins);
        let expected = (4.0 + 9.0_f64).sqrt();
        assert!((dp - expected).abs() < 0.01);
    }

    #[test]
    fn test_p_correct_increases_with_d_prime() {
        let p1 = p_correct_from_d_prime(0.5);
        let p2 = p_correct_from_d_prime(2.0);
        let p3 = p_correct_from_d_prime(4.0);
        assert!(p1 < p2 && p2 < p3, "P(correct) should increase with d'");
        assert!(p3 > 0.95, "Large d' should give P(correct) near 1");
    }

    #[test]
    fn test_jnd_matrix_all_discriminable() {
        let params = vec![
            PerceptualParams::basic(200.0, 40.0, 0.0, -30.0),
            PerceptualParams::basic(1000.0, 60.0, 100.0, 0.0),
            PerceptualParams::basic(4000.0, 80.0, 200.0, 30.0),
        ];
        let matrix = JndMatrix::compute(&params);
        assert!(
            matrix.all_pairs_discriminable(),
            "Well-separated params should all be discriminable"
        );
    }

    #[test]
    fn test_elevation_jnd() {
        let model = SpatialJnd::new();
        let jnd_0 = model.jnd_elevation(0.0);
        let jnd_45 = model.jnd_elevation(45.0);
        assert!((jnd_0 - 4.0).abs() < 1.0, "Elevation JND at 0° ~4°");
        assert!(jnd_45 > jnd_0, "Elevation JND increases off-axis");
    }

    #[test]
    fn test_timbre_jnd_centroid() {
        let model = TimbreJnd::new();
        let jnd = model.jnd_spectral_centroid(2000.0);
        assert!(
            (jnd - 100.0).abs() < 20.0,
            "Centroid JND at 2 kHz ~100 Hz, got {jnd}"
        );
    }
}
