//! Loudness perception models for psychoacoustic analysis.
//!
//! Implements Zwicker loudness (ISO 532B), Stevens' power law, ISO 226
//! equal-loudness contours, and standard frequency-weighting filters
//! (A-weighting, C-weighting).

use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of critical bands in the Bark scale (0–24 Bark).
pub const NUM_BARK_BANDS: usize = 24;

/// Reference sound pressure level (20 µPa) expressed in dB SPL.
const REFERENCE_DB: f64 = 0.0;

/// Reference intensity used by the Stevens model (dimensionless, = 1.0).
const STEVENS_REF_INTENSITY: f64 = 1.0;

/// Stevens power-law exponent for loudness of a 1 kHz tone.
const STEVENS_EXPONENT: f64 = 0.3;

/// Standard Bark-band centre frequencies (Hz) – one per critical band.
pub const BARK_CENTER_FREQUENCIES: [f64; NUM_BARK_BANDS] = [
    50.0, 150.0, 250.0, 350.0, 450.0, 570.0, 700.0, 840.0, 1000.0, 1170.0,
    1370.0, 1600.0, 1850.0, 2150.0, 2500.0, 2900.0, 3400.0, 4000.0, 4800.0,
    5800.0, 7000.0, 8500.0, 10500.0, 13500.0,
];

/// Bark-band edges – 25 edges delineating 24 bands.
pub const BARK_EDGES: [f64; 25] = [
    20.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0,
    1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0,
    5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 15500.0,
];

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert a frequency in Hz to critical-band rate in Bark.
///
/// Uses the Traunmüller (1990) formula.
#[inline]
pub fn hz_to_bark(f: f64) -> f64 {
    26.81 * f / (1960.0 + f) - 0.53
}

/// Convert a critical-band rate in Bark back to Hz.
#[inline]
pub fn bark_to_hz(z: f64) -> f64 {
    1960.0 * (z + 0.53) / (26.81 - z - 0.53)
}

/// Convert a level in dB SPL to linear intensity (power-proportional).
#[inline]
pub fn db_to_intensity(db: f64) -> f64 {
    10.0_f64.powf(db / 10.0)
}

/// Convert linear intensity to dB SPL.
#[inline]
pub fn intensity_to_db(intensity: f64) -> f64 {
    if intensity <= 0.0 {
        return -200.0; // silence floor
    }
    10.0 * intensity.log10()
}

/// Convert a level in dB SPL to linear pressure ratio (voltage-proportional).
#[inline]
pub fn db_to_amplitude(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear amplitude to dB SPL.
#[inline]
pub fn amplitude_to_db(amp: f64) -> f64 {
    if amp <= 0.0 {
        return -200.0;
    }
    20.0 * amp.log10()
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Complete result of a Zwicker loudness computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoudnessResult {
    /// Specific loudness in sone/Bark for each critical band.
    pub specific_loudness: Vec<f64>,
    /// Total loudness integrated across all bands, in sone.
    pub total_loudness_sone: f64,
    /// Overall loudness level in phon.
    pub loudness_level_phon: f64,
}

// ---------------------------------------------------------------------------
// WeightingType
// ---------------------------------------------------------------------------

/// Frequency-weighting curve selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightingType {
    /// A-weighting – approximates human sensitivity at moderate levels.
    AWeighting,
    /// C-weighting – flatter; approximates sensitivity at high levels.
    CWeighting,
    /// Z-weighting – flat / no weighting.
    ZWeighting,
}

// ===========================================================================
// ZwickerLoudnessModel
// ===========================================================================

/// Stationary loudness model following Zwicker (ISO 532B).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZwickerLoudnessModel {
    /// Threshold-in-quiet values for each Bark band centre frequency (dB SPL).
    thresholds_quiet: Vec<f64>,
    /// Bark bandwidth (constant 1 Bark per band for the 24-band model).
    bark_bandwidth: f64,
}

impl ZwickerLoudnessModel {
    /// Create a new model instance, pre-computing thresholds in quiet.
    pub fn new() -> Self {
        let thresholds_quiet: Vec<f64> = BARK_CENTER_FREQUENCIES
            .iter()
            .map(|&f| Self::threshold_in_quiet_static(f))
            .collect();
        Self {
            thresholds_quiet,
            bark_bandwidth: 1.0,
        }
    }

    // -- Threshold in quiet ---------------------------------------------------

    /// Absolute threshold of hearing in dB SPL (static helper).
    ///
    /// Standard approximation (ISO 226 simplified):
    ///
    /// ```text
    /// T_q(f) = 3.64·(f/1000)^{-0.8}
    ///        - 6.5·exp(-0.6·(f/1000 - 3.3)^2)
    ///        + 1e-3·(f/1000)^4
    /// ```
    fn threshold_in_quiet_static(freq_hz: f64) -> f64 {
        let f_khz = freq_hz / 1000.0;
        if f_khz <= 0.0 {
            return 120.0; // out of range
        }
        let term1 = 3.64 * f_khz.powf(-0.8);
        let term2 = -6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp();
        let term3 = 1e-3 * f_khz.powi(4);
        let tq = term1 + term2 + term3;
        // Clamp to plausible auditory range
        tq.clamp(-10.0, 120.0)
    }

    /// Absolute threshold of hearing in dB SPL for the given frequency.
    pub fn threshold_in_quiet(&self, freq_hz: f64) -> f64 {
        Self::threshold_in_quiet_static(freq_hz)
    }

    // -- Specific loudness ----------------------------------------------------

    /// Compute specific loudness N'(z) in sone/Bark for one critical band.
    ///
    /// Zwicker formula (simplified):
    ///
    /// ```text
    /// N'(z) = 0.08 * (E_TQ / E_0)^{0.23}
    ///       * [ (0.5 + 0.5 * E / E_TQ)^{0.23} - 1 ]
    /// ```
    ///
    /// where E = 10^(excitation_db / 10), E_TQ = 10^(threshold_db / 10),
    /// E_0 = 10^(REFERENCE_DB / 10).
    pub fn specific_loudness(&self, excitation_db: f64, threshold_db: f64) -> f64 {
        if excitation_db <= threshold_db {
            return 0.0; // below threshold → inaudible
        }

        let e = db_to_intensity(excitation_db);
        let e_tq = db_to_intensity(threshold_db);
        let e_0 = db_to_intensity(REFERENCE_DB);

        if e_tq <= 0.0 || e_0 <= 0.0 {
            return 0.0;
        }

        let ratio_tq_e0 = (e_tq / e_0).powf(0.23);
        let inner = 0.5 + 0.5 * (e / e_tq);
        let inner_pow = inner.powf(0.23) - 1.0;

        let n_prime = 0.08 * ratio_tq_e0 * inner_pow;
        n_prime.max(0.0)
    }

    /// Specific loudness for each Bark band given a vector of band levels (dB SPL).
    pub fn specific_loudness_band(&self, band_level_db: &[f64]) -> Vec<f64> {
        band_level_db
            .iter()
            .enumerate()
            .map(|(i, &level)| {
                let tq = if i < self.thresholds_quiet.len() {
                    self.thresholds_quiet[i]
                } else {
                    Self::threshold_in_quiet_static(1000.0)
                };
                self.specific_loudness(level, tq)
            })
            .collect()
    }

    /// Integrate specific loudness across bands to obtain total loudness (sone).
    ///
    /// N = Σ N'(z) · Δz    (Δz = 1 Bark per band)
    pub fn total_loudness(&self, specific_loudness: &[f64]) -> f64 {
        specific_loudness.iter().sum::<f64>() * self.bark_bandwidth
    }

    // -- Sone ↔ Phon conversions ---------------------------------------------

    /// Convert total loudness in sone to loudness level in phon.
    ///
    /// For N ≥ 1 sone:  L_N = 40 + 33.22 · log₁₀(N)
    /// For N <  1 sone:  L_N = 40 · (N + 0.0005)^{0.35}
    pub fn loudness_level_phon(&self, loudness_sone: f64) -> f64 {
        self.sone_to_phon(loudness_sone)
    }

    /// Convert sone to phon (alias for `loudness_level_phon`).
    pub fn sone_to_phon(&self, sone: f64) -> f64 {
        if sone <= 0.0 {
            return 0.0;
        }
        if sone >= 1.0 {
            40.0 + 33.22 * sone.log10()
        } else {
            // Below 1 sone: inverse of the sub-threshold phon→sone mapping.
            // phon = 40 * (sone + 0.0005)^0.35
            40.0 * (sone + 0.0005).powf(0.35)
        }
    }

    /// Convert phon to sone.
    ///
    /// For phon ≥ 40:  N = 2^{(phon − 40)/10}
    /// For phon <  40:  N = (phon / 40)^{1/0.35} − 0.0005
    pub fn phon_to_sone(&self, phon: f64) -> f64 {
        if phon <= 0.0 {
            return 0.0;
        }
        if phon >= 40.0 {
            2.0_f64.powf((phon - 40.0) / 10.0)
        } else {
            let raw = (phon / 40.0).powf(1.0 / 0.35) - 0.0005;
            raw.max(0.0)
        }
    }

    /// Convert an excitation pattern (dB per Bark band) to specific loudness.
    pub fn excitation_to_specific_loudness(
        &self,
        excitation_pattern: &[f64],
    ) -> Vec<f64> {
        self.specific_loudness_band(excitation_pattern)
    }

    /// Full loudness computation from a 24-band Bark spectrum.
    pub fn compute_loudness_from_spectrum(
        &self,
        spectrum_db: &[f64; NUM_BARK_BANDS],
    ) -> LoudnessResult {
        let specific = self.specific_loudness_band(spectrum_db);
        let total_sone = self.total_loudness(&specific);
        let phon = self.sone_to_phon(total_sone);
        LoudnessResult {
            specific_loudness: specific,
            total_loudness_sone: total_sone,
            loudness_level_phon: phon,
        }
    }
}

impl Default for ZwickerLoudnessModel {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// StevensLoudnessModel
// ===========================================================================

/// Simple Stevens' power-law loudness model.
///
/// L = k · I^{0.3}, calibrated so that L = 1 sone at I = 1.0 (reference).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StevensLoudnessModel {
    /// Power-law exponent (default 0.3).
    exponent: f64,
    /// Scaling constant k so that loudness(I=1) = 1 sone.
    k: f64,
}

impl StevensLoudnessModel {
    /// Create a new Stevens model with default exponent 0.3.
    pub fn new() -> Self {
        let exponent = STEVENS_EXPONENT;
        // k chosen so that k * 1.0^exponent = 1.0  ⟹  k = 1.0
        let k = 1.0 / STEVENS_REF_INTENSITY.powf(exponent);
        Self { exponent, k }
    }

    /// Perceived loudness (sone-like) from linear intensity.
    ///
    /// Returns 0 for non-positive intensity.
    pub fn loudness(&self, intensity: f64) -> f64 {
        if intensity <= 0.0 {
            return 0.0;
        }
        self.k * intensity.powf(self.exponent)
    }

    /// Perceived loudness from a level in dB SPL.
    pub fn loudness_from_db(&self, level_db: f64) -> f64 {
        let intensity = db_to_intensity(level_db);
        self.loudness(intensity)
    }

    /// Ratio of perceived loudness between two levels.
    pub fn loudness_ratio(&self, level1_db: f64, level2_db: f64) -> f64 {
        let l1 = self.loudness_from_db(level1_db);
        let l2 = self.loudness_from_db(level2_db);
        if l2 == 0.0 {
            return f64::INFINITY;
        }
        l1 / l2
    }

    /// dB SPL level that yields a given loudness (sone) value.
    ///
    /// Inverse of the power law: I = (sone / k)^{1/exponent}, then to dB.
    pub fn equal_loudness_level(&self, sone: f64) -> f64 {
        if sone <= 0.0 {
            return -200.0;
        }
        let intensity = (sone / self.k).powf(1.0 / self.exponent);
        intensity_to_db(intensity)
    }
}

impl Default for StevensLoudnessModel {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// EqualLoudnessContour (ISO 226:2003 approximation)
// ===========================================================================

/// ISO 226:2003 equal-loudness contour model.
///
/// Stores parametric coefficients for the 29 standard frequencies defined in
/// the standard and interpolates for arbitrary frequencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqualLoudnessContour {
    /// Standard frequencies (Hz) from ISO 226.
    frequencies: Vec<f64>,
    /// α_f exponent per frequency.
    alpha_f: Vec<f64>,
    /// L_U reference level per frequency (dB).
    l_u: Vec<f64>,
    /// T_f threshold per frequency (dB SPL).
    t_f: Vec<f64>,
}

impl EqualLoudnessContour {
    /// Create a new contour model pre-loaded with ISO 226 coefficients.
    pub fn new() -> Self {
        // ISO 226:2003 Table 1 – 29 reference frequencies
        let frequencies = vec![
            20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
            200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0,
            1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
            10000.0, 12500.0,
        ];

        // α_f – exponent of loudness perception per frequency
        let alpha_f = vec![
            0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349,
            0.330, 0.315, 0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250,
            0.246, 0.244, 0.243, 0.243, 0.243, 0.242, 0.242, 0.245, 0.254,
            0.271, 0.301,
        ];

        // L_U – magnitude of the linear transfer function at each freq (dB)
        let l_u = vec![
            -31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2,
            -4.5, -3.1, -2.0, -1.1, -0.4, 0.0, 0.3, 0.5, 0.0, -2.7, -4.1,
            -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1,
        ];

        // T_f – threshold of hearing (dB SPL) at each freq
        let t_f = vec![
            78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9,
            14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3,
            -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3,
        ];

        Self {
            frequencies,
            alpha_f,
            l_u,
            t_f,
        }
    }

    /// Find the bracketing indices for log-interpolation of `freq_hz`.
    fn find_bracket(&self, freq_hz: f64) -> (usize, usize, f64) {
        let n = self.frequencies.len();
        if freq_hz <= self.frequencies[0] {
            return (0, 0, 0.0);
        }
        if freq_hz >= self.frequencies[n - 1] {
            return (n - 1, n - 1, 0.0);
        }
        for i in 0..n - 1 {
            if freq_hz >= self.frequencies[i] && freq_hz <= self.frequencies[i + 1] {
                let log_f = freq_hz.ln();
                let log_lo = self.frequencies[i].ln();
                let log_hi = self.frequencies[i + 1].ln();
                let t = if (log_hi - log_lo).abs() < 1e-12 {
                    0.0
                } else {
                    (log_f - log_lo) / (log_hi - log_lo)
                };
                return (i, i + 1, t);
            }
        }
        (n - 1, n - 1, 0.0)
    }

    /// Linearly interpolate between two values.
    #[inline]
    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + t * (b - a)
    }

    /// Interpolate a coefficient at an arbitrary frequency.
    fn interp_coeff(&self, coeffs: &[f64], freq_hz: f64) -> f64 {
        let (lo, hi, t) = self.find_bracket(freq_hz);
        Self::lerp(coeffs[lo], coeffs[hi], t)
    }

    /// SPL in dB at a given frequency for a given phon level.
    ///
    /// ISO 226:2003 forward equation:
    ///
    /// ```text
    /// A_f = 4.47e-3 * (10^{0.025·L_N} − 1.15)
    ///     + (0.4 · 10^{(T_f + L_U) / 10 − 9})^{α_f}
    /// L_p = (A_f − L_U) / α_f + 94   (approximate inversion)
    /// ```
    ///
    /// Here we use a simplified parametric model that matches the standard
    /// within a few dB for most practical phon levels (20–90 phon).
    pub fn contour_at_phon(&self, phon: f64, freq_hz: f64) -> f64 {
        let af = self.interp_coeff(&self.alpha_f, freq_hz);
        let lu = self.interp_coeff(&self.l_u, freq_hz);
        let tf = self.interp_coeff(&self.t_f, freq_hz);

        // ISO 226 forward model (simplified)
        let bf = (0.4 * 10.0_f64.powf((tf + lu) / 10.0 - 9.0)).powf(af);
        let ln_exp = 10.0_f64.powf(0.025 * phon);
        let a_f = 4.47e-3 * (ln_exp - 1.15) + bf;

        if a_f <= 0.0 {
            return tf; // below threshold
        }

        // Invert to SPL
        let spl = (10.0 / af) * a_f.powf(1.0 / af).log10() - lu + 94.0;
        spl
    }

    /// Returns (frequency, SPL) pairs along the equal-loudness contour.
    pub fn contour_values(&self, phon: f64) -> Vec<(f64, f64)> {
        self.frequencies
            .iter()
            .map(|&f| (f, self.contour_at_phon(phon, f)))
            .collect()
    }

    /// Hearing threshold (0-phon contour) at the given frequency.
    pub fn hearing_threshold(&self, freq_hz: f64) -> f64 {
        self.interp_coeff(&self.t_f, freq_hz)
    }
}

impl Default for EqualLoudnessContour {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// LoudnessNormalization  (A / C / Z weighting)
// ===========================================================================

/// Frequency-weighting filters (A, C, Z) and weighted-SPL computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoudnessNormalization {
    _private: (),
}

impl LoudnessNormalization {
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// A-weighting relative response in dB at a given frequency.
    ///
    /// ```text
    /// R_A(f) = 12194² · f⁴
    ///        / [ (f² + 20.6²) · √((f² + 107.7²)(f² + 737.9²)) · (f² + 12194²) ]
    ///
    /// A(f)  = 20·log₁₀(R_A) + 2.00
    /// ```
    pub fn a_weighting(&self, freq_hz: f64) -> f64 {
        let f2 = freq_hz * freq_hz;

        let num = 12194.0_f64.powi(2) * f2 * f2;

        let d1 = f2 + 20.6_f64.powi(2);
        let d2 = ((f2 + 107.7_f64.powi(2)) * (f2 + 737.9_f64.powi(2))).sqrt();
        let d3 = f2 + 12194.0_f64.powi(2);

        let denom = d1 * d2 * d3;
        if denom <= 0.0 {
            return -200.0;
        }

        let r_a = num / denom;
        if r_a <= 0.0 {
            return -200.0;
        }

        20.0 * r_a.log10() + 2.00
    }

    /// C-weighting relative response in dB at a given frequency.
    ///
    /// ```text
    /// R_C(f) = 12194² · f²
    ///        / [ (f² + 20.6²) · (f² + 12194²) ]
    ///
    /// C(f)  = 20·log₁₀(R_C) + 0.06
    /// ```
    pub fn c_weighting(&self, freq_hz: f64) -> f64 {
        let f2 = freq_hz * freq_hz;

        let num = 12194.0_f64.powi(2) * f2;

        let d1 = f2 + 20.6_f64.powi(2);
        let d2 = f2 + 12194.0_f64.powi(2);

        let denom = d1 * d2;
        if denom <= 0.0 {
            return -200.0;
        }

        let r_c = num / denom;
        if r_c <= 0.0 {
            return -200.0;
        }

        20.0 * r_c.log10() + 0.06
    }

    /// Apply A-weighting to a list of (frequency, level_dB) pairs.
    pub fn apply_a_weighting(&self, spectrum: &[(f64, f64)]) -> Vec<(f64, f64)> {
        spectrum
            .iter()
            .map(|&(f, level)| (f, level + self.a_weighting(f)))
            .collect()
    }

    /// Apply C-weighting to a list of (frequency, level_dB) pairs.
    pub fn apply_c_weighting(&self, spectrum: &[(f64, f64)]) -> Vec<(f64, f64)> {
        spectrum
            .iter()
            .map(|&(f, level)| (f, level + self.c_weighting(f)))
            .collect()
    }

    /// Compute the overall weighted sound-pressure level.
    ///
    /// Energy-sum of weighted band levels:
    ///
    /// ```text
    /// L_w = 10 · log₁₀( Σ 10^{L_i / 10} )
    /// ```
    pub fn weighted_spl(
        &self,
        spectrum: &[(f64, f64)],
        weighting: WeightingType,
    ) -> f64 {
        let weighted: Vec<(f64, f64)> = match weighting {
            WeightingType::AWeighting => self.apply_a_weighting(spectrum),
            WeightingType::CWeighting => self.apply_c_weighting(spectrum),
            WeightingType::ZWeighting => spectrum.to_vec(),
        };

        let energy_sum: f64 = weighted
            .iter()
            .map(|&(_, level)| db_to_intensity(level))
            .sum();

        intensity_to_db(energy_sum)
    }
}

impl Default for LoudnessNormalization {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// LoudnessEqualizer
// ===========================================================================

/// Utility for equalizing perceived loudness across frequencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoudnessEqualizer {
    contour: EqualLoudnessContour,
}

impl LoudnessEqualizer {
    /// Create a new equalizer backed by ISO 226 contours.
    pub fn new() -> Self {
        Self {
            contour: EqualLoudnessContour::new(),
        }
    }

    /// Adjust levels so that each frequency band is perceived at the same
    /// loudness as a 1 kHz tone at the mean level.
    ///
    /// Algorithm:
    /// 1. Determine the average level → use as target phon reference.
    /// 2. For each frequency, compute the SPL offset from the equal-loudness
    ///    contour at that phon level relative to 1 kHz.
    /// 3. Apply compensation.
    pub fn equalize_loudness(
        &self,
        levels_db: &[f64],
        frequencies: &[f64],
    ) -> Vec<f64> {
        if levels_db.is_empty() || frequencies.is_empty() {
            return Vec::new();
        }
        let n = levels_db.len().min(frequencies.len());

        // Mean level as phon estimate (1 kHz reference)
        let mean_level: f64 = levels_db[..n].iter().sum::<f64>() / n as f64;

        // SPL at 1 kHz for this phon value
        let spl_1k = self.contour.contour_at_phon(mean_level, 1000.0);

        (0..n)
            .map(|i| {
                let spl_f = self.contour.contour_at_phon(mean_level, frequencies[i]);
                // Compensation: difference between contour at this freq and at 1 kHz
                let compensation = spl_f - spl_1k;
                levels_db[i] - compensation
            })
            .collect()
    }

    /// Compute the SPL required at each frequency to achieve `target_phon`.
    pub fn target_loudness_levels(
        &self,
        target_phon: f64,
        frequencies: &[f64],
    ) -> Vec<f64> {
        frequencies
            .iter()
            .map(|&f| self.contour.contour_at_phon(target_phon, f))
            .collect()
    }
}

impl Default for LoudnessEqualizer {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Additional utility functions
// ===========================================================================

/// Compute the energy-average of a set of dB levels.
pub fn energy_average_db(levels: &[f64]) -> f64 {
    if levels.is_empty() {
        return -200.0;
    }
    let sum: f64 = levels.iter().map(|&l| db_to_intensity(l)).sum();
    intensity_to_db(sum / levels.len() as f64)
}

/// Compute the energy-sum of a set of dB levels.
pub fn energy_sum_db(levels: &[f64]) -> f64 {
    if levels.is_empty() {
        return -200.0;
    }
    let sum: f64 = levels.iter().map(|&l| db_to_intensity(l)).sum();
    intensity_to_db(sum)
}

/// Map a linear frequency axis onto Bark and return (bark, level) pairs.
pub fn spectrum_to_bark_bands(
    freq_levels: &[(f64, f64)],
) -> [f64; NUM_BARK_BANDS] {
    let mut band_energy = [0.0_f64; NUM_BARK_BANDS];
    let mut band_count = [0u32; NUM_BARK_BANDS];

    for &(f, level) in freq_levels {
        let bark = hz_to_bark(f);
        let band = (bark.round() as usize).min(NUM_BARK_BANDS - 1);
        band_energy[band] += db_to_intensity(level);
        band_count[band] += 1;
    }

    let mut result = [0.0_f64; NUM_BARK_BANDS];
    for i in 0..NUM_BARK_BANDS {
        if band_count[i] > 0 {
            result[i] = intensity_to_db(band_energy[i] / band_count[i] as f64);
        } else {
            result[i] = -200.0; // silence
        }
    }
    result
}

/// Compute A-weighted equivalent continuous sound level (L_Aeq) from a time
/// series of (frequency, level) snapshots.
pub fn l_aeq(snapshots: &[Vec<(f64, f64)>]) -> f64 {
    if snapshots.is_empty() {
        return -200.0;
    }
    let norm = LoudnessNormalization::new();
    let energy_sum: f64 = snapshots
        .iter()
        .map(|s| {
            let la = norm.weighted_spl(s, WeightingType::AWeighting);
            db_to_intensity(la)
        })
        .sum();
    intensity_to_db(energy_sum / snapshots.len() as f64)
}

/// Compute the percentile level (L_N) from a sorted slice of dB values.
///
/// E.g. `percentile_level(sorted, 10)` → L10 (exceeded 10% of the time).
pub fn percentile_level(sorted_descending: &[f64], percent: u8) -> f64 {
    if sorted_descending.is_empty() {
        return -200.0;
    }
    let idx = ((percent as f64 / 100.0) * sorted_descending.len() as f64) as usize;
    let idx = idx.min(sorted_descending.len() - 1);
    sorted_descending[idx]
}

/// Estimate loudness in sone from a single dB SPL value at 1 kHz using the
/// Zwicker model (shortcut for quick estimates).
pub fn quick_loudness_sone(level_db_spl: f64) -> f64 {
    let model = ZwickerLoudnessModel::new();
    let tq = model.threshold_in_quiet(1000.0);
    let specific = model.specific_loudness(level_db_spl, tq);
    // Scale from sone/Bark to approximate total sone (assuming roughly
    // single-band stimulus)
    specific
}

/// Estimate phon from a single dB SPL value at 1 kHz.
pub fn quick_loudness_phon(level_db_spl: f64) -> f64 {
    let model = ZwickerLoudnessModel::new();
    let sone = quick_loudness_sone(level_db_spl);
    model.sone_to_phon(sone)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;
    const LOOSE_EPSILON: f64 = 1.0; // 1 dB tolerance for psychoacoustic models

    // --- Bark-scale helpers -------------------------------------------------

    #[test]
    fn test_hz_to_bark_and_back() {
        for &f in &[100.0, 500.0, 1000.0, 4000.0, 10000.0] {
            let bark = hz_to_bark(f);
            let recovered = bark_to_hz(bark);
            assert!(
                (recovered - f).abs() < 1.0,
                "Round-trip failed for {f} Hz: got {recovered}"
            );
        }
    }

    // --- Threshold in quiet -------------------------------------------------

    #[test]
    fn test_threshold_in_quiet_lowest_around_3_4khz() {
        let model = ZwickerLoudnessModel::new();
        let tq_1k = model.threshold_in_quiet(1000.0);
        let tq_3k = model.threshold_in_quiet(3500.0);
        // 3–4 kHz region should have a lower threshold than 1 kHz
        assert!(
            tq_3k < tq_1k,
            "Threshold at 3.5 kHz ({tq_3k:.1}) should be lower than at 1 kHz ({tq_1k:.1})"
        );
    }

    #[test]
    fn test_threshold_in_quiet_rises_at_low_frequencies() {
        let model = ZwickerLoudnessModel::new();
        let tq_100 = model.threshold_in_quiet(100.0);
        let tq_1k = model.threshold_in_quiet(1000.0);
        assert!(
            tq_100 > tq_1k,
            "Threshold at 100 Hz ({tq_100:.1}) should be higher than at 1 kHz ({tq_1k:.1})"
        );
    }

    #[test]
    fn test_threshold_in_quiet_rises_at_high_frequencies() {
        let model = ZwickerLoudnessModel::new();
        let tq_12k = model.threshold_in_quiet(12000.0);
        let tq_4k = model.threshold_in_quiet(4000.0);
        assert!(
            tq_12k > tq_4k,
            "Threshold at 12 kHz ({tq_12k:.1}) should be higher than at 4 kHz ({tq_4k:.1})"
        );
    }

    // --- Sone ↔ Phon --------------------------------------------------------

    #[test]
    fn test_sone_phon_inverse() {
        let model = ZwickerLoudnessModel::new();
        for &phon in &[40.0, 50.0, 60.0, 70.0, 80.0] {
            let sone = model.phon_to_sone(phon);
            let recovered_phon = model.sone_to_phon(sone);
            assert!(
                (recovered_phon - phon).abs() < LOOSE_EPSILON,
                "Round-trip failed for {phon} phon: got {recovered_phon:.2}"
            );
        }
    }

    #[test]
    fn test_1_sone_equals_40_phon() {
        let model = ZwickerLoudnessModel::new();
        let phon = model.sone_to_phon(1.0);
        assert!(
            (phon - 40.0).abs() < LOOSE_EPSILON,
            "1 sone should be 40 phon, got {phon:.2}"
        );
    }

    #[test]
    fn test_phon_to_sone_40_is_1() {
        let model = ZwickerLoudnessModel::new();
        let sone = model.phon_to_sone(40.0);
        assert!(
            (sone - 1.0).abs() < 0.01,
            "40 phon should be 1 sone, got {sone:.4}"
        );
    }

    // --- Stevens power law --------------------------------------------------

    #[test]
    fn test_stevens_doubling_intensity_less_than_doubles_loudness() {
        let model = StevensLoudnessModel::new();
        let l1 = model.loudness(1.0);
        let l2 = model.loudness(2.0);
        // With exponent 0.3, doubling I gives 2^0.3 ≈ 1.23× loudness
        assert!(
            l2 < 2.0 * l1,
            "Doubling intensity should less than double loudness: L(1)={l1:.4}, L(2)={l2:.4}"
        );
        assert!(
            l2 > l1,
            "Doubling intensity should increase loudness"
        );
    }

    #[test]
    fn test_stevens_reference_loudness_is_one() {
        let model = StevensLoudnessModel::new();
        let l_ref = model.loudness(STEVENS_REF_INTENSITY);
        assert!(
            (l_ref - 1.0).abs() < EPSILON,
            "Loudness at reference intensity should be 1.0, got {l_ref}"
        );
    }

    // --- A-weighting --------------------------------------------------------

    #[test]
    fn test_a_weighting_zero_at_1khz() {
        let norm = LoudnessNormalization::new();
        let a_1k = norm.a_weighting(1000.0);
        assert!(
            a_1k.abs() < 0.5,
            "A-weighting at 1 kHz should be ~0 dB, got {a_1k:.2}"
        );
    }

    #[test]
    fn test_a_weighting_negative_at_low_frequencies() {
        let norm = LoudnessNormalization::new();
        let a_100 = norm.a_weighting(100.0);
        assert!(
            a_100 < -10.0,
            "A-weighting at 100 Hz should be strongly negative, got {a_100:.2}"
        );
    }

    #[test]
    fn test_a_weighting_negative_at_very_high_frequencies() {
        let norm = LoudnessNormalization::new();
        let a_15k = norm.a_weighting(15000.0);
        assert!(
            a_15k < 0.0,
            "A-weighting at 15 kHz should be negative, got {a_15k:.2}"
        );
    }

    // --- C-weighting --------------------------------------------------------

    #[test]
    fn test_c_weighting_flatter_than_a_at_low_freq() {
        let norm = LoudnessNormalization::new();
        let a_100 = norm.a_weighting(100.0);
        let c_100 = norm.c_weighting(100.0);
        assert!(
            c_100 > a_100,
            "C-weighting should be flatter (higher) than A at 100 Hz: C={c_100:.2}, A={a_100:.2}"
        );
    }

    #[test]
    fn test_c_weighting_near_zero_at_1khz() {
        let norm = LoudnessNormalization::new();
        let c_1k = norm.c_weighting(1000.0);
        assert!(
            c_1k.abs() < 0.5,
            "C-weighting at 1 kHz should be ~0 dB, got {c_1k:.2}"
        );
    }

    // --- Specific loudness ---------------------------------------------------

    #[test]
    fn test_specific_loudness_non_negative() {
        let model = ZwickerLoudnessModel::new();
        for exc in [-10.0, 0.0, 10.0, 40.0, 60.0, 80.0] {
            let sl = model.specific_loudness(exc, 4.0);
            assert!(
                sl >= 0.0,
                "Specific loudness must be non-negative, got {sl} for excitation {exc} dB"
            );
        }
    }

    #[test]
    fn test_specific_loudness_increases_with_level() {
        let model = ZwickerLoudnessModel::new();
        let sl_40 = model.specific_loudness(40.0, 4.0);
        let sl_60 = model.specific_loudness(60.0, 4.0);
        let sl_80 = model.specific_loudness(80.0, 4.0);
        assert!(
            sl_60 > sl_40,
            "Specific loudness should increase: SL(40)={sl_40:.4}, SL(60)={sl_60:.4}"
        );
        assert!(
            sl_80 > sl_60,
            "Specific loudness should increase: SL(60)={sl_60:.4}, SL(80)={sl_80:.4}"
        );
    }

    // --- Total loudness ------------------------------------------------------

    #[test]
    fn test_total_loudness_increases_with_level() {
        let model = ZwickerLoudnessModel::new();

        let low: [f64; NUM_BARK_BANDS] = [40.0; NUM_BARK_BANDS];
        let high: [f64; NUM_BARK_BANDS] = [70.0; NUM_BARK_BANDS];

        let result_low = model.compute_loudness_from_spectrum(&low);
        let result_high = model.compute_loudness_from_spectrum(&high);

        assert!(
            result_high.total_loudness_sone > result_low.total_loudness_sone,
            "Louder spectrum must have higher total loudness: low={:.2}, high={:.2}",
            result_low.total_loudness_sone,
            result_high.total_loudness_sone
        );
    }

    #[test]
    fn test_total_loudness_silent_spectrum() {
        let model = ZwickerLoudnessModel::new();
        let silent: [f64; NUM_BARK_BANDS] = [-40.0; NUM_BARK_BANDS];
        let result = model.compute_loudness_from_spectrum(&silent);
        assert!(
            result.total_loudness_sone < 0.01,
            "Silent spectrum should produce ~0 sone, got {:.4}",
            result.total_loudness_sone
        );
    }

    // --- Energy helpers ------------------------------------------------------

    #[test]
    fn test_energy_sum_db() {
        // Two equal sources: 10·log10(2) ≈ 3 dB increase
        let sum = energy_sum_db(&[60.0, 60.0]);
        assert!(
            (sum - 63.01).abs() < 0.1,
            "Sum of two 60 dB sources should be ~63 dB, got {sum:.2}"
        );
    }

    #[test]
    fn test_db_intensity_roundtrip() {
        for &db in &[-20.0, 0.0, 40.0, 80.0, 100.0] {
            let intensity = db_to_intensity(db);
            let recovered = intensity_to_db(intensity);
            assert!(
                (recovered - db).abs() < EPSILON,
                "dB round-trip failed for {db}: got {recovered}"
            );
        }
    }

    // --- Equal-loudness contour ----------------------------------------------

    #[test]
    fn test_hearing_threshold_shape() {
        let contour = EqualLoudnessContour::new();
        let thr_100 = contour.hearing_threshold(100.0);
        let thr_1k = contour.hearing_threshold(1000.0);
        let thr_4k = contour.hearing_threshold(4000.0);
        // Threshold should be higher at 100 Hz than at 1 kHz
        assert!(thr_100 > thr_1k, "100 Hz threshold should exceed 1 kHz");
        // 4 kHz region has lower threshold than 1 kHz in ISO 226
        assert!(thr_4k < thr_1k, "4 kHz threshold should be below 1 kHz");
    }

    // --- LoudnessEqualizer ---------------------------------------------------

    #[test]
    fn test_equalizer_returns_correct_length() {
        let eq = LoudnessEqualizer::new();
        let levels = vec![60.0; 10];
        let freqs: Vec<f64> = (0..10).map(|i| 100.0 * (i + 1) as f64).collect();
        let result = eq.equalize_loudness(&levels, &freqs);
        assert_eq!(result.len(), 10);
    }

    // --- Weighted SPL --------------------------------------------------------

    #[test]
    fn test_weighted_spl_z_equals_unweighted() {
        let norm = LoudnessNormalization::new();
        let spectrum: Vec<(f64, f64)> = vec![(500.0, 70.0), (1000.0, 70.0), (2000.0, 70.0)];
        let z_spl = norm.weighted_spl(&spectrum, WeightingType::ZWeighting);
        let unweighted = energy_sum_db(&[70.0, 70.0, 70.0]);
        assert!(
            (z_spl - unweighted).abs() < 0.1,
            "Z-weighted SPL should equal unweighted: Z={z_spl:.2}, raw={unweighted:.2}"
        );
    }

    // --- Spectrum to Bark bands ----------------------------------------------

    #[test]
    fn test_spectrum_to_bark_bands_basic() {
        let spectrum: Vec<(f64, f64)> = vec![(1000.0, 60.0), (2000.0, 65.0), (4000.0, 55.0)];
        let bands = spectrum_to_bark_bands(&spectrum);
        // There should be exactly NUM_BARK_BANDS entries
        assert_eq!(bands.len(), NUM_BARK_BANDS);
        // Most bands should be silence (-200 dB) since only 3 frequencies contribute
        let non_silent = bands.iter().filter(|&&b| b > -100.0).count();
        assert!(
            non_silent >= 1 && non_silent <= 5,
            "Expected a few non-silent bands, got {non_silent}"
        );
    }
}
