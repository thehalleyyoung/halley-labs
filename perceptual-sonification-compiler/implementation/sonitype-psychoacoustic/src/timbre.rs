//! Timbre perception models for psychoacoustic analysis.
//!
//! Implements spectral moment extraction, ADSR envelope detection,
//! Plomp–Levelt roughness/dissonance modelling, Zwicker sharpness,
//! and perceptually-weighted timbre-space distance metrics.

use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// TimbreDescriptor
// ---------------------------------------------------------------------------

/// Multi-dimensional descriptor capturing the perceptual timbre of a sound.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TimbreDescriptor {
    /// Spectral centroid in Hz (brightness-correlated "center of mass").
    pub spectral_centroid: f64,
    /// Spectral spread in Hz (standard deviation around the centroid).
    pub spectral_spread: f64,
    /// Spectral skewness (asymmetry of the spectral distribution).
    pub spectral_skewness: f64,
    /// Spectral kurtosis (peakedness / tail weight of the distribution).
    pub spectral_kurtosis: f64,
    /// Spectral flux (frame-to-frame spectral change).
    pub spectral_flux: f64,
    /// Spectral roll-off frequency: below this frequency 85 % of energy lies.
    pub spectral_rolloff: f64,
    /// Perceptual brightness, normalised 0–1.
    pub brightness: f64,
    /// Perceptual warmth, normalised 0–1.
    pub warmth: f64,
    /// Zwicker sharpness in acum.
    pub sharpness: f64,
    /// Roughness in asper (Plomp–Levelt pairwise model).
    pub roughness: f64,
    /// Attack time in milliseconds (10 %→90 % of peak).
    pub attack_time_ms: f64,
    /// Decay time in milliseconds (peak→sustain level).
    pub decay_time_ms: f64,
    /// Sustain level, normalised 0–1.
    pub sustain_level: f64,
    /// Release time in milliseconds (note-off→10 % of sustain).
    pub release_time_ms: f64,
}

impl Default for TimbreDescriptor {
    fn default() -> Self {
        Self {
            spectral_centroid: 0.0,
            spectral_spread: 0.0,
            spectral_skewness: 0.0,
            spectral_kurtosis: 0.0,
            spectral_flux: 0.0,
            spectral_rolloff: 0.0,
            brightness: 0.0,
            warmth: 0.0,
            sharpness: 0.0,
            roughness: 0.0,
            attack_time_ms: 0.0,
            decay_time_ms: 0.0,
            sustain_level: 0.0,
            release_time_ms: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// AdsrParams
// ---------------------------------------------------------------------------

/// Attack / Decay / Sustain / Release parameters extracted from an envelope.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AdsrParams {
    pub attack_ms: f64,
    pub decay_ms: f64,
    pub sustain_level: f64,
    pub release_ms: f64,
}

// ---------------------------------------------------------------------------
// TimbreWeights
// ---------------------------------------------------------------------------

/// Per-dimension weights used by [`TimbreDistance::weighted_distance`].
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TimbreWeights {
    pub centroid_weight: f64,
    pub spread_weight: f64,
    pub skewness_weight: f64,
    pub kurtosis_weight: f64,
    pub flux_weight: f64,
    pub rolloff_weight: f64,
    pub brightness_weight: f64,
    pub warmth_weight: f64,
    pub sharpness_weight: f64,
    pub roughness_weight: f64,
    pub attack_weight: f64,
    pub decay_weight: f64,
    pub sustain_weight: f64,
    pub release_weight: f64,
}

impl TimbreWeights {
    /// Default perceptual weighting – emphasises the dimensions that
    /// listeners rely on most when judging timbral similarity (Grey 1977,
    /// McAdams 1995).
    pub fn default_perceptual() -> Self {
        Self {
            centroid_weight: 1.0,
            spread_weight: 0.6,
            skewness_weight: 0.3,
            kurtosis_weight: 0.2,
            flux_weight: 0.5,
            rolloff_weight: 0.4,
            brightness_weight: 0.9,
            warmth_weight: 0.7,
            sharpness_weight: 0.8,
            roughness_weight: 0.7,
            attack_weight: 0.85,
            decay_weight: 0.4,
            sustain_weight: 0.3,
            release_weight: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// TimbreSpace  –  spectral moments + envelope + roughness
// ---------------------------------------------------------------------------

/// Core analysis engine for spectral-moment computation, ADSR detection
/// and Plomp–Levelt roughness modelling.
#[derive(Debug, Clone)]
pub struct TimbreSpace;

impl TimbreSpace {
    pub fn new() -> Self {
        Self
    }

    // -- Spectral moments --------------------------------------------------

    /// Spectral centroid: Σ(f_i · |X_i|) / Σ(|X_i|).
    pub fn spectral_centroid(&self, magnitudes: &[f64], frequencies: &[f64]) -> f64 {
        assert_eq!(magnitudes.len(), frequencies.len(), "length mismatch");
        let total_mag: f64 = magnitudes.iter().copied().sum();
        if total_mag <= f64::EPSILON {
            return 0.0;
        }
        let weighted: f64 = magnitudes
            .iter()
            .zip(frequencies.iter())
            .map(|(&m, &f)| f * m)
            .sum();
        weighted / total_mag
    }

    /// Spectral spread (standard deviation around the centroid):
    /// sqrt(Σ((f_i − centroid)² · |X_i|) / Σ(|X_i|)).
    pub fn spectral_spread(
        &self,
        magnitudes: &[f64],
        frequencies: &[f64],
        centroid: f64,
    ) -> f64 {
        assert_eq!(magnitudes.len(), frequencies.len(), "length mismatch");
        let total_mag: f64 = magnitudes.iter().copied().sum();
        if total_mag <= f64::EPSILON {
            return 0.0;
        }
        let weighted_var: f64 = magnitudes
            .iter()
            .zip(frequencies.iter())
            .map(|(&m, &f)| {
                let diff = f - centroid;
                diff * diff * m
            })
            .sum();
        (weighted_var / total_mag).sqrt()
    }

    /// Spectral skewness (third standardised moment of the spectral
    /// distribution).
    pub fn spectral_skewness(
        &self,
        magnitudes: &[f64],
        frequencies: &[f64],
        centroid: f64,
        spread: f64,
    ) -> f64 {
        if spread <= f64::EPSILON {
            return 0.0;
        }
        let total_mag: f64 = magnitudes.iter().copied().sum();
        if total_mag <= f64::EPSILON {
            return 0.0;
        }
        let m3: f64 = magnitudes
            .iter()
            .zip(frequencies.iter())
            .map(|(&m, &f)| {
                let d = f - centroid;
                d * d * d * m
            })
            .sum();
        (m3 / total_mag) / (spread * spread * spread)
    }

    /// Spectral kurtosis (fourth standardised moment).
    pub fn spectral_kurtosis(
        &self,
        magnitudes: &[f64],
        frequencies: &[f64],
        centroid: f64,
        spread: f64,
    ) -> f64 {
        if spread <= f64::EPSILON {
            return 0.0;
        }
        let total_mag: f64 = magnitudes.iter().copied().sum();
        if total_mag <= f64::EPSILON {
            return 0.0;
        }
        let m4: f64 = magnitudes
            .iter()
            .zip(frequencies.iter())
            .map(|(&m, &f)| {
                let d = f - centroid;
                d * d * d * d * m
            })
            .sum();
        let spread4 = spread * spread * spread * spread;
        (m4 / total_mag) / spread4
    }

    /// Spectral flux between two consecutive frames:
    /// sqrt(Σ(|X_curr_i − X_prev_i|²)).
    pub fn spectral_flux(
        &self,
        current_magnitudes: &[f64],
        previous_magnitudes: &[f64],
    ) -> f64 {
        assert_eq!(
            current_magnitudes.len(),
            previous_magnitudes.len(),
            "length mismatch"
        );
        let sum_sq: f64 = current_magnitudes
            .iter()
            .zip(previous_magnitudes.iter())
            .map(|(&c, &p)| {
                let d = c - p;
                d * d
            })
            .sum();
        sum_sq.sqrt()
    }

    /// Spectral roll-off: lowest frequency below which `percentile`
    /// fraction (0–1) of the total spectral energy lies.
    pub fn spectral_rolloff(
        &self,
        magnitudes: &[f64],
        frequencies: &[f64],
        percentile: f64,
    ) -> f64 {
        assert_eq!(magnitudes.len(), frequencies.len(), "length mismatch");
        let total_energy: f64 = magnitudes.iter().map(|m| m * m).sum();
        if total_energy <= f64::EPSILON {
            return 0.0;
        }
        let threshold = percentile * total_energy;
        let mut cumulative = 0.0;
        for (&m, &f) in magnitudes.iter().zip(frequencies.iter()) {
            cumulative += m * m;
            if cumulative >= threshold {
                return f;
            }
        }
        *frequencies.last().unwrap_or(&0.0)
    }

    // -- Temporal envelope / ADSR ------------------------------------------

    /// Attack time: duration (ms) from 10 % to 90 % of the envelope peak.
    pub fn detect_attack_time(&self, envelope: &[f64], sample_rate: f64) -> f64 {
        if envelope.is_empty() || sample_rate <= 0.0 {
            return 0.0;
        }
        let peak = envelope
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if peak <= f64::EPSILON {
            return 0.0;
        }
        let lo = 0.10 * peak;
        let hi = 0.90 * peak;

        let start_idx = envelope.iter().position(|&v| v >= lo).unwrap_or(0);
        let end_idx = envelope.iter().position(|&v| v >= hi).unwrap_or(start_idx);

        let samples = if end_idx > start_idx {
            end_idx - start_idx
        } else {
            0
        };
        (samples as f64 / sample_rate) * 1000.0
    }

    /// Decay time: duration (ms) from the peak sample to the point where
    /// the envelope first falls to `sustain_level` (absolute, 0–1 of peak).
    pub fn detect_decay_time(
        &self,
        envelope: &[f64],
        sample_rate: f64,
        sustain_level: f64,
    ) -> f64 {
        if envelope.is_empty() || sample_rate <= 0.0 {
            return 0.0;
        }
        let peak = envelope
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if peak <= f64::EPSILON {
            return 0.0;
        }

        let peak_idx = envelope
            .iter()
            .position(|&v| (v - peak).abs() < f64::EPSILON)
            .unwrap_or(0);

        let target = sustain_level * peak;
        let decay_end = envelope[peak_idx..]
            .iter()
            .position(|&v| v <= target)
            .map(|i| i + peak_idx)
            .unwrap_or(envelope.len() - 1);

        let samples = if decay_end > peak_idx {
            decay_end - peak_idx
        } else {
            0
        };
        (samples as f64 / sample_rate) * 1000.0
    }

    /// Sustain level: average amplitude in the middle 40 %–70 % of the
    /// envelope, normalised to peak.  Returns 0–1.
    pub fn detect_sustain_level(&self, envelope: &[f64]) -> f64 {
        if envelope.is_empty() {
            return 0.0;
        }
        let peak = envelope
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if peak <= f64::EPSILON {
            return 0.0;
        }
        let n = envelope.len();
        let start = (0.40 * n as f64) as usize;
        let end = (0.70 * n as f64) as usize;
        if end <= start {
            return 0.0;
        }
        let region = &envelope[start..end];
        let avg: f64 = region.iter().copied().sum::<f64>() / region.len() as f64;
        (avg / peak).clamp(0.0, 1.0)
    }

    /// Release time: duration (ms) from the last sample above
    /// `sustain_level * peak` to the point where the envelope falls
    /// below 10 % of `sustain_level * peak`.
    pub fn detect_release_time(&self, envelope: &[f64], sample_rate: f64) -> f64 {
        if envelope.is_empty() || sample_rate <= 0.0 {
            return 0.0;
        }
        let peak = envelope
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if peak <= f64::EPSILON {
            return 0.0;
        }

        let sustain = self.detect_sustain_level(envelope) * peak;
        let release_threshold = 0.10 * sustain;

        // Walk backwards to find the last sample above sustain level.
        let note_off = envelope
            .iter()
            .rposition(|&v| v >= sustain)
            .unwrap_or(envelope.len().saturating_sub(1));

        let release_end = envelope[note_off..]
            .iter()
            .position(|&v| v <= release_threshold)
            .map(|i| i + note_off)
            .unwrap_or(envelope.len() - 1);

        let samples = if release_end > note_off {
            release_end - note_off
        } else {
            0
        };
        (samples as f64 / sample_rate) * 1000.0
    }

    /// Convenience: compute all four ADSR parameters at once.
    pub fn adsr_parameters(&self, envelope: &[f64], sample_rate: f64) -> AdsrParams {
        let sustain_level = self.detect_sustain_level(envelope);
        AdsrParams {
            attack_ms: self.detect_attack_time(envelope, sample_rate),
            decay_ms: self.detect_decay_time(envelope, sample_rate, sustain_level),
            sustain_level,
            release_ms: self.detect_release_time(envelope, sample_rate),
        }
    }

    // -- Roughness / dissonance (Plomp–Levelt) -----------------------------

    /// Plomp–Levelt dissonance between two pure tones at frequencies
    /// `f1`, `f2` with amplitudes `a1`, `a2`.
    ///
    /// d = a_min · (e^(−3.5·s·Δf) − e^(−5.75·s·Δf))
    /// where s = 0.24 / (0.021·f_min + 19), Δf = |f2 − f1|.
    pub fn plomp_levelt_dissonance(&self, f1: f64, f2: f64, a1: f64, a2: f64) -> f64 {
        let delta = (f2 - f1).abs();
        if delta < f64::EPSILON {
            return 0.0;
        }
        let f_min = f1.min(f2);
        let a_min = a1.min(a2);
        let s = 0.24 / (0.021 * f_min + 19.0);
        let d = a_min * ((-3.5 * s * delta).exp() - (-5.75 * s * delta).exp());
        d.max(0.0)
    }

    /// Total roughness: sum of pairwise Plomp–Levelt dissonance across
    /// all unique pairs of partials.
    pub fn total_roughness(&self, frequencies: &[f64], amplitudes: &[f64]) -> f64 {
        assert_eq!(frequencies.len(), amplitudes.len(), "length mismatch");
        let n = frequencies.len();
        let mut total = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                total += self.plomp_levelt_dissonance(
                    frequencies[i],
                    frequencies[j],
                    amplitudes[i],
                    amplitudes[j],
                );
            }
        }
        total
    }

    /// Dissonance curve: evaluate the Plomp–Levelt dissonance of a
    /// reference tone `f_ref` (with amplitude 1) against a second tone
    /// swept through frequency ratios from `ratio_min` to `ratio_max`.
    ///
    /// Returns `(ratio, dissonance)` pairs.
    pub fn dissonance_curve(
        &self,
        f_ref: f64,
        ratio_min: f64,
        ratio_max: f64,
        steps: usize,
    ) -> Vec<(f64, f64)> {
        if steps == 0 {
            return Vec::new();
        }
        let step_size = (ratio_max - ratio_min) / steps as f64;
        (0..=steps)
            .map(|i| {
                let ratio = ratio_min + i as f64 * step_size;
                let f2 = f_ref * ratio;
                let d = self.plomp_levelt_dissonance(f_ref, f2, 1.0, 1.0);
                (ratio, d)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// TimbreFeatureExtractor
// ---------------------------------------------------------------------------

/// High-level feature extractor that combines spectral-moment analysis,
/// temporal-envelope analysis and psychoacoustic metrics into a single
/// [`TimbreDescriptor`].
pub struct TimbreFeatureExtractor {
    space: TimbreSpace,
}

impl TimbreFeatureExtractor {
    pub fn new() -> Self {
        Self {
            space: TimbreSpace::new(),
        }
    }

    /// Extract a complete [`TimbreDescriptor`] from a single analysis frame.
    ///
    /// * `magnitudes` / `frequencies` – paired FFT bin magnitudes and
    ///   centre frequencies.
    /// * `envelope` – optional amplitude envelope of the full note (used
    ///   for ADSR detection).
    /// * `sample_rate` – audio sample rate in Hz.
    pub fn extract(
        &self,
        magnitudes: &[f64],
        frequencies: &[f64],
        envelope: Option<&[f64]>,
        sample_rate: f64,
    ) -> TimbreDescriptor {
        let centroid = self.space.spectral_centroid(magnitudes, frequencies);
        let spread = self.space.spectral_spread(magnitudes, frequencies, centroid);
        let skewness =
            self.space
                .spectral_skewness(magnitudes, frequencies, centroid, spread);
        let kurtosis =
            self.space
                .spectral_kurtosis(magnitudes, frequencies, centroid, spread);
        let rolloff = self.space.spectral_rolloff(magnitudes, frequencies, 0.85);

        let brightness = self.brightness(magnitudes, frequencies, 3000.0);
        let warmth = self.warmth(magnitudes, frequencies);

        // Roughness from the same spectral data (treat bins as partials).
        let roughness = self.space.total_roughness(frequencies, magnitudes);

        // Approximate specific-loudness from magnitudes for Zwicker sharpness.
        let specific_loudness: Vec<f64> = magnitudes.iter().map(|&m| m.max(0.0)).collect();
        let sharpness = self.sharpness_zwicker(&specific_loudness);

        let (attack_ms, decay_ms, sustain_level, release_ms) = match envelope {
            Some(env) => {
                let adsr = self.space.adsr_parameters(env, sample_rate);
                (adsr.attack_ms, adsr.decay_ms, adsr.sustain_level, adsr.release_ms)
            }
            None => (0.0, 0.0, 0.0, 0.0),
        };

        TimbreDescriptor {
            spectral_centroid: centroid,
            spectral_spread: spread,
            spectral_skewness: skewness,
            spectral_kurtosis: kurtosis,
            spectral_flux: 0.0, // needs previous frame
            spectral_rolloff: rolloff,
            brightness,
            warmth,
            sharpness,
            roughness,
            attack_time_ms: attack_ms,
            decay_time_ms: decay_ms,
            sustain_level,
            release_time_ms: release_ms,
        }
    }

    /// Brightness: ratio of spectral energy above `cutoff_hz` to the
    /// total energy.  Returns 0–1.
    pub fn brightness(
        &self,
        magnitudes: &[f64],
        frequencies: &[f64],
        cutoff_hz: f64,
    ) -> f64 {
        let total: f64 = magnitudes.iter().map(|m| m * m).sum();
        if total <= f64::EPSILON {
            return 0.0;
        }
        let above: f64 = magnitudes
            .iter()
            .zip(frequencies.iter())
            .filter(|(_, &f)| f >= cutoff_hz)
            .map(|(&m, _)| m * m)
            .sum();
        (above / total).clamp(0.0, 1.0)
    }

    /// Warmth: ratio of spectral energy in the 100–500 Hz band to total
    /// energy.  Returns 0–1.
    pub fn warmth(
        &self,
        magnitudes: &[f64],
        frequencies: &[f64],
    ) -> f64 {
        let total: f64 = magnitudes.iter().map(|m| m * m).sum();
        if total <= f64::EPSILON {
            return 0.0;
        }
        let warm_band: f64 = magnitudes
            .iter()
            .zip(frequencies.iter())
            .filter(|(_, &f)| (100.0..=500.0).contains(&f))
            .map(|(&m, _)| m * m)
            .sum();
        (warm_band / total).clamp(0.0, 1.0)
    }

    /// Zwicker sharpness (simplified): weighted first moment of the
    /// specific-loudness distribution along critical bands.
    ///
    /// S = 0.11 · Σ(N'(z) · g(z) · z · Δz) / Σ(N'(z) · Δz)   [acum]
    ///
    /// where g(z) ≈ 1 for z ≤ 16 Bark and rises sharply above.
    pub fn sharpness_zwicker(&self, specific_loudness: &[f64]) -> f64 {
        let n = specific_loudness.len();
        if n == 0 {
            return 0.0;
        }
        // Map bins linearly to a 0–24 Bark scale for weighting.
        let dz = 24.0 / n as f64;
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (i, &nl) in specific_loudness.iter().enumerate() {
            let z = (i as f64 + 0.5) * dz; // centre of bin in Bark
            let g = if z <= 16.0 {
                1.0
            } else {
                // Quadratic weighting above 16 Bark (Zwicker & Fastl 1999).
                0.066 * (z / 16.0_f64).exp() / (z / 16.0)
            };
            numerator += nl * g * z * dz;
            denominator += nl * dz;
        }
        if denominator <= f64::EPSILON {
            return 0.0;
        }
        0.11 * numerator / denominator
    }
}

// ---------------------------------------------------------------------------
// TimbreDistance
// ---------------------------------------------------------------------------

/// Metrics for comparing [`TimbreDescriptor`]s in a perceptual timbre
/// space.
#[derive(Debug, Clone)]
pub struct TimbreDistance {
    /// Normalisation ranges for each dimension (approximate).  Using
    /// fixed representative ranges allows Euclidean distance to be
    /// meaningful without per-dataset standardisation.
    norm_centroid: f64,
    norm_spread: f64,
    norm_skewness: f64,
    norm_kurtosis: f64,
    norm_flux: f64,
    norm_rolloff: f64,
    norm_brightness: f64,
    norm_warmth: f64,
    norm_sharpness: f64,
    norm_roughness: f64,
    norm_attack: f64,
    norm_decay: f64,
    norm_sustain: f64,
    norm_release: f64,
}

impl TimbreDistance {
    pub fn new() -> Self {
        Self {
            norm_centroid: 8000.0,
            norm_spread: 4000.0,
            norm_skewness: 4.0,
            norm_kurtosis: 20.0,
            norm_flux: 10.0,
            norm_rolloff: 16000.0,
            norm_brightness: 1.0,
            norm_warmth: 1.0,
            norm_sharpness: 4.0,
            norm_roughness: 1.0,
            norm_attack: 500.0,
            norm_decay: 1000.0,
            norm_sustain: 1.0,
            norm_release: 2000.0,
        }
    }

    /// Normalised Euclidean distance between two descriptors.
    pub fn euclidean(&self, a: &TimbreDescriptor, b: &TimbreDescriptor) -> f64 {
        let diffs = self.normalised_diffs(a, b);
        diffs.iter().map(|d| d * d).sum::<f64>().sqrt()
    }

    /// Perceptually-weighted distance (each dimension scaled by its
    /// weight before summation).
    pub fn weighted_distance(
        &self,
        a: &TimbreDescriptor,
        b: &TimbreDescriptor,
        weights: &TimbreWeights,
    ) -> f64 {
        let diffs = self.normalised_diffs(a, b);
        let w = [
            weights.centroid_weight,
            weights.spread_weight,
            weights.skewness_weight,
            weights.kurtosis_weight,
            weights.flux_weight,
            weights.rolloff_weight,
            weights.brightness_weight,
            weights.warmth_weight,
            weights.sharpness_weight,
            weights.roughness_weight,
            weights.attack_weight,
            weights.decay_weight,
            weights.sustain_weight,
            weights.release_weight,
        ];
        diffs
            .iter()
            .zip(w.iter())
            .map(|(d, wt)| wt * d * d)
            .sum::<f64>()
            .sqrt()
    }

    /// Similarity metric: 1 / (1 + euclidean_distance).  Returns 0–1
    /// where 1 means identical.
    pub fn similarity(&self, a: &TimbreDescriptor, b: &TimbreDescriptor) -> f64 {
        1.0 / (1.0 + self.euclidean(a, b))
    }

    /// Pairwise similarity matrix for a set of descriptors.
    pub fn similarity_matrix(&self, descriptors: &[TimbreDescriptor]) -> Vec<Vec<f64>> {
        let n = descriptors.len();
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n {
                let s = self.similarity(&descriptors[i], &descriptors[j]);
                matrix[i][j] = s;
                matrix[j][i] = s;
            }
        }
        matrix
    }

    // -- internal helpers --------------------------------------------------

    fn normalised_diffs(&self, a: &TimbreDescriptor, b: &TimbreDescriptor) -> [f64; 14] {
        [
            (a.spectral_centroid - b.spectral_centroid) / self.norm_centroid,
            (a.spectral_spread - b.spectral_spread) / self.norm_spread,
            (a.spectral_skewness - b.spectral_skewness) / self.norm_skewness,
            (a.spectral_kurtosis - b.spectral_kurtosis) / self.norm_kurtosis,
            (a.spectral_flux - b.spectral_flux) / self.norm_flux,
            (a.spectral_rolloff - b.spectral_rolloff) / self.norm_rolloff,
            (a.brightness - b.brightness) / self.norm_brightness,
            (a.warmth - b.warmth) / self.norm_warmth,
            (a.sharpness - b.sharpness) / self.norm_sharpness,
            (a.roughness - b.roughness) / self.norm_roughness,
            (a.attack_time_ms - b.attack_time_ms) / self.norm_attack,
            (a.decay_time_ms - b.decay_time_ms) / self.norm_decay,
            (a.sustain_level - b.sustain_level) / self.norm_sustain,
            (a.release_time_ms - b.release_time_ms) / self.norm_release,
        ]
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -- Spectral centroid -------------------------------------------------

    #[test]
    fn centroid_of_single_peak_equals_peak_frequency() {
        let ts = TimbreSpace::new();
        let freqs = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let mags = vec![0.0, 0.0, 1.0, 0.0, 0.0]; // peak at 300 Hz
        let c = ts.spectral_centroid(&mags, &freqs);
        assert!(
            approx_eq(c, 300.0, 1e-9),
            "centroid should be 300 Hz, got {c}"
        );
    }

    #[test]
    fn centroid_of_uniform_spectrum_equals_midpoint() {
        let ts = TimbreSpace::new();
        let n = 5;
        let freqs: Vec<f64> = (1..=n).map(|i| i as f64 * 100.0).collect();
        let mags = vec![1.0; n];
        let c = ts.spectral_centroid(&mags, &freqs);
        let expected = freqs.iter().sum::<f64>() / n as f64; // 300
        assert!(
            approx_eq(c, expected, 1e-9),
            "centroid should be {expected}, got {c}"
        );
    }

    // -- Spectral spread ---------------------------------------------------

    #[test]
    fn spread_of_single_peak_is_near_zero() {
        let ts = TimbreSpace::new();
        let freqs = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let mags = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let c = ts.spectral_centroid(&mags, &freqs);
        let s = ts.spectral_spread(&mags, &freqs, c);
        assert!(
            approx_eq(s, 0.0, 1e-9),
            "spread of single peak should be ~0, got {s}"
        );
    }

    // -- Spectral flux -----------------------------------------------------

    #[test]
    fn flux_of_identical_spectra_is_zero() {
        let ts = TimbreSpace::new();
        let mags = vec![0.5, 0.3, 0.8, 0.1];
        let f = ts.spectral_flux(&mags, &mags);
        assert!(approx_eq(f, 0.0, 1e-12), "flux should be 0, got {f}");
    }

    #[test]
    fn flux_of_different_spectra_is_positive() {
        let ts = TimbreSpace::new();
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let f = ts.spectral_flux(&a, &b);
        assert!(f > 0.0, "flux should be > 0, got {f}");
        assert!(
            approx_eq(f, 2.0, 1e-9),
            "flux should be 2.0, got {f}"
        );
    }

    // -- Plomp–Levelt dissonance -------------------------------------------

    #[test]
    fn plomp_levelt_unison_is_zero() {
        let ts = TimbreSpace::new();
        let d = ts.plomp_levelt_dissonance(440.0, 440.0, 1.0, 1.0);
        assert!(
            approx_eq(d, 0.0, 1e-12),
            "unison dissonance should be 0, got {d}"
        );
    }

    #[test]
    fn plomp_levelt_has_maximum_near_quarter_critical_bandwidth() {
        let ts = TimbreSpace::new();
        // Critical bandwidth at 440 Hz ≈ 0.024·440 + 19 ≈ 29.56 Hz
        // Quarter CB ≈ 7.4 Hz  — but the Plomp–Levelt peak is actually
        // closer to ~1/4 of the CB scaled by s; let's just verify a
        // peak exists between ratio 1.01 and 1.10.
        let curve = ts.dissonance_curve(440.0, 1.0, 1.15, 500);
        let max_d = curve
            .iter()
            .map(|&(_, d)| d)
            .fold(f64::NEG_INFINITY, f64::max);
        let at_max = curve
            .iter()
            .find(|&&(_, d)| (d - max_d).abs() < 1e-12)
            .unwrap()
            .0;
        assert!(
            (1.01..=1.10).contains(&at_max),
            "peak ratio should be in 1.01..1.10, got {at_max}"
        );
    }

    #[test]
    fn plomp_levelt_decreases_for_large_intervals() {
        let ts = TimbreSpace::new();
        let d_semitone = ts.plomp_levelt_dissonance(440.0, 466.16, 1.0, 1.0);
        let d_octave = ts.plomp_levelt_dissonance(440.0, 880.0, 1.0, 1.0);
        assert!(
            d_octave < d_semitone,
            "octave dissonance ({d_octave}) should be less than semitone ({d_semitone})"
        );
    }

    // -- Total roughness ---------------------------------------------------

    #[test]
    fn roughness_of_harmonic_series_is_positive() {
        let ts = TimbreSpace::new();
        let f0 = 200.0;
        let freqs: Vec<f64> = (1..=8).map(|h| h as f64 * f0).collect();
        let amps = vec![1.0; 8];
        let r = ts.total_roughness(&freqs, &amps);
        assert!(r > 0.0, "harmonic-series roughness should be > 0, got {r}");
    }

    // -- TimbreDistance -----------------------------------------------------

    #[test]
    fn distance_between_identical_descriptors_is_zero() {
        let td = TimbreDistance::new();
        let desc = TimbreDescriptor::default();
        let d = td.euclidean(&desc, &desc);
        assert!(
            approx_eq(d, 0.0, 1e-12),
            "distance to self should be 0, got {d}"
        );
    }

    #[test]
    fn distance_is_symmetric() {
        let td = TimbreDistance::new();
        let a = TimbreDescriptor {
            spectral_centroid: 1000.0,
            brightness: 0.8,
            attack_time_ms: 10.0,
            ..TimbreDescriptor::default()
        };
        let b = TimbreDescriptor {
            spectral_centroid: 3000.0,
            brightness: 0.2,
            attack_time_ms: 100.0,
            ..TimbreDescriptor::default()
        };
        let d_ab = td.euclidean(&a, &b);
        let d_ba = td.euclidean(&b, &a);
        assert!(
            approx_eq(d_ab, d_ba, 1e-12),
            "distance should be symmetric: {d_ab} vs {d_ba}"
        );
    }

    // -- Brightness --------------------------------------------------------

    #[test]
    fn brightness_of_low_passed_spectrum_is_low() {
        let ext = TimbreFeatureExtractor::new();
        // All energy below 1000 Hz, cutoff at 3000 Hz.
        let freqs: Vec<f64> = (1..=20).map(|i| i as f64 * 50.0).collect();
        let mags = vec![1.0; 20]; // 50–1000 Hz
        let b = ext.brightness(&mags, &freqs, 3000.0);
        assert!(
            b < 0.01,
            "brightness should be near 0 for low-passed signal, got {b}"
        );
    }

    // -- ADSR detection ----------------------------------------------------

    #[test]
    fn adsr_detection_on_synthetic_envelope() {
        let ts = TimbreSpace::new();
        let sr = 1000.0; // 1 sample = 1 ms for easy reasoning

        // Build a synthetic envelope: attack 50 ms, decay 100 ms to 0.5,
        // sustain 200 ms at 0.5, release 100 ms.
        let mut env: Vec<f64> = Vec::new();

        // Attack: linear ramp 0→1 over 50 samples
        for i in 0..50 {
            env.push(i as f64 / 50.0);
        }
        // Decay: linear 1→0.5 over 100 samples
        for i in 0..100 {
            env.push(1.0 - 0.5 * (i as f64 / 100.0));
        }
        // Sustain: constant 0.5 for 200 samples
        for _ in 0..200 {
            env.push(0.5);
        }
        // Release: linear 0.5→0 over 100 samples
        for i in 0..100 {
            env.push(0.5 * (1.0 - i as f64 / 100.0));
        }

        let attack = ts.detect_attack_time(&env, sr);
        assert!(
            attack > 30.0 && attack < 60.0,
            "attack should be ~40 ms (10%→90%), got {attack}"
        );

        let sustain = ts.detect_sustain_level(&env);
        assert!(
            approx_eq(sustain, 0.5, 0.15),
            "sustain level should be ~0.5, got {sustain}"
        );

        let adsr = ts.adsr_parameters(&env, sr);
        assert!(adsr.attack_ms > 0.0, "attack_ms should be > 0");
        assert!(adsr.decay_ms > 0.0, "decay_ms should be > 0");
        assert!(adsr.sustain_level > 0.0, "sustain_level should be > 0");
    }

    // -- Similarity matrix -------------------------------------------------

    #[test]
    fn similarity_matrix_diagonal_is_one() {
        let td = TimbreDistance::new();
        let descs = vec![
            TimbreDescriptor::default(),
            TimbreDescriptor {
                spectral_centroid: 2000.0,
                brightness: 0.5,
                ..TimbreDescriptor::default()
            },
        ];
        let mat = td.similarity_matrix(&descs);
        assert!(approx_eq(mat[0][0], 1.0, 1e-12));
        assert!(approx_eq(mat[1][1], 1.0, 1e-12));
        assert!(mat[0][1] < 1.0);
        assert!(approx_eq(mat[0][1], mat[1][0], 1e-12));
    }

    // -- Spectral rolloff --------------------------------------------------

    #[test]
    fn rolloff_of_single_bin_equals_that_bin() {
        let ts = TimbreSpace::new();
        let freqs = vec![100.0, 200.0, 300.0];
        let mags = vec![0.0, 1.0, 0.0]; // all energy at 200 Hz
        let ro = ts.spectral_rolloff(&mags, &freqs, 0.85);
        assert!(
            approx_eq(ro, 200.0, 1e-9),
            "rolloff should be 200 Hz, got {ro}"
        );
    }

    // -- Zwicker sharpness -------------------------------------------------

    #[test]
    fn sharpness_increases_with_high_frequency_energy() {
        let ext = TimbreFeatureExtractor::new();
        // Low-frequency specific loudness (concentrated at low Bark).
        let low: Vec<f64> = (0..100)
            .map(|i| if i < 30 { 1.0 } else { 0.0 })
            .collect();
        // High-frequency specific loudness.
        let high: Vec<f64> = (0..100)
            .map(|i| if i >= 70 { 1.0 } else { 0.0 })
            .collect();
        let s_low = ext.sharpness_zwicker(&low);
        let s_high = ext.sharpness_zwicker(&high);
        assert!(
            s_high > s_low,
            "high-freq sharpness ({s_high}) should exceed low-freq ({s_low})"
        );
    }

    // -- Weighted distance -------------------------------------------------

    #[test]
    fn weighted_distance_respects_weights() {
        let td = TimbreDistance::new();
        let a = TimbreDescriptor::default();
        let b = TimbreDescriptor {
            spectral_centroid: 4000.0,
            ..TimbreDescriptor::default()
        };

        // All-zero weights → distance 0.
        let zero_weights = TimbreWeights {
            centroid_weight: 0.0,
            spread_weight: 0.0,
            skewness_weight: 0.0,
            kurtosis_weight: 0.0,
            flux_weight: 0.0,
            rolloff_weight: 0.0,
            brightness_weight: 0.0,
            warmth_weight: 0.0,
            sharpness_weight: 0.0,
            roughness_weight: 0.0,
            attack_weight: 0.0,
            decay_weight: 0.0,
            sustain_weight: 0.0,
            release_weight: 0.0,
        };
        let dw = td.weighted_distance(&a, &b, &zero_weights);
        assert!(
            approx_eq(dw, 0.0, 1e-12),
            "zero weights should give distance 0, got {dw}"
        );

        // Perceptual weights → positive distance.
        let pw = TimbreWeights::default_perceptual();
        let dp = td.weighted_distance(&a, &b, &pw);
        assert!(dp > 0.0, "perceptual weights should give positive distance");
    }

    // -- Feature extractor full pipeline -----------------------------------

    #[test]
    fn extractor_produces_valid_descriptor() {
        let ext = TimbreFeatureExtractor::new();
        let freqs: Vec<f64> = (1..=512).map(|i| i as f64 * 10.0).collect();
        let mags: Vec<f64> = freqs.iter().map(|&f| 1.0 / (1.0 + f / 1000.0)).collect();
        let desc = ext.extract(&mags, &freqs, None, 44100.0);
        assert!(desc.spectral_centroid > 0.0);
        assert!(desc.spectral_spread > 0.0);
        assert!(desc.brightness >= 0.0 && desc.brightness <= 1.0);
        assert!(desc.warmth >= 0.0 && desc.warmth <= 1.0);
    }
}
