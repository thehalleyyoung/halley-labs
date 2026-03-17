//! Pitch perception models for psychoacoustic analysis.
//!
//! Implements place theory (spectral), temporal theory (autocorrelation),
//! virtual pitch (missing fundamental), pitch salience, scale conversion,
//! and pitch contour analysis.

use serde::{Serialize, Deserialize};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Scale-conversion helpers
// ---------------------------------------------------------------------------

/// Convert frequency in Hz to Bark scale (Traunmüller 1990).
pub fn hz_to_bark(f: f64) -> f64 {
    26.81 * f / (1960.0 + f) - 0.53
}

/// Convert Bark scale value back to Hz.
pub fn bark_to_hz(z: f64) -> f64 {
    1960.0 * (z + 0.53) / (26.81 - z - 0.53)
}

/// Convert frequency in Hz to Mel scale (O'Shaughnessy 1987).
pub fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

/// Convert Mel value back to Hz.
pub fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10.0_f64.powf(m / 2595.0) - 1.0)
}

/// Convert frequency in Hz to MIDI note number (fractional).
pub fn hz_to_midi(f: f64) -> f64 {
    69.0 + 12.0 * (f / 440.0).log2()
}

/// Convert MIDI note number (fractional) to Hz.
pub fn midi_to_hz(midi: f64) -> f64 {
    440.0 * 2.0_f64.powf((midi - 69.0) / 12.0)
}

/// Cents distance from `ref_hz` to `f`.
pub fn hz_to_cents(f: f64, ref_hz: f64) -> f64 {
    1200.0 * (f / ref_hz).log2()
}

// ---------------------------------------------------------------------------
// PitchModel — core pitch-extraction algorithms
// ---------------------------------------------------------------------------

/// Core pitch-perception model combining spectral, temporal, and
/// virtual-pitch methods.
#[derive(Debug, Clone)]
pub struct PitchModel {
    /// Minimum plausible pitch (Hz) for peak search.
    pub min_freq: f64,
    /// Maximum plausible pitch (Hz) for peak search.
    pub max_freq: f64,
    /// Number of sub-harmonics to test when resolving the fundamental.
    pub max_subharmonic_order: usize,
    /// Relative threshold (0–1) for autocorrelation peak detection.
    pub autocorrelation_threshold: f64,
    /// Maximum harmonic number to consider in subharmonic summation.
    pub max_harmonic_number: usize,
    /// Tolerance in cents for matching a partial to a harmonic.
    pub harmonic_tolerance_cents: f64,
}

impl PitchModel {
    pub fn new() -> Self {
        Self {
            min_freq: 50.0,
            max_freq: 5000.0,
            max_subharmonic_order: 5,
            autocorrelation_threshold: 0.3,
            max_harmonic_number: 10,
            harmonic_tolerance_cents: 50.0,
        }
    }

    // ---- Place theory (spectral peak-picking) ----------------------------

    /// Estimate pitch from a magnitude spectrum using spectral peak picking
    /// with sub-harmonic disambiguation.
    ///
    /// `spectrum_magnitudes` and `freq_bins` must have the same length.
    /// Returns `None` if no valid peak is found in the plausible range.
    pub fn spectral_pitch(
        &self,
        spectrum_magnitudes: &[f64],
        freq_bins: &[f64],
    ) -> Option<f64> {
        if spectrum_magnitudes.len() != freq_bins.len() || spectrum_magnitudes.is_empty() {
            return None;
        }

        // Collect local peaks inside the plausible range.
        let mut peaks: Vec<(usize, f64, f64)> = Vec::new(); // (index, freq, mag)
        let n = spectrum_magnitudes.len();
        for i in 1..n.saturating_sub(1) {
            let f = freq_bins[i];
            if f < self.min_freq || f > self.max_freq {
                continue;
            }
            let m = spectrum_magnitudes[i];
            if m > spectrum_magnitudes[i - 1] && m > spectrum_magnitudes[i + 1] {
                peaks.push((i, f, m));
            }
        }

        if peaks.is_empty() {
            return None;
        }

        // Sort by magnitude descending.
        peaks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let dominant_freq = peaks[0].1;
        let dominant_mag = peaks[0].2;

        // Check whether the dominant peak is actually a harmonic of a lower
        // peak.  For each sub-harmonic order k = 2..max, look for a peak near
        // dominant_freq / k.
        for k in 2..=self.max_subharmonic_order {
            let candidate = dominant_freq / k as f64;
            if candidate < self.min_freq {
                break;
            }
            for &(_idx, pf, pm) in &peaks {
                let cents_diff = (hz_to_cents(pf, candidate)).abs();
                if cents_diff < self.harmonic_tolerance_cents && pm > dominant_mag * 0.15 {
                    return Some(pf);
                }
            }
        }

        Some(dominant_freq)
    }

    // ---- Temporal theory (autocorrelation) --------------------------------

    /// Estimate pitch via normalized autocorrelation of the time-domain
    /// signal, with parabolic interpolation for sub-sample accuracy.
    pub fn autocorrelation_pitch(
        &self,
        signal: &[f64],
        sample_rate: f64,
    ) -> Option<f64> {
        let n = signal.len();
        if n < 4 {
            return None;
        }

        // Determine lag range from frequency bounds.
        let min_lag = (sample_rate / self.max_freq).ceil() as usize;
        let max_lag = (sample_rate / self.min_freq).floor() as usize;
        let max_lag = max_lag.min(n / 2);

        if min_lag >= max_lag {
            return None;
        }

        // Compute normalized autocorrelation r[lag].
        let energy: f64 = signal.iter().map(|&s| s * s).sum();
        if energy == 0.0 {
            return None;
        }

        let mut acf = vec![0.0_f64; max_lag + 1];
        for lag in min_lag..=max_lag {
            let mut num = 0.0;
            let mut e1 = 0.0;
            let mut e2 = 0.0;
            for i in 0..(n - lag) {
                num += signal[i] * signal[i + lag];
                e1 += signal[i] * signal[i];
                e2 += signal[i + lag] * signal[i + lag];
            }
            let denom = (e1 * e2).sqrt();
            acf[lag] = if denom > 0.0 { num / denom } else { 0.0 };
        }

        // Find the first peak above threshold after the minimum lag.
        let mut best_lag: Option<usize> = None;
        let mut best_val: f64 = self.autocorrelation_threshold;

        for lag in (min_lag + 1)..max_lag {
            if acf[lag] > acf[lag - 1] && acf[lag] > acf[lag + 1] && acf[lag] > best_val {
                best_val = acf[lag];
                best_lag = Some(lag);
                break; // take the first significant peak (fundamental period)
            }
        }

        let lag = best_lag?;

        // Parabolic interpolation around the peak for sub-sample accuracy.
        let alpha = acf[lag - 1];
        let beta = acf[lag];
        let gamma = acf[lag + 1];
        let denom = alpha - 2.0 * beta + gamma;
        let correction = if denom.abs() > 1e-12 {
            0.5 * (alpha - gamma) / denom
        } else {
            0.0
        };
        let refined_lag = lag as f64 + correction;

        if refined_lag <= 0.0 {
            return None;
        }

        Some(sample_rate / refined_lag)
    }

    // ---- Virtual pitch (missing fundamental) ------------------------------

    /// Given a set of observed frequencies (partials), find the best-fit
    /// fundamental using subharmonic summation over a default search range.
    pub fn virtual_pitch(&self, harmonics: &[f64]) -> Option<f64> {
        if harmonics.is_empty() {
            return None;
        }

        let min_f0 = self.min_freq;
        // The fundamental cannot be larger than the smallest partial.
        let max_f0 = harmonics
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .min(self.max_freq);

        if min_f0 >= max_f0 {
            // Fall back: try range down to 20 Hz.
            let results = self.subharmonic_summation(harmonics, 20.0, max_f0 + 1.0, 0.5);
            return results
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|&(f, _)| f);
        }

        let results = self.subharmonic_summation(harmonics, min_f0, max_f0, 0.5);
        results
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(f, _)| f)
    }

    /// Subharmonic summation: for every candidate f0 in
    /// `[min_f0, max_f0]` stepped by `resolution` Hz, accumulate weighted
    /// evidence from each observed frequency that falls near a harmonic of
    /// f0.  Returns `(candidate_hz, evidence)` pairs.
    pub fn subharmonic_summation(
        &self,
        frequencies: &[f64],
        min_f0: f64,
        max_f0: f64,
        resolution: f64,
    ) -> Vec<(f64, f64)> {
        let mut results: Vec<(f64, f64)> = Vec::new();
        if resolution <= 0.0 || min_f0 > max_f0 {
            return results;
        }

        let mut f0 = min_f0;
        while f0 <= max_f0 {
            let mut evidence = 0.0_f64;
            for &freq in frequencies {
                // For each harmonic number, check if freq ≈ n * f0.
                for n in 1..=self.max_harmonic_number {
                    let expected = f0 * n as f64;
                    if expected < 1.0 {
                        continue;
                    }
                    let cents_diff = hz_to_cents(freq, expected).abs();
                    if cents_diff < self.harmonic_tolerance_cents {
                        // Weight decreases with harmonic number (lower
                        // harmonics contribute more).
                        let weight = 1.0 / n as f64;
                        // Gaussian weighting by cents deviation.
                        let sigma = self.harmonic_tolerance_cents / 2.0;
                        let gauss = (-0.5 * (cents_diff / sigma).powi(2)).exp();
                        evidence += weight * gauss;
                    }
                }
            }
            results.push((f0, evidence));
            f0 += resolution;
        }
        results
    }

    // ---- Pitch salience ---------------------------------------------------

    /// Compute the perceptual salience (clarity) of a pitch percept at `f0`,
    /// on a 0–1 scale.  Based on the harmonic-to-noise ratio and the number
    /// of resolved harmonics present in the spectrum.
    pub fn pitch_salience(
        &self,
        spectrum_magnitudes: &[f64],
        freq_bins: &[f64],
        f0: f64,
    ) -> f64 {
        if spectrum_magnitudes.len() != freq_bins.len()
            || spectrum_magnitudes.is_empty()
            || f0 <= 0.0
        {
            return 0.0;
        }

        let total_energy: f64 = spectrum_magnitudes.iter().map(|&m| m * m).sum();
        if total_energy == 0.0 {
            return 0.0;
        }

        let mut harmonic_energy = 0.0_f64;
        let mut harmonics_found = 0_usize;
        let tolerance_ratio = 2.0_f64.powf(self.harmonic_tolerance_cents / 1200.0);

        for n in 1..=self.max_harmonic_number {
            let target = f0 * n as f64;
            let lo = target / tolerance_ratio;
            let hi = target * tolerance_ratio;

            // Find the bin with the maximum magnitude inside [lo, hi].
            let mut best_mag = 0.0_f64;
            for (i, &f) in freq_bins.iter().enumerate() {
                if f >= lo && f <= hi && spectrum_magnitudes[i] > best_mag {
                    best_mag = spectrum_magnitudes[i];
                }
            }
            if best_mag > 0.0 {
                harmonic_energy += best_mag * best_mag;
                harmonics_found += 1;
            }
        }

        if harmonics_found == 0 {
            return 0.0;
        }

        let hnr = harmonic_energy / total_energy; // 0..1
        let resolved_ratio = harmonics_found as f64 / self.max_harmonic_number as f64;

        // Combine HNR and resolved-harmonic fraction (geometric mean-ish).
        let salience = (0.7 * hnr + 0.3 * resolved_ratio).clamp(0.0, 1.0);
        salience
    }
}

impl Default for PitchModel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PitchScaleConverter
// ---------------------------------------------------------------------------

/// Utility for converting between pitch scales and quantising to
/// musical tuning systems.
pub struct PitchScaleConverter {
    _private: (),
}

impl PitchScaleConverter {
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Natural-log frequency (useful for linear pitch axes).
    pub fn hz_to_log(&self, f: f64) -> f64 {
        f.ln()
    }

    /// Inverse of `hz_to_log`.
    pub fn log_to_hz(&self, log_f: f64) -> f64 {
        log_f.exp()
    }

    /// Quantise a frequency to the nearest MIDI note.
    /// Returns `(midi_note, cents_deviation)` where cents is in [-50, +50).
    pub fn quantize_to_midi(&self, f: f64) -> (u8, f64) {
        let midi_float = hz_to_midi(f);
        let midi_note = midi_float.round() as i32;
        let cents = (midi_float - midi_note as f64) * 100.0;
        let clamped_note = midi_note.clamp(0, 127) as u8;
        (clamped_note, cents)
    }

    /// Snap `f` to the nearest frequency in `scale_degrees`.
    /// `scale_degrees` should be a list of frequencies in Hz (one octave
    /// or absolute values).  Returns the closest degree.
    pub fn quantize_to_scale(&self, f: f64, scale_degrees: &[f64]) -> f64 {
        if scale_degrees.is_empty() {
            return f;
        }

        // For each scale degree, also consider octave transpositions to
        // cover the target frequency's range.
        let mut best = scale_degrees[0];
        let mut best_dist = f64::INFINITY;

        for &deg in scale_degrees {
            if deg <= 0.0 {
                continue;
            }
            // Try octave shifts from -4 to +4 around the reference.
            for oct in -4..=4_i32 {
                let candidate = deg * 2.0_f64.powi(oct);
                let dist = (hz_to_cents(f, candidate)).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best = candidate;
                }
            }
        }
        best
    }

    /// Quantise `f` to the nearest step in an N-tone equal-temperament
    /// scale with reference frequency `ref_hz`.
    pub fn microtonal_quantize(
        &self,
        f: f64,
        divisions_per_octave: u32,
        ref_hz: f64,
    ) -> f64 {
        if divisions_per_octave == 0 || ref_hz <= 0.0 || f <= 0.0 {
            return f;
        }
        let n = divisions_per_octave as f64;
        let steps = n * (f / ref_hz).log2();
        let quantised_steps = steps.round();
        ref_hz * 2.0_f64.powf(quantised_steps / n)
    }

    /// Bark critical-band rate (convenience wrapper).
    pub fn bark_rate(&self, f: f64) -> f64 {
        hz_to_bark(f)
    }

    /// Mel rate (convenience wrapper).
    pub fn mel_rate(&self, f: f64) -> f64 {
        hz_to_mel(f)
    }

    /// Equivalent Rectangular Bandwidth rate (Glasberg & Moore 1990).
    pub fn erb_rate(&self, f: f64) -> f64 {
        21.4 * (4.37 * f / 1000.0 + 1.0).log10()
    }
}

impl Default for PitchScaleConverter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PitchDirection
// ---------------------------------------------------------------------------

/// Overall direction of a pitch contour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PitchDirection {
    Rising,
    Falling,
    Steady,
    Complex,
}

// ---------------------------------------------------------------------------
// PitchContour
// ---------------------------------------------------------------------------

/// A time-varying pitch trajectory, stored as (time_ms, frequency_hz) pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchContour {
    pub points: Vec<(f64, f64)>,
}

impl PitchContour {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Append a point.
    pub fn add_point(&mut self, time_ms: f64, freq_hz: f64) {
        self.points.push((time_ms, freq_hz));
    }

    /// Generate a linear glide in log-frequency space (portamento).
    pub fn glide(
        &self,
        start_hz: f64,
        end_hz: f64,
        duration_ms: f64,
        num_points: usize,
    ) -> PitchContour {
        let mut contour = PitchContour::new();
        if num_points < 2 {
            contour.add_point(0.0, start_hz);
            return contour;
        }
        let log_start = start_hz.ln();
        let log_end = end_hz.ln();
        for i in 0..num_points {
            let t = i as f64 / (num_points - 1) as f64;
            let time_ms = t * duration_ms;
            let log_f = log_start + t * (log_end - log_start);
            contour.add_point(time_ms, log_f.exp());
        }
        contour
    }

    /// Generate a sinusoidal vibrato contour around `center_hz`.
    pub fn vibrato(
        &self,
        center_hz: f64,
        rate_hz: f64,
        depth_cents: f64,
        duration_ms: f64,
        num_points: usize,
    ) -> PitchContour {
        let mut contour = PitchContour::new();
        if num_points == 0 {
            return contour;
        }
        let log_center = center_hz.ln();
        let depth_ratio = depth_cents / 1200.0 * 2.0_f64.ln();
        for i in 0..num_points {
            let t = if num_points > 1 {
                i as f64 / (num_points - 1) as f64
            } else {
                0.0
            };
            let time_ms = t * duration_ms;
            let phase = 2.0 * PI * rate_hz * (time_ms / 1000.0);
            let log_f = log_center + depth_ratio * phase.sin();
            contour.add_point(time_ms, log_f.exp());
        }
        contour
    }

    /// Linearly interpolate the frequency at a given time.
    pub fn frequency_at(&self, time_ms: f64) -> Option<f64> {
        if self.points.is_empty() {
            return None;
        }
        if self.points.len() == 1 {
            return Some(self.points[0].1);
        }

        // Before first point or after last point — clamp.
        if time_ms <= self.points.first().unwrap().0 {
            return Some(self.points.first().unwrap().1);
        }
        if time_ms >= self.points.last().unwrap().0 {
            return Some(self.points.last().unwrap().1);
        }

        // Find bracketing pair.
        for w in self.points.windows(2) {
            let (t0, f0) = w[0];
            let (t1, f1) = w[1];
            if time_ms >= t0 && time_ms <= t1 {
                let frac = if (t1 - t0).abs() < 1e-12 {
                    0.0
                } else {
                    (time_ms - t0) / (t1 - t0)
                };
                return Some(f0 + frac * (f1 - f0));
            }
        }
        None
    }

    /// Mean frequency across all points.
    pub fn average_frequency(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.points.iter().map(|&(_, f)| f).sum();
        sum / self.points.len() as f64
    }

    /// Minimum and maximum frequency in the contour.
    pub fn pitch_range(&self) -> (f64, f64) {
        if self.points.is_empty() {
            return (0.0, 0.0);
        }
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &(_, f) in &self.points {
            if f < lo {
                lo = f;
            }
            if f > hi {
                hi = f;
            }
        }
        (lo, hi)
    }

    /// Instantaneous slope (Hz per ms) at the given time, estimated via
    /// finite differences of the two surrounding points.
    pub fn slope_at(&self, time_ms: f64) -> Option<f64> {
        if self.points.len() < 2 {
            return None;
        }
        for w in self.points.windows(2) {
            let (t0, f0) = w[0];
            let (t1, f1) = w[1];
            if time_ms >= t0 && time_ms <= t1 {
                let dt = t1 - t0;
                if dt.abs() < 1e-12 {
                    return Some(0.0);
                }
                return Some((f1 - f0) / dt);
            }
        }
        // Outside range — use nearest segment.
        if time_ms < self.points.first().unwrap().0 {
            let (t0, f0) = self.points[0];
            let (t1, f1) = self.points[1];
            let dt = t1 - t0;
            return if dt.abs() < 1e-12 {
                Some(0.0)
            } else {
                Some((f1 - f0) / dt)
            };
        }
        let n = self.points.len();
        let (t0, f0) = self.points[n - 2];
        let (t1, f1) = self.points[n - 1];
        let dt = t1 - t0;
        if dt.abs() < 1e-12 {
            Some(0.0)
        } else {
            Some((f1 - f0) / dt)
        }
    }

    /// Is the contour approximately steady (constant pitch)?
    /// A contour is "steady" if all points fall within `tolerance_cents` of
    /// the mean frequency.
    pub fn is_steady(&self, tolerance_cents: f64) -> bool {
        if self.points.len() < 2 {
            return true;
        }
        let avg = self.average_frequency();
        if avg <= 0.0 {
            return true;
        }
        self.points.iter().all(|&(_, f)| {
            hz_to_cents(f, avg).abs() <= tolerance_cents
        })
    }

    /// Classify the overall direction of the contour.
    pub fn direction(&self) -> PitchDirection {
        if self.points.len() < 2 {
            return PitchDirection::Steady;
        }

        let mut rising = 0_usize;
        let mut falling = 0_usize;
        let threshold_cents = 10.0;
        for w in self.points.windows(2) {
            let cents = hz_to_cents(w[1].1, w[0].1);
            if cents > threshold_cents {
                rising += 1;
            } else if cents < -threshold_cents {
                falling += 1;
            }
        }

        let segments = self.points.len() - 1;
        let rise_frac = rising as f64 / segments as f64;
        let fall_frac = falling as f64 / segments as f64;

        if rise_frac > 0.7 {
            PitchDirection::Rising
        } else if fall_frac > 0.7 {
            PitchDirection::Falling
        } else if rise_frac < 0.15 && fall_frac < 0.15 {
            PitchDirection::Steady
        } else {
            PitchDirection::Complex
        }
    }
}

impl Default for PitchContour {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PitchAnalysisResult & PitchAnalyzer
// ---------------------------------------------------------------------------

/// Result of a single-frame pitch analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchAnalysisResult {
    /// Detected fundamental frequency (if any).
    pub fundamental: Option<f64>,
    /// Perceptual salience of the detected pitch (0–1).
    pub salience: f64,
    /// Detected harmonic partials as (frequency_hz, amplitude) pairs.
    pub harmonics: Vec<(f64, f64)>,
    /// Label of the algorithm that produced the estimate.
    pub method_used: String,
}

/// High-level pitch analyser that chains spectral peak-picking, harmonic
/// extraction, virtual-pitch estimation, and salience computation.
pub struct PitchAnalyzer {
    model: PitchModel,
}

impl PitchAnalyzer {
    pub fn new() -> Self {
        Self {
            model: PitchModel::new(),
        }
    }

    /// Run a full spectral-domain pitch analysis on one frame.
    pub fn analyze_spectrum(
        &self,
        magnitudes: &[f64],
        freq_bins: &[f64],
    ) -> PitchAnalysisResult {
        // 1. Attempt spectral peak picking.
        let spectral_f0 = self.model.spectral_pitch(magnitudes, freq_bins);

        if let Some(f0) = spectral_f0 {
            let harmonics = self.extract_harmonics(magnitudes, freq_bins, f0, 10);
            let salience = self.model.pitch_salience(magnitudes, freq_bins, f0);
            return PitchAnalysisResult {
                fundamental: Some(f0),
                salience,
                harmonics,
                method_used: "spectral_peak".to_string(),
            };
        }

        // 2. Fall back to virtual pitch from the strongest peaks.
        let peaks = self.top_peaks(magnitudes, freq_bins, 8);
        if peaks.is_empty() {
            return PitchAnalysisResult {
                fundamental: None,
                salience: 0.0,
                harmonics: Vec::new(),
                method_used: "none".to_string(),
            };
        }

        let peak_freqs: Vec<f64> = peaks.iter().map(|&(f, _)| f).collect();
        let virtual_f0 = self.model.virtual_pitch(&peak_freqs);

        if let Some(f0) = virtual_f0 {
            let harmonics = self.extract_harmonics(magnitudes, freq_bins, f0, 10);
            let salience = self.model.pitch_salience(magnitudes, freq_bins, f0);
            PitchAnalysisResult {
                fundamental: Some(f0),
                salience,
                harmonics,
                method_used: "virtual_pitch".to_string(),
            }
        } else {
            PitchAnalysisResult {
                fundamental: None,
                salience: 0.0,
                harmonics: Vec::new(),
                method_used: "none".to_string(),
            }
        }
    }

    /// Extract harmonic partials from the spectrum near expected locations
    /// n × f0 for n = 1..=`n_harmonics`.  Returns (frequency, amplitude).
    pub fn extract_harmonics(
        &self,
        magnitudes: &[f64],
        freq_bins: &[f64],
        f0: f64,
        n_harmonics: usize,
    ) -> Vec<(f64, f64)> {
        let mut result: Vec<(f64, f64)> = Vec::new();
        let tolerance = 2.0_f64.powf(self.model.harmonic_tolerance_cents / 1200.0);

        for n in 1..=n_harmonics {
            let target = f0 * n as f64;
            let lo = target / tolerance;
            let hi = target * tolerance;

            let mut best_freq = 0.0_f64;
            let mut best_mag = 0.0_f64;

            for (i, &f) in freq_bins.iter().enumerate() {
                if f >= lo && f <= hi && magnitudes[i] > best_mag {
                    best_mag = magnitudes[i];
                    best_freq = f;
                }
            }

            if best_mag > 0.0 {
                result.push((best_freq, best_mag));
            }
        }
        result
    }

    // -- internal helpers ---------------------------------------------------

    /// Return up to `n` spectral peaks sorted by descending magnitude.
    fn top_peaks(
        &self,
        magnitudes: &[f64],
        freq_bins: &[f64],
        n: usize,
    ) -> Vec<(f64, f64)> {
        let len = magnitudes.len();
        let mut peaks: Vec<(f64, f64)> = Vec::new();
        for i in 1..len.saturating_sub(1) {
            if magnitudes[i] > magnitudes[i - 1] && magnitudes[i] > magnitudes[i + 1] {
                peaks.push((freq_bins[i], magnitudes[i]));
            }
        }
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        peaks.truncate(n);
        peaks
    }
}

impl Default for PitchAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    // ---- helper conversion tests -----------------------------------------

    #[test]
    fn test_hz_to_midi_a4() {
        let midi = hz_to_midi(440.0);
        assert!((midi - 69.0).abs() < EPSILON, "A4=440 Hz should be MIDI 69, got {midi}");
    }

    #[test]
    fn test_midi_to_hz_middle_c() {
        let hz = midi_to_hz(60.0);
        assert!(
            (hz - 261.625_565_3).abs() < 0.01,
            "MIDI 60 should ≈ 261.63 Hz, got {hz}"
        );
    }

    #[test]
    fn test_mel_roundtrip() {
        for &f in &[100.0, 440.0, 1000.0, 4000.0, 8000.0] {
            let m = hz_to_mel(f);
            let back = mel_to_hz(m);
            assert!(
                (back - f).abs() < 0.01,
                "Mel roundtrip failed for {f} Hz: got {back}"
            );
        }
    }

    #[test]
    fn test_bark_roundtrip() {
        for &f in &[200.0, 500.0, 1000.0, 3000.0, 6000.0] {
            let z = hz_to_bark(f);
            let back = bark_to_hz(z);
            assert!(
                (back - f).abs() < 1.0,
                "Bark roundtrip failed for {f} Hz: got {back}"
            );
        }
    }

    // ---- virtual pitch ---------------------------------------------------

    #[test]
    fn test_virtual_pitch_full_series() {
        let model = PitchModel::new();
        let harmonics = vec![200.0, 300.0, 400.0, 500.0];
        let f0 = model.virtual_pitch(&harmonics).expect("should find f0");
        assert!(
            (f0 - 100.0).abs() < 2.0,
            "f0 of 200,300,400,500 should ≈ 100 Hz, got {f0}"
        );
    }

    #[test]
    fn test_virtual_pitch_missing_fundamental() {
        let model = PitchModel::new();
        let harmonics = vec![300.0, 400.0, 500.0];
        let f0 = model.virtual_pitch(&harmonics).expect("should find f0");
        assert!(
            (f0 - 100.0).abs() < 2.0,
            "Missing-fundamental f0 should ≈ 100 Hz, got {f0}"
        );
    }

    // ---- autocorrelation pitch -------------------------------------------

    #[test]
    fn test_autocorrelation_pure_sine() {
        let model = PitchModel::new();
        let sample_rate = 16000.0;
        let target_hz = 440.0;
        let n_samples = 2048;

        let signal: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * PI * target_hz * t).sin()
            })
            .collect();

        let f0 = model
            .autocorrelation_pitch(&signal, sample_rate)
            .expect("should detect pitch");
        assert!(
            (f0 - target_hz).abs() < 5.0,
            "Autocorrelation pitch should ≈ {target_hz} Hz, got {f0}"
        );
    }

    // ---- spectral pitch --------------------------------------------------

    #[test]
    fn test_spectral_pitch_single_peak() {
        let model = PitchModel::new();
        let n = 512;
        let df = 10.0; // 10 Hz per bin
        let freq_bins: Vec<f64> = (0..n).map(|i| i as f64 * df).collect();
        let target_bin = 44; // 440 Hz
        let mut mags = vec![0.0_f64; n];
        // Create a sharp peak at 440 Hz with a little spread.
        for offset in -2..=2_i32 {
            let idx = (target_bin as i32 + offset) as usize;
            let falloff = 1.0 - 0.3 * offset.unsigned_abs() as f64;
            mags[idx] = falloff.max(0.0);
        }

        let f0 = model
            .spectral_pitch(&mags, &freq_bins)
            .expect("should find spectral pitch");
        assert!(
            (f0 - 440.0).abs() < df,
            "Spectral pitch should ≈ 440 Hz, got {f0}"
        );
    }

    // ---- pitch contour ---------------------------------------------------

    #[test]
    fn test_glide_endpoints() {
        let contour = PitchContour::new();
        let glide = contour.glide(220.0, 440.0, 1000.0, 100);
        let first = glide.points.first().unwrap().1;
        let last = glide.points.last().unwrap().1;
        assert!(
            (first - 220.0).abs() < 0.1,
            "Glide start should ≈ 220 Hz, got {first}"
        );
        assert!(
            (last - 440.0).abs() < 0.1,
            "Glide end should ≈ 440 Hz, got {last}"
        );
    }

    #[test]
    fn test_vibrato_oscillates_around_center() {
        let contour = PitchContour::new();
        let vib = contour.vibrato(440.0, 5.0, 100.0, 1000.0, 200);
        let avg = vib.average_frequency();
        // Mean should be approximately 440 Hz (slight bias from log domain
        // but small for 100-cent depth).
        assert!(
            (avg - 440.0).abs() < 10.0,
            "Vibrato average should ≈ 440 Hz, got {avg}"
        );
        let (lo, hi) = vib.pitch_range();
        assert!(lo < 440.0, "Vibrato should dip below center");
        assert!(hi > 440.0, "Vibrato should rise above center");
    }

    // ---- microtonal quantize ---------------------------------------------

    #[test]
    fn test_microtonal_quantize_12tet() {
        let conv = PitchScaleConverter::new();
        // 445 Hz should snap to 440 Hz in 12-TET with ref 440.
        let q = conv.microtonal_quantize(445.0, 12, 440.0);
        assert!(
            (q - 440.0).abs() < 0.01,
            "445 Hz should quantise to 440 Hz in 12-TET, got {q}"
        );
    }

    // ---- ERB rate --------------------------------------------------------

    #[test]
    fn test_erb_rate_1khz() {
        let conv = PitchScaleConverter::new();
        let erb = conv.erb_rate(1000.0);
        // ERB_rate(1kHz) = 21.4 * log10(4.37 + 1) ≈ 21.4 * 0.7299 ≈ 15.62
        let expected = 21.4 * (4.37 + 1.0_f64).log10();
        assert!(
            (erb - expected).abs() < 0.01,
            "ERB rate at 1 kHz should ≈ {expected}, got {erb}"
        );
    }

    // ---- salience: harmonic vs inharmonic --------------------------------

    #[test]
    fn test_salience_harmonic_vs_inharmonic() {
        let model = PitchModel::new();
        let n = 1024;
        let df = 5.0;
        let freq_bins: Vec<f64> = (0..n).map(|i| i as f64 * df).collect();
        let f0 = 200.0;

        // Harmonic spectrum: energy at 200, 400, 600, 800, 1000 Hz.
        let mut harmonic_mags = vec![0.0_f64; n];
        for k in 1..=5 {
            let bin = (f0 * k as f64 / df).round() as usize;
            if bin < n {
                harmonic_mags[bin] = 1.0 / k as f64;
            }
        }

        // Inharmonic spectrum: energy at arbitrary non-harmonic locations.
        let mut inharmonic_mags = vec![0.0_f64; n];
        for &freq in &[173.0, 311.0, 587.0, 743.0, 1019.0] {
            let bin = (freq / df).round() as usize;
            if bin < n {
                inharmonic_mags[bin] = 0.5;
            }
        }

        let s_harm = model.pitch_salience(&harmonic_mags, &freq_bins, f0);
        let s_inharm = model.pitch_salience(&inharmonic_mags, &freq_bins, f0);

        assert!(
            s_harm > s_inharm,
            "Harmonic salience ({s_harm}) should exceed inharmonic ({s_inharm})"
        );
    }

    // ---- pitch analyzer end-to-end ---------------------------------------

    #[test]
    fn test_analyzer_extract_harmonics() {
        let analyzer = PitchAnalyzer::new();
        let n = 1024;
        let df = 5.0;
        let freq_bins: Vec<f64> = (0..n).map(|i| i as f64 * df).collect();
        let f0 = 100.0;
        let mut mags = vec![0.0_f64; n];
        for k in 1..=6 {
            let bin = (f0 * k as f64 / df).round() as usize;
            if bin < n {
                mags[bin] = 1.0 / k as f64;
            }
        }

        let harmonics = analyzer.extract_harmonics(&mags, &freq_bins, f0, 6);
        assert!(
            harmonics.len() >= 5,
            "Should find at least 5 harmonics, got {}",
            harmonics.len()
        );
        // First harmonic should be near f0.
        let first_freq = harmonics[0].0;
        assert!(
            (first_freq - f0).abs() < df,
            "First harmonic should ≈ {f0}, got {first_freq}"
        );
    }

    #[test]
    fn test_pitch_contour_direction_rising() {
        let mut contour = PitchContour::new();
        for i in 0..20 {
            let t = i as f64 * 50.0;
            let f = 200.0 * 2.0_f64.powf(i as f64 / 19.0); // one-octave rise
            contour.add_point(t, f);
        }
        assert_eq!(contour.direction(), PitchDirection::Rising);
    }

    #[test]
    fn test_pitch_contour_steady() {
        let mut contour = PitchContour::new();
        for i in 0..20 {
            contour.add_point(i as f64 * 10.0, 440.0);
        }
        assert!(contour.is_steady(5.0));
        assert_eq!(contour.direction(), PitchDirection::Steady);
    }
}
