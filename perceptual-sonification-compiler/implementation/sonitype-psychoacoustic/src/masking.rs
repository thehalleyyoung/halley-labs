//! Critical-Band Masking Models
//!
//! Implements the Schroeder spreading function, Zwicker masking model (simultaneous
//! and temporal masking), global masking threshold computation, excitation patterns,
//! specific loudness, and multi-stream masking analysis.

use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Local type definitions
// ---------------------------------------------------------------------------

pub type Frequency = f64;
pub type Amplitude = f64;
pub type DecibelSpl = f64;

pub const NUM_BARK_BANDS: usize = 24;

/// Standard Bark-band edge frequencies (Hz) for the 24 critical bands.
const BARK_BAND_EDGES_HZ: [(f64, f64); NUM_BARK_BANDS] = [
    (20.0, 100.0),
    (100.0, 200.0),
    (200.0, 300.0),
    (300.0, 400.0),
    (400.0, 510.0),
    (510.0, 630.0),
    (630.0, 770.0),
    (770.0, 920.0),
    (920.0, 1080.0),
    (1080.0, 1270.0),
    (1270.0, 1480.0),
    (1480.0, 1720.0),
    (1720.0, 2000.0),
    (2000.0, 2320.0),
    (2320.0, 2700.0),
    (2700.0, 3150.0),
    (3150.0, 3700.0),
    (3700.0, 4400.0),
    (4400.0, 5300.0),
    (5300.0, 6400.0),
    (6400.0, 7700.0),
    (7700.0, 9500.0),
    (9500.0, 12000.0),
    (12000.0, 15500.0),
];

/// A single Bark critical band.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarkBand {
    pub index: usize,
    pub lower_hz: f64,
    pub upper_hz: f64,
    pub center_hz: f64,
}

/// Convert frequency in Hz to Bark rate (Traunmüller, 1990).
pub fn hz_to_bark(f: f64) -> f64 {
    let z = 26.81 * f / (1960.0 + f) - 0.53;
    if z < 0.0 { 0.0 } else { z }
}

/// Convert Bark rate to Hz (inverse Traunmüller).
pub fn bark_to_hz(z: f64) -> f64 {
    let z_adj = z + 0.53;
    if z_adj >= 26.81 {
        return 20_000.0;
    }
    1960.0 * z_adj / (26.81 - z_adj)
}

/// Return all 24 standard critical bands.
pub fn bark_band_edges() -> Vec<BarkBand> {
    BARK_BAND_EDGES_HZ
        .iter()
        .enumerate()
        .map(|(i, &(lo, hi))| BarkBand {
            index: i,
            lower_hz: lo,
            upper_hz: hi,
            center_hz: (lo + hi) * 0.5,
        })
        .collect()
}

/// Convert linear power ratio to decibels.
#[inline]
fn lin_to_db(x: f64) -> f64 {
    if x <= 0.0 {
        -120.0
    } else {
        10.0 * x.log10()
    }
}

/// Convert decibels to linear power ratio.
#[inline]
fn db_to_lin(db: f64) -> f64 {
    10.0_f64.powf(db / 10.0)
}

// ---------------------------------------------------------------------------
// Schroeder Spreading Function
// ---------------------------------------------------------------------------

/// Schroeder (1979) masking spread across Bark bands.
///
/// Models how energy in one critical band masks energy in neighbouring bands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchroederSpreadingFunction {
    _private: (),
}

impl SchroederSpreadingFunction {
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Spreading function in dB as a function of Bark distance Δz.
    ///
    /// S(Δz) = 15.81 + 7.5 · (Δz + 0.474) − 17.5 · √(1 + (Δz + 0.474)²)
    pub fn spreading_db(&self, bark_distance: f64) -> f64 {
        let d = bark_distance + 0.474;
        15.81 + 7.5 * d - 17.5 * (1.0 + d * d).sqrt()
    }

    /// Spreading function in linear power ratio.
    pub fn spreading_linear(&self, bark_distance: f64) -> f64 {
        db_to_lin(self.spreading_db(bark_distance))
    }

    /// Given masker levels (dB SPL) in each of the 24 Bark bands, compute the
    /// masked threshold at every band by accumulating spread masking contributions.
    pub fn compute_masking_threshold(&self, masker_levels: &[f64]) -> Vec<f64> {
        let n = masker_levels.len().min(NUM_BARK_BANDS);
        let bands = bark_band_edges();
        let mut threshold = vec![-120.0_f64; n];
        for target in 0..n {
            let target_bark = hz_to_bark(bands[target].center_hz);
            let mut total_lin = 0.0_f64;
            for masker in 0..n {
                if masker_levels[masker] <= -120.0 {
                    continue;
                }
                let masker_bark = hz_to_bark(bands[masker].center_hz);
                let dist = target_bark - masker_bark;
                let spread_db = self.spreading_db(dist);
                let contribution_db = masker_levels[masker] + spread_db;
                if contribution_db > -120.0 {
                    total_lin += db_to_lin(contribution_db);
                }
            }
            threshold[target] = lin_to_db(total_lin);
        }
        threshold
    }

    /// Compute spreading function values for all 24×24 band pairs.
    pub fn spreading_matrix(&self) -> [[f64; NUM_BARK_BANDS]; NUM_BARK_BANDS] {
        let bands = bark_band_edges();
        let mut matrix = [[0.0_f64; NUM_BARK_BANDS]; NUM_BARK_BANDS];
        for i in 0..NUM_BARK_BANDS {
            let zi = hz_to_bark(bands[i].center_hz);
            for j in 0..NUM_BARK_BANDS {
                let zj = hz_to_bark(bands[j].center_hz);
                matrix[i][j] = self.spreading_db(zi - zj);
            }
        }
        matrix
    }
}

impl Default for SchroederSpreadingFunction {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Zwicker Masking Model
// ---------------------------------------------------------------------------

/// Full Zwicker masking model with simultaneous masking, temporal masking,
/// threshold in quiet, excitation patterns, and specific loudness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZwickerMaskingModel {
    /// Reference sound pressure level for 0 dB (20 µPa in air).
    pub reference_spl: f64,
    /// Forward masking decay slope (dB per decade of ms).
    pub forward_decay_slope: f64,
    /// Backward masking maximum advance (ms).
    pub backward_max_advance_ms: f64,
}

impl ZwickerMaskingModel {
    pub fn new() -> Self {
        Self {
            reference_spl: 0.0,
            forward_decay_slope: 25.0,
            backward_max_advance_ms: 5.0,
        }
    }

    // ── Simultaneous masking offsets ─────────────────────────────────────

    /// Tone-masking-noise (TMN) offset in dB.  A tonal masker is less effective
    /// at masking noise than a noise masker is at masking a tone.
    /// TMN ≈ 14.5 + band · 0.15 (increases slightly with Bark band number).
    pub fn tone_masking_noise_offset(&self, bark_band: usize) -> f64 {
        14.5 + bark_band as f64 * 0.15
    }

    /// Noise-masking-tone (NMT) offset in dB.  Roughly 5.5 dB, nearly constant.
    pub fn noise_masking_tone_offset(&self, _bark_band: usize) -> f64 {
        5.5
    }

    /// Simultaneous masking threshold given a single masker.
    ///
    /// Returns the level (dB SPL) at which a signal just becomes audible in the
    /// presence of a masker at `masker_level_db`.
    pub fn simultaneous_masking_threshold(
        &self,
        masker_level_db: f64,
        masker_is_tonal: bool,
        bark_band: usize,
    ) -> f64 {
        let offset = if masker_is_tonal {
            self.tone_masking_noise_offset(bark_band)
        } else {
            self.noise_masking_tone_offset(bark_band)
        };
        masker_level_db - offset
    }

    // ── Temporal masking ─────────────────────────────────────────────────

    /// Forward masking decay: threshold at `delay_ms` after masker offset.
    ///
    /// Approximately: masker_level − slope · log₁₀(delay / t_ref), t_ref = 1 ms.
    pub fn forward_masking_decay(&self, masker_level_db: f64, delay_ms: f64) -> f64 {
        if delay_ms <= 0.0 {
            return masker_level_db;
        }
        let t_ref = 1.0;
        let decay = self.forward_decay_slope * (delay_ms / t_ref).log10();
        (masker_level_db - decay).max(self.threshold_in_quiet(1000.0))
    }

    /// Backward masking: effective only for ~5 ms before the masker onset.
    ///
    /// Returns the threshold elevation at `advance_ms` before the masker.
    pub fn backward_masking(&self, masker_level_db: f64, advance_ms: f64) -> f64 {
        if advance_ms <= 0.0 || advance_ms > self.backward_max_advance_ms {
            return -120.0;
        }
        let fraction = 1.0 - advance_ms / self.backward_max_advance_ms;
        masker_level_db * fraction * 0.5
    }

    // ── Threshold in quiet ───────────────────────────────────────────────

    /// Absolute threshold of hearing (ISO 389-7 approximation).
    ///
    /// T_q(f) = 3.64 · (f/1000)^−0.8  − 6.5 · exp(−0.6 · (f/1000 − 3.3)²)
    ///          + 10⁻³ · (f/1000)⁴
    pub fn threshold_in_quiet(&self, freq_hz: f64) -> f64 {
        let f_khz = freq_hz / 1000.0;
        if f_khz <= 0.0 {
            return 120.0;
        }
        3.64 * f_khz.powf(-0.8)
            - 6.5 * (-0.6 * (f_khz - 3.3).powi(2)).exp()
            + 1e-3 * f_khz.powi(4)
    }

    /// Threshold-in-quiet per Bark band (using center frequency).
    pub fn threshold_in_quiet_bands(&self) -> Vec<f64> {
        bark_band_edges()
            .iter()
            .map(|b| self.threshold_in_quiet(b.center_hz))
            .collect()
    }

    // ── Global masking threshold ─────────────────────────────────────────

    /// Global masking threshold across all 24 Bark bands.
    ///
    /// `band_levels` contains `(level_dB, is_tonal)` per Bark band.
    /// The global threshold is the pointwise maximum of:
    ///   • the threshold in quiet
    ///   • all simultaneous masking contributions (spread across bands)
    pub fn global_masking_threshold(&self, band_levels: &[(f64, bool)]) -> Vec<f64> {
        let n = band_levels.len().min(NUM_BARK_BANDS);
        let bands = bark_band_edges();
        let tiq = self.threshold_in_quiet_bands();
        let spreading = SchroederSpreadingFunction::new();

        let mut threshold = vec![-120.0_f64; n];
        for target in 0..n {
            let target_bark = hz_to_bark(bands[target].center_hz);
            // Start with quiet threshold as floor
            let mut total_lin = db_to_lin(tiq[target]);

            for masker in 0..n {
                let (masker_db, is_tonal) = band_levels[masker];
                if masker_db <= -120.0 {
                    continue;
                }
                let masker_bark = hz_to_bark(bands[masker].center_hz);
                let dist = target_bark - masker_bark;
                let spread_db = spreading.spreading_db(dist);
                let offset = if is_tonal {
                    self.tone_masking_noise_offset(masker)
                } else {
                    self.noise_masking_tone_offset(masker)
                };
                let contrib_db = masker_db + spread_db - offset;
                if contrib_db > -120.0 {
                    total_lin += db_to_lin(contrib_db);
                }
            }
            threshold[target] = lin_to_db(total_lin);
        }
        threshold
    }

    // ── Excitation patterns ──────────────────────────────────────────────

    /// Compute the excitation level at each Bark band from a spectrum.
    ///
    /// The excitation pattern is the spectrum convolved with the spreading function.
    pub fn excitation_pattern(&self, spectrum_db: &[f64]) -> Vec<f64> {
        let spreading = SchroederSpreadingFunction::new();
        spreading.compute_masking_threshold(spectrum_db)
    }

    /// Level-dependent excitation pattern that accounts for the fact that
    /// the upper slope of masking becomes shallower at high levels.
    pub fn excitation_pattern_level_dependent(&self, spectrum_db: &[f64]) -> Vec<f64> {
        let n = spectrum_db.len().min(NUM_BARK_BANDS);
        let bands = bark_band_edges();
        let mut excitation = vec![-120.0_f64; n];

        for target in 0..n {
            let target_bark = hz_to_bark(bands[target].center_hz);
            let mut total_lin = 0.0_f64;
            for masker in 0..n {
                if spectrum_db[masker] <= -120.0 {
                    continue;
                }
                let masker_bark = hz_to_bark(bands[masker].center_hz);
                let dist = target_bark - masker_bark;
                let spread = asymmetric_spreading_db(dist, spectrum_db[masker]);
                let contrib = spectrum_db[masker] + spread;
                if contrib > -120.0 {
                    total_lin += db_to_lin(contrib);
                }
            }
            excitation[target] = lin_to_db(total_lin);
        }
        excitation
    }

    // ── Specific loudness (Zwicker model) ────────────────────────────────

    /// Specific loudness N′(z) in sone/Bark from excitation level.
    ///
    /// Uses a simplified Zwicker-Fastl formula:
    ///   N′ = 0.08 · (E_TQ / s)^0.23 · [( 0.5 + 0.5·E/E_TQ )^0.23 − 1 ]
    pub fn specific_loudness(&self, excitation_db: &[f64]) -> Vec<f64> {
        let tiq = self.threshold_in_quiet_bands();
        let mut result = Vec::with_capacity(excitation_db.len());
        for (i, &e_db) in excitation_db.iter().enumerate() {
            let etq_db = if i < tiq.len() { tiq[i] } else { 3.0 };
            if e_db <= etq_db - 10.0 {
                result.push(0.0);
                continue;
            }
            let e_lin = db_to_lin(e_db);
            let etq_lin = db_to_lin(etq_db).max(1e-12);
            let ratio = (0.5 + 0.5 * e_lin / etq_lin).max(0.0);
            let n_prime = 0.08 * etq_lin.powf(0.23) * (ratio.powf(0.23) - 1.0);
            result.push(n_prime.max(0.0));
        }
        result
    }

    /// Total loudness in sone by integrating specific loudness across bands.
    /// Each Bark band has width ≈ 1 Bark.
    pub fn total_loudness(&self, specific_loudness: &[f64]) -> f64 {
        specific_loudness.iter().sum::<f64>()
    }

    /// Loudness in phon from total loudness in sone.
    pub fn loudness_to_phon(&self, sone: f64) -> f64 {
        if sone <= 0.0 {
            return 0.0;
        }
        if sone >= 1.0 {
            40.0 + 33.22 * sone.log10()
        } else {
            40.0 * (sone + 0.0005).powf(0.35)
        }
    }
}

impl Default for ZwickerMaskingModel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Masking Analyzer
// ---------------------------------------------------------------------------

/// A masked frequency region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskedRegion {
    pub band_index: usize,
    pub signal_level: f64,
    pub threshold_level: f64,
    pub deficit_db: f64,
}

/// A suggestion to reallocate a signal component to avoid masking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReallocationSuggestion {
    pub from_band: usize,
    pub to_band: usize,
    pub estimated_gain_db: f64,
    pub reason: String,
}

/// Multi-stream masking interaction analyser.
#[derive(Debug, Clone)]
pub struct MaskingAnalyzer {
    model: ZwickerMaskingModel,
}

impl MaskingAnalyzer {
    pub fn new(model: ZwickerMaskingModel) -> Self {
        Self { model }
    }

    /// Is a signal component audible above the masked threshold?
    pub fn is_audible(&self, signal_level_db: f64, masked_threshold_db: f64) -> bool {
        signal_level_db > masked_threshold_db
    }

    /// Masking margin: positive means audible, negative means masked.
    pub fn masking_margin(&self, signal_level_db: f64, masked_threshold_db: f64) -> f64 {
        signal_level_db - masked_threshold_db
    }

    /// Identify Bark bands where the signal is below the masked threshold.
    pub fn identify_masked_regions(
        &self,
        signal_spectrum: &[f64],
        masked_threshold: &[f64],
    ) -> Vec<MaskedRegion> {
        let n = signal_spectrum.len().min(masked_threshold.len());
        let mut regions = Vec::new();
        for i in 0..n {
            let margin = signal_spectrum[i] - masked_threshold[i];
            if margin < 0.0 {
                regions.push(MaskedRegion {
                    band_index: i,
                    signal_level: signal_spectrum[i],
                    threshold_level: masked_threshold[i],
                    deficit_db: -margin,
                });
            }
        }
        regions
    }

    /// For each masked region suggest moving the content to an available
    /// (unmasked) band that has the least distance.
    pub fn suggest_frequency_reallocation(
        &self,
        masked_regions: &[MaskedRegion],
        available_bands: &[bool],
    ) -> Vec<ReallocationSuggestion> {
        let mut suggestions = Vec::new();
        for region in masked_regions {
            let mut best: Option<(usize, f64)> = None;
            for (j, &avail) in available_bands.iter().enumerate() {
                if !avail {
                    continue;
                }
                let dist = (j as f64 - region.band_index as f64).abs();
                if best.is_none() || dist < best.unwrap().1 {
                    best = Some((j, dist));
                }
            }
            if let Some((to, _)) = best {
                suggestions.push(ReallocationSuggestion {
                    from_band: region.band_index,
                    to_band: to,
                    estimated_gain_db: region.deficit_db,
                    reason: format!(
                        "Band {} masked by {:.1} dB; move to band {}",
                        region.band_index, region.deficit_db, to
                    ),
                });
            }
        }
        suggestions
    }

    /// Compute the global masking threshold given multiple streams, then check
    /// each stream's audibility against the combined threshold from all *other*
    /// streams.
    pub fn analyze_multi_stream(
        &self,
        stream_spectra: &[Vec<f64>],
        is_tonal: &[bool],
    ) -> Vec<Vec<MaskedRegion>> {
        let n_streams = stream_spectra.len();
        let n_bands = stream_spectra
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0)
            .min(NUM_BARK_BANDS);

        let mut results = Vec::with_capacity(n_streams);

        for target_idx in 0..n_streams {
            let mut combined: Vec<(f64, bool)> = vec![(-120.0, false); n_bands];
            for (src_idx, spec) in stream_spectra.iter().enumerate() {
                if src_idx == target_idx {
                    continue;
                }
                let tonal = if src_idx < is_tonal.len() {
                    is_tonal[src_idx]
                } else {
                    false
                };
                for (b, &lev) in spec.iter().enumerate().take(n_bands) {
                    let cur_lin = db_to_lin(combined[b].0);
                    let add_lin = db_to_lin(lev);
                    combined[b].0 = lin_to_db(cur_lin + add_lin);
                    if tonal {
                        combined[b].1 = true;
                    }
                }
            }

            let threshold = self.model.global_masking_threshold(&combined);
            let target_spec = &stream_spectra[target_idx];
            let sig: Vec<f64> = (0..n_bands)
                .map(|b| {
                    if b < target_spec.len() {
                        target_spec[b]
                    } else {
                        -120.0
                    }
                })
                .collect();
            results.push(self.identify_masked_regions(&sig, &threshold));
        }
        results
    }

    /// One-shot multi-stream masking report returning regions and suggestions.
    pub fn full_multi_stream_analysis(
        &self,
        stream_spectra: &[Vec<f64>],
        is_tonal: &[bool],
    ) -> Vec<MultiStreamMaskingReport> {
        let masked = self.analyze_multi_stream(stream_spectra, is_tonal);
        let n_bands = stream_spectra
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(NUM_BARK_BANDS)
            .min(NUM_BARK_BANDS);

        masked
            .into_iter()
            .enumerate()
            .map(|(idx, regions)| {
                let available: Vec<bool> = (0..n_bands)
                    .map(|b| {
                        if b < stream_spectra[idx].len() {
                            !regions.iter().any(|r| r.band_index == b)
                        } else {
                            true
                        }
                    })
                    .collect();
                let suggestions =
                    self.suggest_frequency_reallocation(&regions, &available);
                MultiStreamMaskingReport {
                    stream_index: idx,
                    masked_regions: regions,
                    reallocation_suggestions: suggestions,
                }
            })
            .collect()
    }
}

/// Per-stream masking report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiStreamMaskingReport {
    pub stream_index: usize,
    pub masked_regions: Vec<MaskedRegion>,
    pub reallocation_suggestions: Vec<ReallocationSuggestion>,
}

// ---------------------------------------------------------------------------
// Spectral Masking Matrix
// ---------------------------------------------------------------------------

/// 24×24 matrix of pairwise masking interaction between Bark bands.
#[derive(Debug, Clone)]
pub struct SpectralMaskingMatrix {
    matrix: [[f64; NUM_BARK_BANDS]; NUM_BARK_BANDS],
}

impl SpectralMaskingMatrix {
    pub fn new() -> Self {
        Self {
            matrix: [[0.0; NUM_BARK_BANDS]; NUM_BARK_BANDS],
        }
    }

    /// Fill the matrix from a set of band levels (dB SPL).
    ///
    /// Entry (i, j) = masking contribution of band j onto band i (in dB).
    pub fn compute(&mut self, band_levels: &[f64]) {
        let spreading = SchroederSpreadingFunction::new();
        let bands = bark_band_edges();
        for target in 0..NUM_BARK_BANDS {
            let target_bark = hz_to_bark(bands[target].center_hz);
            for masker in 0..NUM_BARK_BANDS {
                if masker >= band_levels.len() || band_levels[masker] <= -120.0 {
                    self.matrix[target][masker] = -120.0;
                    continue;
                }
                let masker_bark = hz_to_bark(bands[masker].center_hz);
                let dist = target_bark - masker_bark;
                let spread = spreading.spreading_db(dist);
                self.matrix[target][masker] = band_levels[masker] + spread;
            }
        }
    }

    /// Get masking contribution of `masker_band` on `masked_band` (dB).
    pub fn get(&self, masked_band: usize, masker_band: usize) -> f64 {
        if masked_band < NUM_BARK_BANDS && masker_band < NUM_BARK_BANDS {
            self.matrix[masked_band][masker_band]
        } else {
            -120.0
        }
    }

    /// Which band contributes the most masking at the given `band`?
    pub fn dominant_masker(&self, band: usize) -> usize {
        if band >= NUM_BARK_BANDS {
            return 0;
        }
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;
        for j in 0..NUM_BARK_BANDS {
            if self.matrix[band][j] > best_val {
                best_val = self.matrix[band][j];
                best_idx = j;
            }
        }
        best_idx
    }

    /// Total masking at a band (energy sum of all masker contributions, in dB).
    pub fn total_masking_at(&self, band: usize) -> f64 {
        if band >= NUM_BARK_BANDS {
            return -120.0;
        }
        let total_lin: f64 = (0..NUM_BARK_BANDS)
            .map(|j| db_to_lin(self.matrix[band][j]))
            .sum();
        lin_to_db(total_lin)
    }

    /// Return the raw matrix as a reference.
    pub fn as_rows(&self) -> &[[f64; NUM_BARK_BANDS]; NUM_BARK_BANDS] {
        &self.matrix
    }

    /// Number of bands where the masking from `masker_band` exceeds `threshold_db`.
    pub fn masking_reach(&self, masker_band: usize, threshold_db: f64) -> usize {
        if masker_band >= NUM_BARK_BANDS {
            return 0;
        }
        (0..NUM_BARK_BANDS)
            .filter(|&i| self.matrix[i][masker_band] > threshold_db)
            .count()
    }
}

impl Default for SpectralMaskingMatrix {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utility: tonality estimation (simple heuristic)
// ---------------------------------------------------------------------------

/// Estimate whether each band is tonal based on its level relative to neighbours.
/// A band is considered tonal if its level exceeds the average of its two
/// immediate neighbours by at least `threshold_db`.
pub fn estimate_tonality(band_levels: &[f64], threshold_db: f64) -> Vec<bool> {
    let n = band_levels.len();
    let mut tonal = vec![false; n];
    for i in 0..n {
        let left = if i > 0 { band_levels[i - 1] } else { band_levels[i] };
        let right = if i + 1 < n {
            band_levels[i + 1]
        } else {
            band_levels[i]
        };
        let avg_neighbours = (left + right) / 2.0;
        tonal[i] = band_levels[i] - avg_neighbours >= threshold_db;
    }
    tonal
}

/// Convenience: compute global masking threshold from a raw spectrum with
/// automatic tonality estimation.
pub fn quick_masking_threshold(band_levels: &[f64]) -> Vec<f64> {
    let tonality = estimate_tonality(band_levels, 7.0);
    let pairs: Vec<(f64, bool)> = band_levels
        .iter()
        .zip(tonality.iter())
        .map(|(&l, &t)| (l, t))
        .collect();
    let model = ZwickerMaskingModel::new();
    model.global_masking_threshold(&pairs)
}

/// Compute excitation pattern and specific loudness in one pass.
pub fn excitation_and_loudness(band_levels: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
    let model = ZwickerMaskingModel::new();
    let excitation = model.excitation_pattern(band_levels);
    let specific = model.specific_loudness(&excitation);
    let total = model.total_loudness(&specific);
    (excitation, specific, total)
}

// ---------------------------------------------------------------------------
// Level-dependent spreading (asymmetric upward spread)
// ---------------------------------------------------------------------------

/// Level-dependent upper slope for more accurate upward spread of masking.
/// At higher levels the upward slope becomes shallower (more spread).
pub fn level_dependent_upper_slope(masker_level_db: f64) -> f64 {
    let base_slope = -27.0; // dB/Bark for low-level masker
    let level_factor = 0.17 * (masker_level_db - 40.0).max(0.0);
    base_slope + level_factor
}

/// Asymmetric spreading function that accounts for the level-dependent upper
/// slope (upward spread of masking grows with masker level).
pub fn asymmetric_spreading_db(bark_distance: f64, masker_level_db: f64) -> f64 {
    if bark_distance >= 0.0 {
        let slope = level_dependent_upper_slope(masker_level_db);
        slope * bark_distance
    } else {
        -40.0 * bark_distance.abs()
    }
}

/// Compute masking threshold using asymmetric, level-dependent spreading.
pub fn asymmetric_masking_threshold(band_levels: &[f64]) -> Vec<f64> {
    let n = band_levels.len().min(NUM_BARK_BANDS);
    let bands = bark_band_edges();
    let mut threshold = vec![-120.0_f64; n];
    for target in 0..n {
        let target_bark = hz_to_bark(bands[target].center_hz);
        let mut total_lin = 0.0_f64;
        for masker in 0..n {
            if band_levels[masker] <= -120.0 {
                continue;
            }
            let masker_bark = hz_to_bark(bands[masker].center_hz);
            let dist = target_bark - masker_bark;
            let spread = asymmetric_spreading_db(dist, band_levels[masker]);
            let contrib_db = band_levels[masker] + spread;
            if contrib_db > -120.0 {
                total_lin += db_to_lin(contrib_db);
            }
        }
        threshold[target] = lin_to_db(total_lin);
    }
    threshold
}

// ---------------------------------------------------------------------------
// Bark-band energy aggregator
// ---------------------------------------------------------------------------

/// Given a fine-grained spectrum (freq_bins in Hz, magnitudes in linear
/// amplitude), aggregate energy into 24 Bark bands and return levels in dB SPL.
pub fn aggregate_to_bark_bands(freq_bins: &[f64], magnitudes: &[f64]) -> Vec<f64> {
    let bands = bark_band_edges();
    let mut band_energy = vec![0.0_f64; NUM_BARK_BANDS];
    for (i, &f) in freq_bins.iter().enumerate() {
        if i >= magnitudes.len() {
            break;
        }
        let mag = magnitudes[i];
        let energy = mag * mag;
        for b in &bands {
            if f >= b.lower_hz && f < b.upper_hz {
                band_energy[b.index] += energy;
                break;
            }
        }
    }
    band_energy
        .iter()
        .map(|&e| lin_to_db(e.max(1e-30)))
        .collect()
}

/// Convert an amplitude spectrum to a Bark-band power spectrum, returning
/// both band levels in dB and the raw energies.
pub fn bark_spectrum(freq_bins: &[f64], magnitudes: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let bands = bark_band_edges();
    let mut band_energy = vec![0.0_f64; NUM_BARK_BANDS];
    for (i, &f) in freq_bins.iter().enumerate() {
        if i >= magnitudes.len() {
            break;
        }
        let mag = magnitudes[i];
        let energy = mag * mag;
        for b in &bands {
            if f >= b.lower_hz && f < b.upper_hz {
                band_energy[b.index] += energy;
                break;
            }
        }
    }
    let levels: Vec<f64> = band_energy
        .iter()
        .map(|&e| lin_to_db(e.max(1e-30)))
        .collect();
    (levels, band_energy)
}

// ---------------------------------------------------------------------------
// MaskingReport: a self-contained analysis artifact
// ---------------------------------------------------------------------------

/// Complete masking analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskingReport {
    pub global_threshold: Vec<f64>,
    pub excitation_pattern: Vec<f64>,
    pub specific_loudness: Vec<f64>,
    pub total_loudness_sone: f64,
    pub masked_regions: Vec<MaskedRegion>,
    pub reallocation_suggestions: Vec<ReallocationSuggestion>,
}

/// Produce a full masking report for a signal spectrum against a masking
/// background.
pub fn full_masking_report(
    signal_spectrum: &[f64],
    background_levels: &[(f64, bool)],
) -> MaskingReport {
    let model = ZwickerMaskingModel::new();
    let threshold = model.global_masking_threshold(background_levels);
    let excitation = model.excitation_pattern(signal_spectrum);
    let specific = model.specific_loudness(&excitation);
    let total = model.total_loudness(&specific);

    let analyzer = MaskingAnalyzer::new(model);
    let masked = analyzer.identify_masked_regions(signal_spectrum, &threshold);
    let available: Vec<bool> = signal_spectrum
        .iter()
        .zip(threshold.iter())
        .map(|(&s, &t)| s > t)
        .collect();
    let suggestions = analyzer.suggest_frequency_reallocation(&masked, &available);

    MaskingReport {
        global_threshold: threshold,
        excitation_pattern: excitation,
        specific_loudness: specific,
        total_loudness_sone: total,
        masked_regions: masked,
        reallocation_suggestions: suggestions,
    }
}

// ---------------------------------------------------------------------------
// Signal-to-Mask Ratio helpers
// ---------------------------------------------------------------------------

/// Compute signal-to-mask ratio (SMR) in dB for each band.
pub fn signal_to_mask_ratio(signal_db: &[f64], mask_threshold_db: &[f64]) -> Vec<f64> {
    signal_db
        .iter()
        .zip(mask_threshold_db.iter())
        .map(|(&s, &m)| s - m)
        .collect()
}

/// Minimum SMR across all bands (worst-case audibility).
pub fn min_signal_to_mask_ratio(signal_db: &[f64], mask_threshold_db: &[f64]) -> f64 {
    signal_to_mask_ratio(signal_db, mask_threshold_db)
        .into_iter()
        .fold(f64::INFINITY, f64::min)
}

/// Perceptual Entropy estimate based on masking threshold.
/// Higher PE indicates more perceptually relevant information.
pub fn perceptual_entropy(signal_db: &[f64], mask_threshold_db: &[f64]) -> f64 {
    let mut pe = 0.0;
    for (&s, &m) in signal_db.iter().zip(mask_threshold_db.iter()) {
        let smr = s - m;
        if smr > 0.0 {
            // Number of audible quantisation steps ∝ 10^(SMR/20)
            let steps = db_to_lin(smr * 0.5).max(1.0);
            pe += steps.log2();
        }
    }
    pe
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_to_bark_and_inverse() {
        for &f in &[100.0, 500.0, 1000.0, 4000.0, 10000.0] {
            let z = hz_to_bark(f);
            let f2 = bark_to_hz(z);
            assert!(
                (f - f2).abs() < 1.0,
                "Inverse failed for f={f}: got {f2}"
            );
        }
    }

    #[test]
    fn test_bark_band_count() {
        let bands = bark_band_edges();
        assert_eq!(bands.len(), 24);
    }

    #[test]
    fn test_schroeder_at_zero() {
        let sf = SchroederSpreadingFunction::new();
        let val = sf.spreading_db(0.0);
        // At distance 0 the value should be near 0 dB (within ~1 dB).
        assert!(
            val.abs() < 1.5,
            "Spreading at zero distance should be near 0 dB, got {val}"
        );
    }

    #[test]
    fn test_spreading_asymmetry() {
        let sf = SchroederSpreadingFunction::new();
        let up1 = sf.spreading_db(1.0);
        let down1 = sf.spreading_db(-1.0);
        // Upward spread (positive distance) is shallower → less attenuation
        assert!(
            up1 > down1,
            "Upward spread ({up1} dB) should be > downward ({down1} dB)"
        );
    }

    #[test]
    fn test_threshold_in_quiet_shape() {
        let model = ZwickerMaskingModel::new();
        let t_1k = model.threshold_in_quiet(1000.0);
        let t_4k = model.threshold_in_quiet(4000.0);
        let t_100 = model.threshold_in_quiet(100.0);
        let t_10k = model.threshold_in_quiet(10000.0);
        assert!(t_4k < t_100, "4 kHz ({t_4k}) < 100 Hz ({t_100})");
        assert!(t_4k < t_10k, "4 kHz ({t_4k}) < 10 kHz ({t_10k})");
        assert!(t_1k < t_100, "1 kHz ({t_1k}) < 100 Hz ({t_100})");
    }

    #[test]
    fn test_tmn_greater_than_nmt() {
        let model = ZwickerMaskingModel::new();
        for band in 0..NUM_BARK_BANDS {
            let tmn = model.tone_masking_noise_offset(band);
            let nmt = model.noise_masking_tone_offset(band);
            assert!(tmn > nmt, "TMN ({tmn}) > NMT ({nmt}) at band {band}");
        }
    }

    #[test]
    fn test_forward_masking_decays() {
        let model = ZwickerMaskingModel::new();
        let level = 60.0;
        let early = model.forward_masking_decay(level, 1.0);
        let late = model.forward_masking_decay(level, 100.0);
        assert!(
            early > late,
            "Forward masking at 1 ms ({early}) should exceed 100 ms ({late})"
        );
    }

    #[test]
    fn test_backward_masking_is_brief() {
        let model = ZwickerMaskingModel::new();
        let at_2ms = model.backward_masking(60.0, 2.0);
        let at_10ms = model.backward_masking(60.0, 10.0);
        assert!(at_2ms > 0.0, "Backward masking at 2 ms should be > 0");
        assert!(
            at_10ms <= 0.0 || at_10ms < at_2ms,
            "Backward masking at 10 ms should be weaker"
        );
    }

    #[test]
    fn test_global_threshold_above_quiet() {
        let model = ZwickerMaskingModel::new();
        let levels: Vec<(f64, bool)> =
            (0..NUM_BARK_BANDS).map(|_| (60.0, false)).collect();
        let threshold = model.global_masking_threshold(&levels);
        let tiq = model.threshold_in_quiet_bands();
        for i in 0..NUM_BARK_BANDS {
            assert!(
                threshold[i] >= tiq[i] - 1.0,
                "Global threshold band {i} ({}) >= quiet ({})",
                threshold[i],
                tiq[i]
            );
        }
    }

    #[test]
    fn test_excitation_peak_near_masker() {
        let model = ZwickerMaskingModel::new();
        let mut spec = vec![-120.0; NUM_BARK_BANDS];
        spec[10] = 80.0;
        let excitation = model.excitation_pattern(&spec);
        let peak_band = excitation
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            (peak_band as i32 - 10).abs() <= 1,
            "Excitation peak ({peak_band}) should be near band 10"
        );
    }

    #[test]
    fn test_specific_loudness_nonneg() {
        let model = ZwickerMaskingModel::new();
        let spec: Vec<f64> = (0..NUM_BARK_BANDS).map(|_| 50.0).collect();
        let excitation = model.excitation_pattern(&spec);
        let sl = model.specific_loudness(&excitation);
        for (i, &v) in sl.iter().enumerate() {
            assert!(v >= 0.0, "Specific loudness band {i} negative: {v}");
        }
    }

    #[test]
    fn test_masking_analyzer_finds_masked() {
        let signal = vec![30.0; NUM_BARK_BANDS];
        let threshold = vec![50.0; NUM_BARK_BANDS];
        let analyzer = MaskingAnalyzer::new(ZwickerMaskingModel::new());
        let masked = analyzer.identify_masked_regions(&signal, &threshold);
        assert_eq!(masked.len(), NUM_BARK_BANDS);
    }

    #[test]
    fn test_spectral_matrix_dominant_masker() {
        let mut matrix = SpectralMaskingMatrix::new();
        let mut levels = vec![-120.0; NUM_BARK_BANDS];
        levels[5] = 70.0;
        matrix.compute(&levels);
        assert_eq!(matrix.dominant_masker(5), 5);
    }

    #[test]
    fn test_aggregate_to_bark() {
        let freqs = vec![500.0, 1000.0, 2000.0, 4000.0];
        let mags = vec![1.0, 1.0, 1.0, 1.0];
        let levels = aggregate_to_bark_bands(&freqs, &mags);
        assert_eq!(levels.len(), NUM_BARK_BANDS);
        let max_level = levels.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_level > -100.0, "Should have some non-zero energy");
    }

    #[test]
    fn test_full_masking_report() {
        let signal: Vec<f64> = (0..NUM_BARK_BANDS).map(|_| 40.0).collect();
        let bg: Vec<(f64, bool)> =
            (0..NUM_BARK_BANDS).map(|_| (60.0, false)).collect();
        let report = full_masking_report(&signal, &bg);
        assert_eq!(report.global_threshold.len(), NUM_BARK_BANDS);
        assert!(report.total_loudness_sone >= 0.0);
    }

    #[test]
    fn test_estimate_tonality() {
        let mut levels = vec![50.0; NUM_BARK_BANDS];
        levels[10] = 65.0;
        let tonal = estimate_tonality(&levels, 7.0);
        assert!(tonal[10], "Band 10 should be tonal (peak)");
        assert!(!tonal[5], "Band 5 should not be tonal (flat)");
    }

    #[test]
    fn test_signal_to_mask_ratio() {
        let signal = vec![70.0, 60.0, 50.0];
        let mask = vec![60.0, 60.0, 60.0];
        let smr = signal_to_mask_ratio(&signal, &mask);
        assert!((smr[0] - 10.0).abs() < 0.01);
        assert!((smr[1] - 0.0).abs() < 0.01);
        assert!((smr[2] - (-10.0)).abs() < 0.01);
    }

    #[test]
    fn test_perceptual_entropy_positive() {
        let signal = vec![80.0; NUM_BARK_BANDS];
        let mask = vec![40.0; NUM_BARK_BANDS];
        let pe = perceptual_entropy(&signal, &mask);
        assert!(pe > 0.0, "PE should be positive when signal > mask");
    }
}
