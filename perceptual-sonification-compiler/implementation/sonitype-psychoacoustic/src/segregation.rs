//! Bregman auditory stream segregation predicates.
//!
//! This module implements a suite of psychoacoustic predicates drawn from
//! Albert Bregman's Auditory Scene Analysis framework.  Each predicate
//! evaluates one perceptual cue (onset synchrony, harmonicity, spectral
//! proximity, common fate, spatial separation) and returns a
//! [`SegregationResult`] indicating whether two acoustic streams are
//! predicted to segregate or fuse.
//!
//! A higher-level [`StreamSegregationAnalyzer`] combines all predicates,
//! builds a pairwise [`SegregationMatrix`], and can suggest parameter
//! adjustments to achieve a desired segregation outcome.

use serde::{Serialize, Deserialize};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Local type aliases & conversions
// ---------------------------------------------------------------------------

/// Frequency in Hz.
pub type Frequency = f64;

/// Convert a frequency in Hz to the Bark critical-band-rate scale
/// using Traunmüller's formula.
pub fn hz_to_bark(f: f64) -> f64 {
    26.81 * f / (1960.0 + f) - 0.53
}

/// Inverse Bark-to-Hz conversion.
pub fn bark_to_hz(z: f64) -> f64 {
    1960.0 * (z + 0.53) / (26.81 - (z + 0.53))
}

/// Convert a frequency in Hz to the ERB-rate scale (Glasberg & Moore 1990).
pub fn hz_to_erb(f: f64) -> f64 {
    21.4 * (1.0 + 0.00437 * f).log10()
}

/// Convert a frequency in Hz to a Mel value (O'Shaughnessy 1987).
pub fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

/// Clamp a value to the range [lo, hi].
fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Linear interpolation.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Sigmoid mapping into [0, 1].  Useful for converting unbounded margins
/// into confidence values.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Gaussian weighting function.
fn gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    (-0.5 * ((x - mu) / sigma).powi(2)).exp()
}

// ---------------------------------------------------------------------------
// SegregationResult
// ---------------------------------------------------------------------------

/// Outcome of evaluating a single Bregman segregation predicate.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SegregationResult {
    /// `true` when the predicate predicts the two streams will segregate.
    pub satisfied: bool,
    /// How far the measurement exceeds (positive) or falls short of
    /// (negative) the segregation threshold.
    pub margin: f64,
    /// Confidence in the result on a 0.0–1.0 scale.
    pub confidence: f64,
    /// Machine-readable name of the predicate that was evaluated.
    pub predicate_name: String,
    /// Human-readable explanation of the outcome.
    pub details: String,
}

impl SegregationResult {
    /// Construct a result indicating *segregation*.
    pub fn segregated(
        margin: f64,
        confidence: f64,
        name: impl Into<String>,
        details: impl Into<String>,
    ) -> Self {
        Self {
            satisfied: true,
            margin,
            confidence: clamp(confidence, 0.0, 1.0),
            predicate_name: name.into(),
            details: details.into(),
        }
    }

    /// Construct a result indicating *fusion* (no segregation).
    pub fn fused(
        margin: f64,
        confidence: f64,
        name: impl Into<String>,
        details: impl Into<String>,
    ) -> Self {
        Self {
            satisfied: false,
            margin,
            confidence: clamp(confidence, 0.0, 1.0),
            predicate_name: name.into(),
            details: details.into(),
        }
    }

    /// Convenience: is this a segregation result?
    pub fn is_segregated(&self) -> bool {
        self.satisfied
    }

    /// Convenience: is this a fusion result?
    pub fn is_fused(&self) -> bool {
        !self.satisfied
    }

    /// Return the absolute margin (always non-negative).
    pub fn abs_margin(&self) -> f64 {
        self.margin.abs()
    }
}

// ---------------------------------------------------------------------------
// OnsetSynchronyPredicate
// ---------------------------------------------------------------------------

/// Predicate based on onset-time asynchrony.
///
/// Bregman (1990) established that partials starting within ~30-50 ms of
/// one another are grouped into the same auditory stream, while wider
/// onset disparities lead to segregation.
#[derive(Debug, Clone)]
pub struct OnsetSynchronyPredicate {
    /// Maximum onset difference (ms) that still permits fusion.
    threshold_ms: f64,
}

impl OnsetSynchronyPredicate {
    /// Create with an explicit threshold in milliseconds.
    pub fn new(threshold_ms: f64) -> Self {
        Self { threshold_ms }
    }

    /// Create with the literature-standard 40 ms threshold.
    pub fn default_threshold() -> Self {
        Self { threshold_ms: 40.0 }
    }

    /// Evaluate whether two onset times predict segregation.
    pub fn evaluate(&self, onset1_ms: f64, onset2_ms: f64) -> SegregationResult {
        let diff = (onset1_ms - onset2_ms).abs();
        let margin = diff - self.threshold_ms;
        // Confidence rises with distance from the threshold.
        let confidence = sigmoid(margin / (self.threshold_ms * 0.5));

        if margin > 0.0 {
            SegregationResult::segregated(
                margin,
                confidence,
                "onset_synchrony",
                format!(
                    "Onset asynchrony {:.2} ms exceeds threshold {:.2} ms → segregated",
                    diff, self.threshold_ms
                ),
            )
        } else {
            SegregationResult::fused(
                margin,
                1.0 - confidence,
                "onset_synchrony",
                format!(
                    "Onset asynchrony {:.2} ms within threshold {:.2} ms → fused",
                    diff, self.threshold_ms
                ),
            )
        }
    }

    /// Pairwise evaluation over two onset sequences of possibly different
    /// lengths.  The shorter sequence determines how many pairs are compared.
    pub fn evaluate_sequence(
        &self,
        onsets1: &[f64],
        onsets2: &[f64],
    ) -> Vec<SegregationResult> {
        let n = onsets1.len().min(onsets2.len());
        (0..n)
            .map(|i| self.evaluate(onsets1[i], onsets2[i]))
            .collect()
    }

    /// Adapt the threshold based on the temporal context (tempo).
    ///
    /// At faster tempi the auditory system operates with a narrower grouping
    /// window.  A simple linear scaling is applied: the threshold is halved
    /// when the inter-onset interval drops to 100 ms and doubled when it
    /// reaches 800 ms.
    pub fn adapt_threshold(&mut self, temporal_context_ms: f64) {
        let reference_ioi_ms = 400.0; // moderate tempo reference
        let ratio = temporal_context_ms / reference_ioi_ms;
        // Clamp the scaling factor to a reasonable range [0.5, 2.0].
        let scale = clamp(ratio, 0.5, 2.0);
        self.threshold_ms = 40.0 * scale;
    }

    /// Return the current threshold in ms.
    pub fn threshold(&self) -> f64 {
        self.threshold_ms
    }
}

// ---------------------------------------------------------------------------
// HarmonicityPredicate
// ---------------------------------------------------------------------------

/// Predicate based on harmonic relations among spectral components.
///
/// Components forming a harmonic series (integer multiples of a fundamental)
/// fuse into a single stream.  Components belonging to *different* harmonic
/// series segregate.
#[derive(Debug, Clone)]
pub struct HarmonicityPredicate {
    /// Fractional tolerance for matching a frequency to a harmonic.
    tolerance: f64,
}

impl HarmonicityPredicate {
    /// Create with an explicit tolerance (e.g. 0.02 = 2 %).
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Default tolerance of 2 %.
    pub fn default_tolerance() -> Self {
        Self { tolerance: 0.02 }
    }

    /// Compute how well `frequencies` fit a harmonic series at `f0`.
    ///
    /// Returns a score in [0, 1] where 1 means every frequency is an exact
    /// integer multiple of `f0`.
    pub fn harmonicity_score(&self, frequencies: &[f64], f0: f64) -> f64 {
        if frequencies.is_empty() || f0 <= 0.0 {
            return 0.0;
        }
        let mut total_dev = 0.0;
        for &freq in frequencies {
            if freq <= 0.0 {
                total_dev += 1.0;
                continue;
            }
            // Nearest harmonic number.
            let n = (freq / f0).round().max(1.0);
            let ideal = n * f0;
            let deviation = (freq - ideal).abs() / ideal;
            total_dev += deviation;
        }
        let avg_dev = total_dev / frequencies.len() as f64;
        // Map deviation to [0, 1] score: 0 deviation → 1.0, large → ~0.
        let score = (-avg_dev / self.tolerance).exp();
        clamp(score, 0.0, 1.0)
    }

    /// Search for the fundamental frequency that best explains the given
    /// set of partials.
    ///
    /// Returns `(best_f0, best_score)`.  The search grid covers sub-harmonics
    /// of the lowest frequency and refines around the best candidate.
    pub fn find_best_f0(&self, frequencies: &[f64]) -> (f64, f64) {
        if frequencies.is_empty() {
            return (0.0, 0.0);
        }
        let min_freq = frequencies
            .iter()
            .copied()
            .filter(|&f| f > 0.0)
            .fold(f64::MAX, f64::min);
        if min_freq == f64::MAX {
            return (0.0, 0.0);
        }

        let mut best_f0 = min_freq;
        let mut best_score = 0.0_f64;

        // Coarse search: try f_min / n for n = 1..16
        for n in 1..=16 {
            let candidate = min_freq / n as f64;
            if candidate < 20.0 {
                break;
            }
            let score = self.harmonicity_score(frequencies, candidate);
            if score > best_score {
                best_score = score;
                best_f0 = candidate;
            }
        }

        // Also try GCD-based candidates from pairs of frequencies.
        let len = frequencies.len().min(8);
        for i in 0..len {
            for j in (i + 1)..len {
                let f_a = frequencies[i];
                let f_b = frequencies[j];
                if f_a <= 0.0 || f_b <= 0.0 {
                    continue;
                }
                let ratio = f_b / f_a;
                // If ratio is near a simple integer ratio p/q, candidate f0 = f_a / q.
                for q in 1..=8 {
                    let p = (ratio * q as f64).round() as u32;
                    if p == 0 || p > 32 {
                        continue;
                    }
                    let candidate = f_a / q as f64;
                    if candidate < 20.0 || candidate > 5000.0 {
                        continue;
                    }
                    let score = self.harmonicity_score(frequencies, candidate);
                    if score > best_score {
                        best_score = score;
                        best_f0 = candidate;
                    }
                }
            }
        }

        // Fine-grained refinement around the current best.
        let refine_range = best_f0 * 0.05; // ±5 %
        let steps = 40;
        for s in 0..=steps {
            let candidate = best_f0 - refine_range + 2.0 * refine_range * (s as f64 / steps as f64);
            if candidate < 20.0 {
                continue;
            }
            let score = self.harmonicity_score(frequencies, candidate);
            if score > best_score {
                best_score = score;
                best_f0 = candidate;
            }
        }

        (best_f0, best_score)
    }

    /// Evaluate whether two sets of partials belong to different harmonic
    /// series (segregated) or the same one (fused).
    pub fn evaluate(&self, harmonics1: &[f64], harmonics2: &[f64]) -> SegregationResult {
        let (f0_1, score1) = self.find_best_f0(harmonics1);
        let (f0_2, score2) = self.find_best_f0(harmonics2);

        // If neither set is harmonic there is nothing to decide.
        if score1 < 0.3 && score2 < 0.3 {
            return SegregationResult::fused(
                -1.0,
                0.3,
                "harmonicity",
                format!(
                    "Neither set is strongly harmonic (scores {:.2}, {:.2}) → weak fusion default",
                    score1, score2
                ),
            );
        }

        // Check whether the two f0s are essentially the same.
        let f0_diff_ratio = if f0_1 > 0.0 && f0_2 > 0.0 {
            (f0_1 - f0_2).abs() / f0_1.min(f0_2)
        } else {
            1.0
        };

        // Also check whether one f0 is a near-integer multiple of the other.
        let harmonic_relation = self.are_harmonically_related(f0_1, f0_2);

        let combined_quality = (score1 + score2) / 2.0;
        let segregation_threshold = 0.06; // 6 % difference in f0
        let margin = f0_diff_ratio - segregation_threshold;
        let confidence = combined_quality * sigmoid(margin / 0.05);

        if f0_diff_ratio > segregation_threshold && !harmonic_relation {
            SegregationResult::segregated(
                margin,
                confidence,
                "harmonicity",
                format!(
                    "F0s differ by {:.1}% (f0₁={:.1} Hz, f0₂={:.1} Hz) → segregated",
                    f0_diff_ratio * 100.0,
                    f0_1,
                    f0_2
                ),
            )
        } else {
            SegregationResult::fused(
                margin,
                1.0 - confidence,
                "harmonicity",
                format!(
                    "F0s similar or harmonically related ({:.1} Hz vs {:.1} Hz, diff {:.1}%) → fused",
                    f0_1,
                    f0_2,
                    f0_diff_ratio * 100.0
                ),
            )
        }
    }

    /// Return `true` when `f2 / f1` is close to a small integer ratio.
    pub fn are_harmonically_related(&self, f1: f64, f2: f64) -> bool {
        if f1 <= 0.0 || f2 <= 0.0 {
            return false;
        }
        let ratio = if f2 > f1 { f2 / f1 } else { f1 / f2 };

        // Check against small integer ratios p/q with p,q ≤ 8.
        for q in 1..=8u32 {
            for p in q..=8u32 {
                let ideal = p as f64 / q as f64;
                if (ratio - ideal).abs() < self.tolerance * ideal {
                    return true;
                }
            }
        }
        false
    }

    /// Return the current tolerance.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }
}

// ---------------------------------------------------------------------------
// SpectralProximityPredicate
// ---------------------------------------------------------------------------

/// Predicate based on spectral proximity on the Bark scale.
///
/// Components whose spectral centroids lie within about 1.5 Bark of each
/// other tend to fuse; greater separations promote segregation.
#[derive(Debug, Clone)]
pub struct SpectralProximityPredicate {
    /// Separation in Bark above which segregation is predicted.
    threshold_bark: f64,
}

impl SpectralProximityPredicate {
    /// Create with an explicit Bark threshold.
    pub fn new(threshold_bark: f64) -> Self {
        Self { threshold_bark }
    }

    /// Default 1.5-Bark threshold.
    pub fn default_threshold() -> Self {
        Self {
            threshold_bark: 1.5,
        }
    }

    /// Evaluate using Hz-valued spectral centroids (internally converted to
    /// Bark).
    pub fn evaluate(&self, centroid1_hz: f64, centroid2_hz: f64) -> SegregationResult {
        let b1 = hz_to_bark(centroid1_hz);
        let b2 = hz_to_bark(centroid2_hz);
        self.evaluate_bark(b1, b2)
    }

    /// Evaluate using Bark-valued spectral centroids directly.
    pub fn evaluate_bark(
        &self,
        centroid1_bark: f64,
        centroid2_bark: f64,
    ) -> SegregationResult {
        let center = (centroid1_bark + centroid2_bark) / 2.0;
        let threshold = self.frequency_dependent_threshold(center);
        let diff = (centroid1_bark - centroid2_bark).abs();
        let margin = diff - threshold;
        let confidence = sigmoid(margin / (threshold * 0.5));

        if margin > 0.0 {
            SegregationResult::segregated(
                margin,
                confidence,
                "spectral_proximity",
                format!(
                    "Bark separation {:.2} exceeds threshold {:.2} → segregated",
                    diff, threshold
                ),
            )
        } else {
            SegregationResult::fused(
                margin,
                1.0 - confidence,
                "spectral_proximity",
                format!(
                    "Bark separation {:.2} within threshold {:.2} → fused",
                    diff, threshold
                ),
            )
        }
    }

    /// Return a frequency-dependent threshold.
    ///
    /// At low Bark values (< 4) critical bands are wider so the threshold
    /// is slightly relaxed; at high Bark values (> 20) bands narrow and the
    /// threshold tightens.
    pub fn frequency_dependent_threshold(&self, center_bark: f64) -> f64 {
        let low_scale = if center_bark < 4.0 {
            lerp(1.3, 1.0, center_bark / 4.0)
        } else {
            1.0
        };
        let high_scale = if center_bark > 20.0 {
            lerp(1.0, 0.8, (center_bark - 20.0) / 5.0)
        } else {
            1.0
        };
        self.threshold_bark * low_scale * high_scale
    }

    /// Return the base threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold_bark
    }
}

// ---------------------------------------------------------------------------
// CommonFatePredicate
// ---------------------------------------------------------------------------

/// Predicate based on the *common fate* principle.
///
/// Components that are amplitude-modulated (AM) or frequency-modulated (FM)
/// at the same rate are grouped together.  Different modulation rates promote
/// segregation.
#[derive(Debug, Clone)]
pub struct CommonFatePredicate {
    /// Minimum AM-rate difference (Hz) to segregate.
    am_threshold: f64,
    /// Minimum FM-rate difference (Hz) to segregate.
    fm_threshold: f64,
}

impl CommonFatePredicate {
    /// Create with explicit AM and FM thresholds.
    pub fn new(am_threshold: f64, fm_threshold: f64) -> Self {
        Self {
            am_threshold,
            fm_threshold,
        }
    }

    /// Default thresholds (1.0 Hz each).
    pub fn default_thresholds() -> Self {
        Self {
            am_threshold: 1.0,
            fm_threshold: 1.0,
        }
    }

    /// Evaluate using both AM and FM rate differences.
    ///
    /// Segregation is predicted if *either* AM or FM rates differ by more
    /// than their respective thresholds.
    pub fn evaluate(
        &self,
        am_rate1: f64,
        am_rate2: f64,
        fm_rate1: f64,
        fm_rate2: f64,
    ) -> SegregationResult {
        let am_diff = (am_rate1 - am_rate2).abs();
        let fm_diff = (fm_rate1 - fm_rate2).abs();

        let am_margin = am_diff - self.am_threshold;
        let fm_margin = fm_diff - self.fm_threshold;

        // Use the *maximum* margin — either modulation dimension can drive
        // segregation.
        let effective_margin = am_margin.max(fm_margin);
        let am_conf = sigmoid(am_margin / (self.am_threshold * 0.5));
        let fm_conf = sigmoid(fm_margin / (self.fm_threshold * 0.5));
        let confidence = am_conf.max(fm_conf);

        if effective_margin > 0.0 {
            let dimension = if am_margin >= fm_margin { "AM" } else { "FM" };
            SegregationResult::segregated(
                effective_margin,
                confidence,
                "common_fate",
                format!(
                    "{} rate diff ({:.2} Hz) exceeds threshold → segregated (AM Δ={:.2}, FM Δ={:.2})",
                    dimension,
                    if am_margin >= fm_margin { am_diff } else { fm_diff },
                    am_diff,
                    fm_diff
                ),
            )
        } else {
            SegregationResult::fused(
                effective_margin,
                1.0 - confidence,
                "common_fate",
                format!(
                    "AM Δ={:.2} Hz, FM Δ={:.2} Hz — both within thresholds → fused",
                    am_diff, fm_diff
                ),
            )
        }
    }

    /// Evaluate using AM rates only (FM information unavailable).
    pub fn evaluate_am_only(&self, am_rate1: f64, am_rate2: f64) -> SegregationResult {
        let diff = (am_rate1 - am_rate2).abs();
        let margin = diff - self.am_threshold;
        let confidence = sigmoid(margin / (self.am_threshold * 0.5));

        if margin > 0.0 {
            SegregationResult::segregated(
                margin,
                confidence,
                "common_fate_am",
                format!(
                    "AM rate diff {:.2} Hz exceeds threshold {:.2} Hz → segregated",
                    diff, self.am_threshold
                ),
            )
        } else {
            SegregationResult::fused(
                margin,
                1.0 - confidence,
                "common_fate_am",
                format!(
                    "AM rate diff {:.2} Hz within threshold {:.2} Hz → fused",
                    diff, self.am_threshold
                ),
            )
        }
    }

    /// Evaluate using FM rates only (AM information unavailable).
    pub fn evaluate_fm_only(&self, fm_rate1: f64, fm_rate2: f64) -> SegregationResult {
        let diff = (fm_rate1 - fm_rate2).abs();
        let margin = diff - self.fm_threshold;
        let confidence = sigmoid(margin / (self.fm_threshold * 0.5));

        if margin > 0.0 {
            SegregationResult::segregated(
                margin,
                confidence,
                "common_fate_fm",
                format!(
                    "FM rate diff {:.2} Hz exceeds threshold {:.2} Hz → segregated",
                    diff, self.fm_threshold
                ),
            )
        } else {
            SegregationResult::fused(
                margin,
                1.0 - confidence,
                "common_fate_fm",
                format!(
                    "FM rate diff {:.2} Hz within threshold {:.2} Hz → fused",
                    diff, self.fm_threshold
                ),
            )
        }
    }

    /// Compute a coherence score for a group of components.
    ///
    /// The score is in [0, 1] where 1.0 means all components modulate at
    /// exactly the same rate (strong grouping).  Coherence is measured as
    /// the inverse of the mean pairwise rate difference normalised by the
    /// respective thresholds.
    pub fn coherence_score(&self, am_rates: &[f64], fm_rates: &[f64]) -> f64 {
        let n = am_rates.len().min(fm_rates.len());
        if n <= 1 {
            return 1.0;
        }

        let mut total_am_diff = 0.0;
        let mut total_fm_diff = 0.0;
        let mut pairs = 0u64;

        for i in 0..n {
            for j in (i + 1)..n {
                total_am_diff += (am_rates[i] - am_rates[j]).abs();
                total_fm_diff += (fm_rates[i] - fm_rates[j]).abs();
                pairs += 1;
            }
        }

        if pairs == 0 {
            return 1.0;
        }

        let mean_am = total_am_diff / pairs as f64;
        let mean_fm = total_fm_diff / pairs as f64;

        let normalised = (mean_am / self.am_threshold + mean_fm / self.fm_threshold) / 2.0;
        let score = (-normalised).exp();
        clamp(score, 0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// SpatialSeparationPredicate
// ---------------------------------------------------------------------------

/// Predicate based on spatial (azimuthal) separation.
///
/// Sources perceived at different azimuths are more likely to segregate.
/// The threshold defaults to ~10° for typical listening conditions.
#[derive(Debug, Clone)]
pub struct SpatialSeparationPredicate {
    /// Minimum azimuthal separation (degrees) for segregation.
    azimuth_threshold_deg: f64,
}

impl SpatialSeparationPredicate {
    /// Create with an explicit azimuth threshold in degrees.
    pub fn new(azimuth_threshold: f64) -> Self {
        Self {
            azimuth_threshold_deg: azimuth_threshold,
        }
    }

    /// Default 10° threshold.
    pub fn default_threshold() -> Self {
        Self {
            azimuth_threshold_deg: 10.0,
        }
    }

    /// Evaluate azimuthal separation.
    pub fn evaluate(&self, azimuth1: f64, azimuth2: f64) -> SegregationResult {
        // Wrap-aware angular difference in [-180, 180].
        let mut diff = (azimuth1 - azimuth2) % 360.0;
        if diff > 180.0 {
            diff -= 360.0;
        }
        if diff < -180.0 {
            diff += 360.0;
        }
        let diff = diff.abs();

        let margin = diff - self.azimuth_threshold_deg;
        let confidence = sigmoid(margin / (self.azimuth_threshold_deg * 0.5));

        if margin > 0.0 {
            SegregationResult::segregated(
                margin,
                confidence,
                "spatial_separation",
                format!(
                    "Azimuth difference {:.1}° exceeds threshold {:.1}° → segregated",
                    diff, self.azimuth_threshold_deg
                ),
            )
        } else {
            SegregationResult::fused(
                margin,
                1.0 - confidence,
                "spatial_separation",
                format!(
                    "Azimuth difference {:.1}° within threshold {:.1}° → fused",
                    diff, self.azimuth_threshold_deg
                ),
            )
        }
    }

    /// Return the threshold in degrees.
    pub fn threshold(&self) -> f64 {
        self.azimuth_threshold_deg
    }

    /// Evaluate with an additional elevation component.
    /// Combined angular distance: sqrt(Δaz² + Δel²).
    pub fn evaluate_with_elevation(
        &self,
        azimuth1: f64,
        elevation1: f64,
        azimuth2: f64,
        elevation2: f64,
    ) -> SegregationResult {
        let mut az_diff = (azimuth1 - azimuth2) % 360.0;
        if az_diff > 180.0 {
            az_diff -= 360.0;
        }
        if az_diff < -180.0 {
            az_diff += 360.0;
        }
        let el_diff = elevation1 - elevation2;
        let angular_dist = (az_diff * az_diff + el_diff * el_diff).sqrt();

        let margin = angular_dist - self.azimuth_threshold_deg;
        let confidence = sigmoid(margin / (self.azimuth_threshold_deg * 0.5));

        if margin > 0.0 {
            SegregationResult::segregated(
                margin,
                confidence,
                "spatial_separation_3d",
                format!(
                    "3-D angular distance {:.1}° exceeds threshold {:.1}° → segregated",
                    angular_dist, self.azimuth_threshold_deg
                ),
            )
        } else {
            SegregationResult::fused(
                margin,
                1.0 - confidence,
                "spatial_separation_3d",
                format!(
                    "3-D angular distance {:.1}° within threshold {:.1}° → fused",
                    angular_dist, self.azimuth_threshold_deg
                ),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// StreamDescriptor
// ---------------------------------------------------------------------------

/// Complete description of a single auditory stream (or source) sufficient
/// for evaluating all Bregman segregation predicates.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StreamDescriptor {
    /// Unique identifier within the current analysis.
    pub id: usize,
    /// Onset time in milliseconds.
    pub onset_ms: f64,
    /// Estimated fundamental frequency (Hz).
    pub fundamental_freq: f64,
    /// Frequencies of resolved harmonics / partials (Hz).
    pub harmonics: Vec<f64>,
    /// Spectral centroid (Hz).
    pub spectral_centroid_hz: f64,
    /// Amplitude-modulation rate (Hz).
    pub am_rate: f64,
    /// Frequency-modulation rate (Hz).
    pub fm_rate: f64,
    /// Azimuth in the horizontal plane (degrees, −180 to 180).
    pub azimuth_deg: f64,
    /// Level in dB SPL (informational, not used by predicates directly).
    pub level_db: f64,
}

impl StreamDescriptor {
    /// Convenience builder with all fields explicit.
    pub fn new(
        id: usize,
        onset_ms: f64,
        fundamental_freq: f64,
        harmonics: Vec<f64>,
        spectral_centroid_hz: f64,
        am_rate: f64,
        fm_rate: f64,
        azimuth_deg: f64,
        level_db: f64,
    ) -> Self {
        Self {
            id,
            onset_ms,
            fundamental_freq,
            harmonics,
            spectral_centroid_hz,
            am_rate,
            fm_rate,
            azimuth_deg,
            level_db,
        }
    }

    /// Return the spectral centroid on the Bark scale.
    pub fn spectral_centroid_bark(&self) -> f64 {
        hz_to_bark(self.spectral_centroid_hz)
    }

    /// Build a minimal stream with only an onset and fundamental.
    pub fn minimal(id: usize, onset_ms: f64, fundamental_freq: f64) -> Self {
        let harmonics: Vec<f64> = (1..=6).map(|n| fundamental_freq * n as f64).collect();
        let centroid = harmonics.iter().sum::<f64>() / harmonics.len() as f64;
        Self {
            id,
            onset_ms,
            fundamental_freq,
            harmonics,
            spectral_centroid_hz: centroid,
            am_rate: 0.0,
            fm_rate: 0.0,
            azimuth_deg: 0.0,
            level_db: 60.0,
        }
    }
}

// ---------------------------------------------------------------------------
// PairwiseSegregation & SegregationMatrix
// ---------------------------------------------------------------------------

/// Result of comparing a single pair of streams across all predicates.
#[derive(Debug, Clone)]
pub struct PairwiseSegregation {
    pub onset_result: SegregationResult,
    pub harmonicity_result: SegregationResult,
    pub spectral_result: SegregationResult,
    pub common_fate_result: SegregationResult,
    pub spatial_result: SegregationResult,
    /// Combined verdict.
    pub overall_segregated: bool,
    /// Combined confidence.
    pub overall_confidence: f64,
}

impl PairwiseSegregation {
    /// Return a Vec of references to all five sub-results.
    pub fn all_results(&self) -> Vec<&SegregationResult> {
        vec![
            &self.onset_result,
            &self.harmonicity_result,
            &self.spectral_result,
            &self.common_fate_result,
            &self.spatial_result,
        ]
    }

    /// Number of predicates that predict segregation.
    pub fn segregation_votes(&self) -> usize {
        self.all_results().iter().filter(|r| r.satisfied).count()
    }

    /// Number of predicates that predict fusion.
    pub fn fusion_votes(&self) -> usize {
        self.all_results().iter().filter(|r| !r.satisfied).count()
    }

    /// Mean confidence across all predicates.
    pub fn mean_confidence(&self) -> f64 {
        let results = self.all_results();
        let sum: f64 = results.iter().map(|r| r.confidence).sum();
        sum / results.len() as f64
    }

    /// Return the names of predicates that contributed to the *minority*
    /// vote (i.e. conflicting evidence).
    pub fn conflicting_predicates(&self) -> Vec<String> {
        let majority_segregated = self.overall_segregated;
        self.all_results()
            .iter()
            .filter(|r| r.satisfied != majority_segregated)
            .map(|r| r.predicate_name.clone())
            .collect()
    }
}

/// k×k matrix storing pairwise segregation results for k streams.
#[derive(Debug, Clone)]
pub struct SegregationMatrix {
    size: usize,
    results: Vec<Vec<Option<PairwiseSegregation>>>,
}

impl SegregationMatrix {
    /// Create an empty k×k matrix.
    pub fn new(k: usize) -> Self {
        let results = (0..k)
            .map(|_| (0..k).map(|_| None).collect())
            .collect();
        Self { size: k, results }
    }

    /// Store a result for pair (i, j).  Also stores the same result at (j, i)
    /// for symmetry.
    pub fn set(&mut self, i: usize, j: usize, result: PairwiseSegregation) {
        if i < self.size && j < self.size {
            self.results[i][j] = Some(result.clone());
            self.results[j][i] = Some(result);
        }
    }

    /// Retrieve the result for pair (i, j), if computed.
    pub fn get(&self, i: usize, j: usize) -> Option<&PairwiseSegregation> {
        if i < self.size && j < self.size {
            self.results[i][j].as_ref()
        } else {
            None
        }
    }

    /// Is the pair (i, j) predicted to segregate?
    pub fn is_segregated(&self, i: usize, j: usize) -> bool {
        self.get(i, j).map_or(false, |p| p.overall_segregated)
    }

    /// Are *all* off-diagonal pairs predicted to segregate?
    pub fn all_pairs_segregated(&self) -> bool {
        for i in 0..self.size {
            for j in (i + 1)..self.size {
                if !self.is_segregated(i, j) {
                    return false;
                }
            }
        }
        true
    }

    /// Return indices of pairs where the overall verdict is *fusion*
    /// (these are the pairs that might need parameter adjustment).
    pub fn conflicting_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..self.size {
            for j in (i + 1)..self.size {
                if let Some(p) = self.get(i, j) {
                    if !p.overall_segregated {
                        pairs.push((i, j));
                    }
                }
            }
        }
        pairs
    }

    /// Return the dimensionality of the matrix.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Fraction of pairs that are segregated.
    pub fn segregation_ratio(&self) -> f64 {
        let total = self.size * (self.size.saturating_sub(1)) / 2;
        if total == 0 {
            return 1.0;
        }
        let segregated_count = (0..self.size)
            .flat_map(|i| ((i + 1)..self.size).map(move |j| (i, j)))
            .filter(|&(i, j)| self.is_segregated(i, j))
            .count();
        segregated_count as f64 / total as f64
    }

    /// Mean confidence across all computed pairs.
    pub fn mean_confidence(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for i in 0..self.size {
            for j in (i + 1)..self.size {
                if let Some(p) = self.get(i, j) {
                    sum += p.overall_confidence;
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

// ---------------------------------------------------------------------------
// AdjustmentSuggestion
// ---------------------------------------------------------------------------

/// A suggestion for changing one parameter of a stream to improve
/// segregation.
#[derive(Debug, Clone)]
pub struct AdjustmentSuggestion {
    /// Name of the parameter to adjust (e.g. "spectral_centroid_hz").
    pub parameter: String,
    /// Which stream should be changed.
    pub stream_id: usize,
    /// Current value of the parameter.
    pub current_value: f64,
    /// Suggested new value.
    pub suggested_value: f64,
    /// Estimated improvement in the segregation margin.
    pub expected_improvement: f64,
}

impl AdjustmentSuggestion {
    /// Build a new suggestion.
    pub fn new(
        parameter: impl Into<String>,
        stream_id: usize,
        current_value: f64,
        suggested_value: f64,
        expected_improvement: f64,
    ) -> Self {
        Self {
            parameter: parameter.into(),
            stream_id,
            current_value,
            suggested_value,
            expected_improvement,
        }
    }

    /// Relative change magnitude.
    pub fn relative_change(&self) -> f64 {
        if self.current_value.abs() < 1e-12 {
            self.suggested_value.abs()
        } else {
            (self.suggested_value - self.current_value).abs() / self.current_value.abs()
        }
    }
}

// ---------------------------------------------------------------------------
// StreamSegregationAnalyzer
// ---------------------------------------------------------------------------

/// High-level analyser that combines all Bregman segregation predicates.
///
/// Given a set of [`StreamDescriptor`]s it produces a [`SegregationMatrix`]
/// and can identify conflicting (fused) pairs together with actionable
/// parameter-adjustment suggestions.
#[derive(Debug, Clone)]
pub struct StreamSegregationAnalyzer {
    onset_pred: OnsetSynchronyPredicate,
    harmonicity_pred: HarmonicityPredicate,
    spectral_pred: SpectralProximityPredicate,
    common_fate_pred: CommonFatePredicate,
    spatial_pred: SpatialSeparationPredicate,
    /// Weights for combining predicate results (summed, not necessarily
    /// normalised to 1).
    predicate_weights: [f64; 5],
}

impl StreamSegregationAnalyzer {
    /// Create with all-default thresholds and uniform predicate weights.
    pub fn new() -> Self {
        Self {
            onset_pred: OnsetSynchronyPredicate::default_threshold(),
            harmonicity_pred: HarmonicityPredicate::default_tolerance(),
            spectral_pred: SpectralProximityPredicate::default_threshold(),
            common_fate_pred: CommonFatePredicate::default_thresholds(),
            spatial_pred: SpatialSeparationPredicate::default_threshold(),
            predicate_weights: [1.0; 5],
        }
    }

    /// Create with fully specified thresholds.
    pub fn with_config(
        onset_thresh: f64,
        spectral_thresh: f64,
        am_thresh: f64,
        fm_thresh: f64,
        harmonicity_tol: f64,
        spatial_thresh: f64,
    ) -> Self {
        Self {
            onset_pred: OnsetSynchronyPredicate::new(onset_thresh),
            harmonicity_pred: HarmonicityPredicate::new(harmonicity_tol),
            spectral_pred: SpectralProximityPredicate::new(spectral_thresh),
            common_fate_pred: CommonFatePredicate::new(am_thresh, fm_thresh),
            spatial_pred: SpatialSeparationPredicate::new(spatial_thresh),
            predicate_weights: [1.0; 5],
        }
    }

    /// Override predicate weights (onset, harmonicity, spectral,
    /// common_fate, spatial).
    pub fn set_weights(&mut self, weights: [f64; 5]) {
        self.predicate_weights = weights;
    }

    /// Analyse a single pair of streams.
    pub fn analyze_pair(
        &self,
        s1: &StreamDescriptor,
        s2: &StreamDescriptor,
    ) -> PairwiseSegregation {
        let onset_result = self.onset_pred.evaluate(s1.onset_ms, s2.onset_ms);
        let harmonicity_result =
            self.harmonicity_pred.evaluate(&s1.harmonics, &s2.harmonics);
        let spectral_result = self
            .spectral_pred
            .evaluate(s1.spectral_centroid_hz, s2.spectral_centroid_hz);
        let common_fate_result = self.common_fate_pred.evaluate(
            s1.am_rate, s2.am_rate, s1.fm_rate, s2.fm_rate,
        );
        let spatial_result = self
            .spatial_pred
            .evaluate(s1.azimuth_deg, s2.azimuth_deg);

        // Weighted vote: each predicate contributes its confidence *
        // weight towards segregation or fusion.
        let results = [
            &onset_result,
            &harmonicity_result,
            &spectral_result,
            &common_fate_result,
            &spatial_result,
        ];
        let mut seg_score = 0.0;
        let mut fus_score = 0.0;
        for (r, &w) in results.iter().zip(self.predicate_weights.iter()) {
            if r.satisfied {
                seg_score += r.confidence * w;
            } else {
                fus_score += r.confidence * w;
            }
        }
        let total = seg_score + fus_score;
        let overall_confidence = if total > 0.0 {
            seg_score.max(fus_score) / total
        } else {
            0.5
        };
        let overall_segregated = seg_score > fus_score;

        PairwiseSegregation {
            onset_result,
            harmonicity_result,
            spectral_result,
            common_fate_result,
            spatial_result,
            overall_segregated,
            overall_confidence,
        }
    }

    /// Run pairwise analysis on all streams, producing a [`SegregationMatrix`].
    pub fn check_all_pairs(
        &self,
        streams: &[StreamDescriptor],
    ) -> SegregationMatrix {
        let k = streams.len();
        let mut matrix = SegregationMatrix::new(k);
        for i in 0..k {
            for j in (i + 1)..k {
                let result = self.analyze_pair(&streams[i], &streams[j]);
                matrix.set(i, j, result);
            }
        }
        matrix
    }

    /// Return indices of pairs that are *not* segregated.
    pub fn identify_conflicting_pairs(
        &self,
        streams: &[StreamDescriptor],
    ) -> Vec<(usize, usize)> {
        self.check_all_pairs(streams).conflicting_pairs()
    }

    /// Suggest parameter adjustments to turn a fused pair into a segregated
    /// pair.
    pub fn suggest_parameter_adjustments(
        &self,
        s1: &StreamDescriptor,
        s2: &StreamDescriptor,
        result: &PairwiseSegregation,
    ) -> Vec<AdjustmentSuggestion> {
        let mut suggestions: Vec<AdjustmentSuggestion> = Vec::new();

        // --- Onset ---
        if result.onset_result.is_fused() {
            let current_diff = (s1.onset_ms - s2.onset_ms).abs();
            let target_diff = self.onset_pred.threshold() * 1.5;
            let shift = target_diff - current_diff;
            if shift > 0.0 {
                // Suggest shifting the later stream's onset.
                let target_id = if s1.onset_ms >= s2.onset_ms {
                    s1.id
                } else {
                    s2.id
                };
                let current_onset = if s1.id == target_id {
                    s1.onset_ms
                } else {
                    s2.onset_ms
                };
                suggestions.push(AdjustmentSuggestion::new(
                    "onset_ms",
                    target_id,
                    current_onset,
                    current_onset + shift,
                    shift,
                ));
            }
        }

        // --- Spectral centroid ---
        if result.spectral_result.is_fused() {
            let b1 = hz_to_bark(s1.spectral_centroid_hz);
            let b2 = hz_to_bark(s2.spectral_centroid_hz);
            let current_sep = (b1 - b2).abs();
            let target_sep = self.spectral_pred.threshold() * 1.5;
            let bark_shift = target_sep - current_sep;

            if bark_shift > 0.0 {
                // Shift the higher-centroid stream upward.
                let (target_id, current_hz, current_bark) = if b1 >= b2 {
                    (s1.id, s1.spectral_centroid_hz, b1)
                } else {
                    (s2.id, s2.spectral_centroid_hz, b2)
                };
                let new_bark = current_bark + bark_shift;
                let new_hz = bark_to_hz(new_bark);
                suggestions.push(AdjustmentSuggestion::new(
                    "spectral_centroid_hz",
                    target_id,
                    current_hz,
                    new_hz,
                    bark_shift,
                ));
            }
        }

        // --- AM rate ---
        if result.common_fate_result.is_fused() {
            let am_diff = (s1.am_rate - s2.am_rate).abs();
            let target_diff = self.common_fate_pred.am_threshold * 1.5;
            let shift = target_diff - am_diff;

            if shift > 0.0 {
                let (target_id, current_am) = if s1.am_rate >= s2.am_rate {
                    (s1.id, s1.am_rate)
                } else {
                    (s2.id, s2.am_rate)
                };
                suggestions.push(AdjustmentSuggestion::new(
                    "am_rate",
                    target_id,
                    current_am,
                    current_am + shift,
                    shift,
                ));
            }
        }

        // --- FM rate ---
        if result.common_fate_result.is_fused() {
            let fm_diff = (s1.fm_rate - s2.fm_rate).abs();
            let target_diff = self.common_fate_pred.fm_threshold * 1.5;
            let shift = target_diff - fm_diff;

            if shift > 0.0 {
                let (target_id, current_fm) = if s1.fm_rate >= s2.fm_rate {
                    (s1.id, s1.fm_rate)
                } else {
                    (s2.id, s2.fm_rate)
                };
                suggestions.push(AdjustmentSuggestion::new(
                    "fm_rate",
                    target_id,
                    current_fm,
                    current_fm + shift,
                    shift,
                ));
            }
        }

        // --- Spatial ---
        if result.spatial_result.is_fused() {
            let az_diff = (s1.azimuth_deg - s2.azimuth_deg).abs();
            let target_diff = self.spatial_pred.threshold() * 1.5;
            let shift = target_diff - az_diff;

            if shift > 0.0 {
                let (target_id, current_az) = if s1.azimuth_deg >= s2.azimuth_deg {
                    (s1.id, s1.azimuth_deg)
                } else {
                    (s2.id, s2.azimuth_deg)
                };
                let suggested_az = current_az + shift;
                let suggested_az = if suggested_az > 180.0 {
                    suggested_az - 360.0
                } else {
                    suggested_az
                };
                suggestions.push(AdjustmentSuggestion::new(
                    "azimuth_deg",
                    target_id,
                    current_az,
                    suggested_az,
                    shift,
                ));
            }
        }

        // --- Harmonicity (fundamental frequency) ---
        if result.harmonicity_result.is_fused() {
            let f0_diff_ratio = if s1.fundamental_freq > 0.0 && s2.fundamental_freq > 0.0 {
                (s1.fundamental_freq - s2.fundamental_freq).abs()
                    / s1.fundamental_freq.min(s2.fundamental_freq)
            } else {
                0.0
            };

            if f0_diff_ratio < 0.10 {
                // Suggest increasing the f0 separation.
                let target_ratio = 0.15;
                let min_f0 = s1.fundamental_freq.min(s2.fundamental_freq);
                let new_f0 = min_f0 * (1.0 + target_ratio);
                let (target_id, current_f0) = if s1.fundamental_freq >= s2.fundamental_freq {
                    (s1.id, s1.fundamental_freq)
                } else {
                    (s2.id, s2.fundamental_freq)
                };
                suggestions.push(AdjustmentSuggestion::new(
                    "fundamental_freq",
                    target_id,
                    current_f0,
                    new_f0.max(current_f0),
                    target_ratio - f0_diff_ratio,
                ));
            }
        }

        // Sort by expected improvement (descending).
        suggestions.sort_by(|a, b| {
            b.expected_improvement
                .partial_cmp(&a.expected_improvement)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        suggestions
    }
}

impl Default for StreamSegregationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

impl std::fmt::Display for SegregationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let verdict = if self.satisfied {
            "SEGREGATED"
        } else {
            "FUSED"
        };
        write!(
            f,
            "[{}] {} (margin={:.3}, conf={:.2}): {}",
            self.predicate_name, verdict, self.margin, self.confidence, self.details
        )
    }
}

impl std::fmt::Display for AdjustmentSuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Stream {} — adjust {} from {:.2} → {:.2} (expected improvement {:.3})",
            self.stream_id,
            self.parameter,
            self.current_value,
            self.suggested_value,
            self.expected_improvement
        )
    }
}

impl std::fmt::Display for SegregationMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SegregationMatrix ({}×{}):", self.size, self.size)?;
        for i in 0..self.size {
            for j in (i + 1)..self.size {
                let status = if self.is_segregated(i, j) {
                    "SEG"
                } else {
                    "FUS"
                };
                let conf = self
                    .get(i, j)
                    .map(|p| p.overall_confidence)
                    .unwrap_or(0.0);
                writeln!(f, "  ({}, {}): {} (conf={:.2})", i, j, status, conf)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------ Onset synchrony --------------------------------------------------

    #[test]
    fn test_onset_segregated_with_large_difference() {
        let pred = OnsetSynchronyPredicate::default_threshold();
        let result = pred.evaluate(0.0, 100.0);
        assert!(result.is_segregated(), "100 ms apart should segregate");
        assert!(result.margin > 0.0, "Margin should be positive");
        assert!(
            result.confidence > 0.5,
            "Confidence should be above 0.5 for clear segregation"
        );
    }

    #[test]
    fn test_onset_fused_with_simultaneous_onsets() {
        let pred = OnsetSynchronyPredicate::default_threshold();
        let result = pred.evaluate(50.0, 55.0);
        assert!(result.is_fused(), "5 ms apart should fuse (threshold 40)");
        assert!(result.margin < 0.0, "Margin should be negative");
    }

    #[test]
    fn test_onset_sequence_evaluation() {
        let pred = OnsetSynchronyPredicate::default_threshold();
        let onsets1 = vec![0.0, 200.0, 400.0];
        let onsets2 = vec![5.0, 250.0, 405.0];
        let results = pred.evaluate_sequence(&onsets1, &onsets2);
        assert_eq!(results.len(), 3);
        // First pair: 5 ms → fused
        assert!(results[0].is_fused());
        // Second pair: 50 ms → segregated
        assert!(results[1].is_segregated());
        // Third pair: 5 ms → fused
        assert!(results[2].is_fused());
    }

    #[test]
    fn test_onset_threshold_adaptation() {
        let mut pred = OnsetSynchronyPredicate::default_threshold();
        // Fast tempo (100 ms IOI) → threshold should shrink.
        pred.adapt_threshold(100.0);
        assert!(
            pred.threshold() < 40.0,
            "Fast tempo should shrink threshold, got {}",
            pred.threshold()
        );

        // Slow tempo (800 ms IOI) → threshold should grow.
        pred.adapt_threshold(800.0);
        assert!(
            pred.threshold() > 40.0,
            "Slow tempo should grow threshold, got {}",
            pred.threshold()
        );
    }

    // ------ Harmonicity ------------------------------------------------------

    #[test]
    fn test_harmonicity_perfect_series() {
        let pred = HarmonicityPredicate::default_tolerance();
        let f0 = 200.0;
        let harmonics: Vec<f64> = (1..=8).map(|n| f0 * n as f64).collect();
        let score = pred.harmonicity_score(&harmonics, f0);
        assert!(
            score > 0.95,
            "Perfect harmonic series should score near 1.0, got {}",
            score
        );
    }

    #[test]
    fn test_harmonicity_inharmonic_lower_score() {
        let pred = HarmonicityPredicate::default_tolerance();
        let f0 = 200.0;
        let harmonic: Vec<f64> = (1..=6).map(|n| f0 * n as f64).collect();
        let inharmonic = vec![217.0, 433.0, 689.0, 911.0, 1137.0, 1399.0];

        let harmonic_score = pred.harmonicity_score(&harmonic, f0);
        let inharmonic_score = pred.harmonicity_score(&inharmonic, f0);

        assert!(
            harmonic_score > inharmonic_score,
            "Harmonic set ({:.3}) should score higher than inharmonic ({:.3})",
            harmonic_score,
            inharmonic_score
        );
    }

    #[test]
    fn test_harmonicity_evaluate_different_series() {
        let pred = HarmonicityPredicate::default_tolerance();
        let series_a: Vec<f64> = (1..=6).map(|n| 200.0 * n as f64).collect();
        let series_b: Vec<f64> = (1..=6).map(|n| 311.0 * n as f64).collect();
        let result = pred.evaluate(&series_a, &series_b);
        assert!(
            result.is_segregated(),
            "Two clearly different harmonic series should segregate"
        );
    }

    #[test]
    fn test_harmonically_related() {
        let pred = HarmonicityPredicate::default_tolerance();
        assert!(pred.are_harmonically_related(200.0, 400.0)); // octave
        assert!(pred.are_harmonically_related(200.0, 600.0)); // 3:1
        assert!(!pred.are_harmonically_related(200.0, 311.0)); // not simple ratio
    }

    // ------ Spectral proximity -----------------------------------------------

    #[test]
    fn test_spectral_widely_separated() {
        let pred = SpectralProximityPredicate::default_threshold();
        // 300 Hz vs 3000 Hz → large Bark difference
        let result = pred.evaluate(300.0, 3000.0);
        assert!(
            result.is_segregated(),
            "300 Hz and 3000 Hz should be spectrally segregated"
        );
    }

    #[test]
    fn test_spectral_close_centroids_fused() {
        let pred = SpectralProximityPredicate::default_threshold();
        // 1000 Hz vs 1050 Hz → very small Bark difference
        let result = pred.evaluate(1000.0, 1050.0);
        assert!(
            result.is_fused(),
            "1000 Hz and 1050 Hz should be spectrally fused"
        );
    }

    #[test]
    fn test_spectral_frequency_dependent_threshold() {
        let pred = SpectralProximityPredicate::default_threshold();
        // Low-frequency region has a relaxed threshold.
        let low_thresh = pred.frequency_dependent_threshold(2.0);
        let mid_thresh = pred.frequency_dependent_threshold(10.0);
        assert!(
            low_thresh > mid_thresh,
            "Low Bark threshold ({:.3}) should exceed mid ({:.3})",
            low_thresh,
            mid_thresh
        );
    }

    // ------ Common fate ------------------------------------------------------

    #[test]
    fn test_common_fate_different_am_segregated() {
        let pred = CommonFatePredicate::default_thresholds();
        // Different AM, same FM.
        let result = pred.evaluate(4.0, 0.5, 2.0, 2.0);
        assert!(
            result.is_segregated(),
            "Large AM rate difference should segregate"
        );
    }

    #[test]
    fn test_common_fate_same_rates_fused() {
        let pred = CommonFatePredicate::default_thresholds();
        let result = pred.evaluate(3.0, 3.2, 5.0, 5.1);
        assert!(
            result.is_fused(),
            "Very similar AM and FM rates should fuse"
        );
    }

    #[test]
    fn test_common_fate_coherence_score() {
        let pred = CommonFatePredicate::default_thresholds();
        let am = vec![4.0, 4.1, 3.9, 4.05];
        let fm = vec![2.0, 2.05, 1.95, 2.0];
        let score = pred.coherence_score(&am, &fm);
        assert!(
            score > 0.7,
            "Very similar modulations should have high coherence, got {}",
            score
        );

        let am_diverse = vec![1.0, 5.0, 10.0, 20.0];
        let fm_diverse = vec![0.5, 8.0, 2.0, 15.0];
        let low_score = pred.coherence_score(&am_diverse, &fm_diverse);
        assert!(
            low_score < score,
            "Diverse modulations ({:.3}) should be less coherent than similar ({:.3})",
            low_score,
            score
        );
    }

    // ------ Spatial separation -----------------------------------------------

    #[test]
    fn test_spatial_large_separation_segregated() {
        let pred = SpatialSeparationPredicate::default_threshold();
        let result = pred.evaluate(-45.0, 45.0);
        assert!(result.is_segregated(), "90° apart should segregate");
    }

    #[test]
    fn test_spatial_small_separation_fused() {
        let pred = SpatialSeparationPredicate::default_threshold();
        let result = pred.evaluate(0.0, 5.0);
        assert!(result.is_fused(), "5° apart should fuse (threshold 10°)");
    }

    // ------ Full analyser ----------------------------------------------------

    #[test]
    fn test_full_analysis_three_well_separated_streams() {
        let analyzer = StreamSegregationAnalyzer::new();
        let streams = vec![
            StreamDescriptor::new(
                0,
                0.0,
                200.0,
                vec![200.0, 400.0, 600.0, 800.0],
                400.0,
                3.0,
                1.0,
                -60.0,
                70.0,
            ),
            StreamDescriptor::new(
                1,
                100.0,
                500.0,
                vec![500.0, 1000.0, 1500.0, 2000.0],
                1200.0,
                8.0,
                5.0,
                0.0,
                65.0,
            ),
            StreamDescriptor::new(
                2,
                200.0,
                1100.0,
                vec![1100.0, 2200.0, 3300.0, 4400.0],
                2800.0,
                15.0,
                10.0,
                60.0,
                75.0,
            ),
        ];

        let matrix = analyzer.check_all_pairs(&streams);
        assert_eq!(matrix.size(), 3);

        // All pairs should be segregated because every cue differs.
        assert!(
            matrix.all_pairs_segregated(),
            "Three maximally different streams should all segregate.\n{}",
            matrix
        );
    }

    #[test]
    fn test_conflicting_pairs_identification() {
        let analyzer = StreamSegregationAnalyzer::new();
        // Two very similar streams → should fuse.
        let streams = vec![
            StreamDescriptor::new(
                0,
                0.0,
                200.0,
                vec![200.0, 400.0, 600.0],
                400.0,
                3.0,
                1.0,
                0.0,
                70.0,
            ),
            StreamDescriptor::new(
                1,
                2.0,
                201.0,
                vec![201.0, 402.0, 603.0],
                402.0,
                3.1,
                1.05,
                1.0,
                71.0,
            ),
        ];

        let conflicts = analyzer.identify_conflicting_pairs(&streams);
        assert!(
            !conflicts.is_empty(),
            "Nearly identical streams should produce a fused (conflicting) pair"
        );
        assert_eq!(conflicts[0], (0, 1));
    }

    #[test]
    fn test_suggestion_generation_for_fused_pair() {
        let analyzer = StreamSegregationAnalyzer::new();
        let s1 = StreamDescriptor::new(
            0,
            0.0,
            200.0,
            vec![200.0, 400.0, 600.0],
            400.0,
            3.0,
            1.0,
            0.0,
            70.0,
        );
        let s2 = StreamDescriptor::new(
            1,
            5.0,
            202.0,
            vec![202.0, 404.0, 606.0],
            410.0,
            3.2,
            1.1,
            2.0,
            68.0,
        );

        let result = analyzer.analyze_pair(&s1, &s2);
        assert!(
            result.overall_segregated == false,
            "Very similar pair should fuse"
        );

        let suggestions =
            analyzer.suggest_parameter_adjustments(&s1, &s2, &result);
        assert!(
            !suggestions.is_empty(),
            "Should produce at least one adjustment suggestion"
        );

        // Every suggestion should have a positive expected improvement.
        for s in &suggestions {
            assert!(
                s.expected_improvement > 0.0,
                "Suggestion '{}' should have positive improvement",
                s.parameter
            );
        }
    }

    #[test]
    fn test_segregation_matrix_properties() {
        let mut matrix = SegregationMatrix::new(4);
        assert_eq!(matrix.size(), 4);

        // Set some results.
        let dummy_seg = PairwiseSegregation {
            onset_result: SegregationResult::segregated(10.0, 0.9, "onset", "test"),
            harmonicity_result: SegregationResult::segregated(0.1, 0.8, "harm", "test"),
            spectral_result: SegregationResult::segregated(2.0, 0.85, "spec", "test"),
            common_fate_result: SegregationResult::segregated(3.0, 0.9, "cf", "test"),
            spatial_result: SegregationResult::segregated(20.0, 0.95, "spat", "test"),
            overall_segregated: true,
            overall_confidence: 0.9,
        };

        let dummy_fus = PairwiseSegregation {
            onset_result: SegregationResult::fused(-5.0, 0.7, "onset", "test"),
            harmonicity_result: SegregationResult::fused(-0.03, 0.6, "harm", "test"),
            spectral_result: SegregationResult::fused(-0.5, 0.65, "spec", "test"),
            common_fate_result: SegregationResult::fused(-0.3, 0.6, "cf", "test"),
            spatial_result: SegregationResult::fused(-3.0, 0.7, "spat", "test"),
            overall_segregated: false,
            overall_confidence: 0.65,
        };

        matrix.set(0, 1, dummy_seg.clone());
        matrix.set(0, 2, dummy_seg.clone());
        matrix.set(0, 3, dummy_fus.clone());
        matrix.set(1, 2, dummy_seg.clone());
        matrix.set(1, 3, dummy_seg.clone());
        matrix.set(2, 3, dummy_seg.clone());

        // Symmetry check
        assert!(matrix.is_segregated(0, 1));
        assert!(matrix.is_segregated(1, 0));

        // Not all segregated because (0,3) is fused.
        assert!(!matrix.all_pairs_segregated());

        // Conflicting pairs should include (0, 3).
        let conflicts = matrix.conflicting_pairs();
        assert_eq!(conflicts, vec![(0, 3)]);

        // Segregation ratio: 5 out of 6 pairs.
        let ratio = matrix.segregation_ratio();
        let expected = 5.0 / 6.0;
        assert!(
            (ratio - expected).abs() < 0.01,
            "Ratio should be ~{:.3}, got {:.3}",
            expected,
            ratio
        );
    }

    // ------ Helper functions -------------------------------------------------

    #[test]
    fn test_hz_to_bark_and_back() {
        let freqs = [100.0, 500.0, 1000.0, 4000.0, 8000.0];
        for &f in &freqs {
            let bark = hz_to_bark(f);
            let recovered = bark_to_hz(bark);
            assert!(
                (f - recovered).abs() < 1.0,
                "Round-trip failed for {} Hz: got {} Hz (bark={})",
                f,
                recovered,
                bark
            );
        }
    }

    #[test]
    fn test_pairwise_segregation_votes() {
        let pw = PairwiseSegregation {
            onset_result: SegregationResult::segregated(10.0, 0.9, "onset", ""),
            harmonicity_result: SegregationResult::fused(-0.02, 0.5, "harm", ""),
            spectral_result: SegregationResult::segregated(2.0, 0.85, "spec", ""),
            common_fate_result: SegregationResult::fused(-0.1, 0.6, "cf", ""),
            spatial_result: SegregationResult::segregated(15.0, 0.95, "spat", ""),
            overall_segregated: true,
            overall_confidence: 0.8,
        };
        assert_eq!(pw.segregation_votes(), 3);
        assert_eq!(pw.fusion_votes(), 2);
        let conflicts = pw.conflicting_predicates();
        assert_eq!(conflicts.len(), 2);
        assert!(conflicts.contains(&"harm".to_string()));
        assert!(conflicts.contains(&"cf".to_string()));
    }

    #[test]
    fn test_analyzer_with_custom_config() {
        let analyzer = StreamSegregationAnalyzer::with_config(
            50.0, // onset threshold
            2.0,  // spectral threshold
            1.5,  // AM threshold
            1.5,  // FM threshold
            0.03, // harmonicity tolerance
            15.0, // spatial threshold
        );

        let s1 = StreamDescriptor::minimal(0, 0.0, 220.0);
        let s2 = StreamDescriptor::minimal(1, 80.0, 880.0);
        let result = analyzer.analyze_pair(&s1, &s2);

        // Large onset difference (80 ms > 50 ms threshold) → onset segregated.
        assert!(
            result.onset_result.is_segregated(),
            "80 ms diff should exceed 50 ms threshold"
        );
    }

    #[test]
    fn test_stream_descriptor_minimal() {
        let s = StreamDescriptor::minimal(0, 10.0, 440.0);
        assert_eq!(s.id, 0);
        assert_eq!(s.onset_ms, 10.0);
        assert_eq!(s.fundamental_freq, 440.0);
        assert_eq!(s.harmonics.len(), 6);
        assert!((s.harmonics[0] - 440.0).abs() < 1e-6);
        assert!((s.harmonics[5] - 2640.0).abs() < 1e-6);
        // Centroid should be mean of harmonics.
        let expected_centroid: f64 =
            (1..=6).map(|n| 440.0 * n as f64).sum::<f64>() / 6.0;
        assert!(
            (s.spectral_centroid_hz - expected_centroid).abs() < 1e-6,
            "Centroid should be {:.1}, got {:.1}",
            expected_centroid,
            s.spectral_centroid_hz
        );
    }
}
