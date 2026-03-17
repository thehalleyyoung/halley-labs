//! Integrated Psychoacoustic Model
//!
//! Combines all sub-models (masking, loudness, pitch, timbre, segregation,
//! JND, cognitive load) into a unified analysis pipeline and constraint checker.

use serde::{Serialize, Deserialize};

use crate::masking::{
    self, MaskingAnalyzer, ZwickerMaskingModel,
    NUM_BARK_BANDS, MaskedRegion,
};
use crate::loudness::{
    ZwickerLoudnessModel, StevensLoudnessModel, LoudnessNormalization,
};
use crate::pitch::PitchModel;
use crate::timbre::{TimbreSpace, TimbreDescriptor, TimbreDistance};
use crate::jnd::{
    JndValidator, PerceptualParams,
    d_prime_from_margins, p_correct_from_d_prime,
};
use crate::segregation::{
    StreamSegregationAnalyzer, SegregationMatrix, StreamDescriptor as SegStreamDescriptor,
};
use crate::cognitive_load::{
    CognitiveLoadModel,
    StreamDescriptor as CogStreamDescriptor,
};

// ---------------------------------------------------------------------------
// PerceptualAnalysisResult
// ---------------------------------------------------------------------------

/// Full analysis result from the integrated psychoacoustic model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualAnalysisResult {
    /// Per-band masked thresholds (dB SPL).
    pub masking_threshold: Vec<f64>,
    /// Per-band excitation levels (dB SPL).
    pub excitation_pattern: Vec<f64>,
    /// Per-band specific loudness (sone/Bark).
    pub specific_loudness: Vec<f64>,
    /// Total loudness (sone).
    pub total_loudness_sone: f64,
    /// Loudness level (phon).
    pub loudness_level_phon: f64,
    /// Bands where signal is masked.
    pub masked_regions: Vec<MaskedRegion>,
    /// Per-stream timbre descriptors.
    pub timbre_descriptors: Vec<TimbreDescriptor>,
    /// Overall perceptual quality score (0–1).
    pub quality_score: f64,
    /// Summary string.
    pub summary: String,
}

// ---------------------------------------------------------------------------
// IntegratedPsychoacousticModel
// ---------------------------------------------------------------------------

/// Combines all psychoacoustic sub-models into a single analysis pipeline.
///
/// Pipeline: input spectrum → masking → loudness → pitch → timbre
///           → segregation → cognitive load → quality score.
#[derive(Debug, Clone)]
pub struct IntegratedPsychoacousticModel {
    pub masking_model: ZwickerMaskingModel,
    pub loudness_model: ZwickerLoudnessModel,
    pub stevens_model: StevensLoudnessModel,
    pub normalization: LoudnessNormalization,
    pub pitch_model: PitchModel,
    pub timbre_space: TimbreSpace,
    pub timbre_distance: TimbreDistance,
    pub jnd_validator: JndValidator,
    pub segregation_analyzer: StreamSegregationAnalyzer,
    pub cognitive_model: CognitiveLoadModel,
}

impl IntegratedPsychoacousticModel {
    pub fn new() -> Self {
        Self {
            masking_model: ZwickerMaskingModel::new(),
            loudness_model: ZwickerLoudnessModel::new(),
            stevens_model: StevensLoudnessModel::new(),
            normalization: LoudnessNormalization::new(),
            pitch_model: PitchModel::new(),
            timbre_space: TimbreSpace::new(),
            timbre_distance: TimbreDistance::new(),
            jnd_validator: JndValidator::new(),
            segregation_analyzer: StreamSegregationAnalyzer::new(),
            cognitive_model: CognitiveLoadModel::with_default_budget(),
        }
    }

    /// Full analysis of a single spectrum against a masking background.
    pub fn analyze_spectrum(
        &self,
        signal_spectrum_db: &[f64],
        background_levels: &[(f64, bool)],
    ) -> PerceptualAnalysisResult {
        // 1. Masking threshold
        let threshold = self.masking_model.global_masking_threshold(background_levels);

        // 2. Excitation pattern
        let excitation = self.masking_model.excitation_pattern(signal_spectrum_db);

        // 3. Specific loudness & total loudness
        let specific = self.masking_model.specific_loudness(&excitation);
        let total_sone = self.masking_model.total_loudness(&specific);
        let phon = self.masking_model.loudness_to_phon(total_sone);

        // 4. Masked regions
        let analyzer = MaskingAnalyzer::new(self.masking_model.clone());
        let masked = analyzer.identify_masked_regions(signal_spectrum_db, &threshold);

        // 5. Quality score (heuristic: fraction of audible bands)
        let n_bands = signal_spectrum_db.len().min(NUM_BARK_BANDS);
        let n_masked = masked.len();
        let audible_fraction = if n_bands > 0 {
            1.0 - n_masked as f64 / n_bands as f64
        } else {
            1.0
        };
        let quality = audible_fraction;

        let summary = format!(
            "Loudness: {:.1} sone ({:.1} phon), {}/{} bands audible, quality={:.2}",
            total_sone,
            phon,
            n_bands - n_masked,
            n_bands,
            quality
        );

        PerceptualAnalysisResult {
            masking_threshold: threshold,
            excitation_pattern: excitation,
            specific_loudness: specific,
            total_loudness_sone: total_sone,
            loudness_level_phon: phon,
            masked_regions: masked,
            timbre_descriptors: Vec::new(),
            quality_score: quality,
            summary,
        }
    }

    /// Compute the model-predicted discriminability d' between two parameter sets.
    pub fn d_prime(&self, p1: &PerceptualParams, p2: &PerceptualParams) -> f64 {
        let report = self.jnd_validator.all_dimensions_discriminable(p1, p2);
        let margins: Vec<f64> = report.results.iter().map(|r| r.margin).collect();
        d_prime_from_margins(&margins)
    }

    /// Probability of correct discrimination between two parameter sets.
    pub fn p_correct(&self, p1: &PerceptualParams, p2: &PerceptualParams) -> f64 {
        p_correct_from_d_prime(self.d_prime(p1, p2))
    }

    /// Multi-stream analysis: masking + segregation + cognitive load.
    pub fn analyze_multi_stream(
        &self,
        stream_spectra: &[Vec<f64>],
        stream_params: &[PerceptualParams],
        seg_descriptors: &[SegStreamDescriptor],
        cog_descriptors: &[CogStreamDescriptor],
    ) -> MultiStreamAnalysis {
        // Masking analysis
        let tonal = vec![false; stream_spectra.len()];
        let masking_analyzer = MaskingAnalyzer::new(self.masking_model.clone());
        let masking_reports =
            masking_analyzer.full_multi_stream_analysis(stream_spectra, &tonal);

        // Segregation analysis
        let seg_matrix = self.segregation_analyzer.check_all_pairs(seg_descriptors);

        // JND pairwise
        let n = stream_params.len();
        let mut jnd_reports = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let report =
                    self.jnd_validator.all_dimensions_discriminable(&stream_params[i], &stream_params[j]);
                jnd_reports.push((i, j, report));
            }
        }

        // Cognitive load
        let cog_report = self.cognitive_model.evaluate(cog_descriptors);

        // Overall quality
        let seg_ok = seg_matrix.all_pairs_segregated();
        let jnd_ok = jnd_reports.iter().all(|(_, _, r)| r.any_passed);
        let cog_ok = cog_report.within_budget;
        let total_masked: usize = masking_reports.iter().map(|r| r.masked_regions.len()).sum();
        let mask_ok = total_masked == 0;

        let quality = [seg_ok, jnd_ok, cog_ok, mask_ok]
            .iter()
            .filter(|&&b| b)
            .count() as f64
            / 4.0;

        MultiStreamAnalysis {
            n_streams: n,
            masking_reports,
            segregation_ok: seg_ok,
            jnd_ok,
            cognitive_load_ok: cog_ok,
            all_audible: mask_ok,
            overall_quality: quality,
            cognitive_load_utilization: cog_report.utilization,
        }
    }
}

impl Default for IntegratedPsychoacousticModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-stream analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiStreamAnalysis {
    pub n_streams: usize,
    pub masking_reports: Vec<masking::MultiStreamMaskingReport>,
    pub segregation_ok: bool,
    pub jnd_ok: bool,
    pub cognitive_load_ok: bool,
    pub all_audible: bool,
    pub overall_quality: f64,
    pub cognitive_load_utilization: f64,
}

// ---------------------------------------------------------------------------
// PerceptualConstraintChecker
// ---------------------------------------------------------------------------

/// A constraint violation with severity and details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub constraint_name: String,
    pub severity: ConstraintSeverity,
    pub message: String,
    pub margin: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintSeverity {
    /// Hard violation – must be fixed.
    Error,
    /// Soft violation – may degrade quality.
    Warning,
    /// Informational.
    Info,
}

/// Detailed constraint satisfaction report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintReport {
    pub all_satisfied: bool,
    pub n_errors: usize,
    pub n_warnings: usize,
    pub violations: Vec<ConstraintViolation>,
    pub summary: String,
}

/// Checks all psychoacoustic constraints for a stream configuration.
#[derive(Debug, Clone)]
pub struct PerceptualConstraintChecker {
    pub model: IntegratedPsychoacousticModel,
    /// Maximum allowed cognitive load (budget).
    pub max_cognitive_load: f64,
    /// Minimum required masking margin (dB) for audibility.
    pub min_masking_margin_db: f64,
    /// Minimum required JND margin (multiples of JND).
    pub min_jnd_margin: f64,
    /// Maximum number of streams allowed.
    pub max_streams: usize,
}

impl PerceptualConstraintChecker {
    pub fn new() -> Self {
        Self {
            model: IntegratedPsychoacousticModel::new(),
            max_cognitive_load: 1.0,
            min_masking_margin_db: 3.0,
            min_jnd_margin: 1.5,
            max_streams: 5,
        }
    }

    pub fn with_budget(mut self, budget: f64) -> Self {
        self.max_cognitive_load = budget;
        self
    }

    /// Check constraints on a set of streams.
    pub fn check(
        &self,
        stream_spectra: &[Vec<f64>],
        stream_params: &[PerceptualParams],
        seg_descriptors: &[SegStreamDescriptor],
        cog_descriptors: &[CogStreamDescriptor],
    ) -> ConstraintReport {
        let mut violations = Vec::new();

        // 1. Stream count
        if stream_spectra.len() > self.max_streams {
            violations.push(ConstraintViolation {
                constraint_name: "MaxStreams".into(),
                severity: ConstraintSeverity::Error,
                message: format!(
                    "{} streams exceed max {}",
                    stream_spectra.len(),
                    self.max_streams
                ),
                margin: -(stream_spectra.len() as f64 - self.max_streams as f64),
            });
        }

        // 2. Masking: check each stream is audible
        let tonal = vec![false; stream_spectra.len()];
        let masking_analyzer = MaskingAnalyzer::new(self.model.masking_model.clone());
        let masking_results =
            masking_analyzer.full_multi_stream_analysis(stream_spectra, &tonal);
        for report in &masking_results {
            for region in &report.masked_regions {
                if region.deficit_db > self.min_masking_margin_db {
                    violations.push(ConstraintViolation {
                        constraint_name: "Audibility".into(),
                        severity: ConstraintSeverity::Error,
                        message: format!(
                            "Stream {} band {} masked by {:.1} dB",
                            report.stream_index, region.band_index, region.deficit_db
                        ),
                        margin: -region.deficit_db,
                    });
                }
            }
        }

        // 3. JND: pairwise discriminability
        for i in 0..stream_params.len() {
            for j in (i + 1)..stream_params.len() {
                let report = self
                    .model
                    .jnd_validator
                    .all_dimensions_discriminable(&stream_params[i], &stream_params[j]);
                if !report.any_passed {
                    violations.push(ConstraintViolation {
                        constraint_name: "Discriminability".into(),
                        severity: ConstraintSeverity::Error,
                        message: format!(
                            "Streams {} and {} indistinguishable (min margin {:.2})",
                            i, j, report.min_margin
                        ),
                        margin: report.min_margin - self.min_jnd_margin,
                    });
                } else if report.min_margin < self.min_jnd_margin {
                    violations.push(ConstraintViolation {
                        constraint_name: "Discriminability".into(),
                        severity: ConstraintSeverity::Warning,
                        message: format!(
                            "Streams {} and {} marginally discriminable ({:.2} JND)",
                            i, j, report.min_margin
                        ),
                        margin: report.min_margin - self.min_jnd_margin,
                    });
                }
            }
        }

        // 4. Segregation: pairwise
        let seg_matrix = self.model.segregation_analyzer.check_all_pairs(seg_descriptors);
        let conflicts = seg_matrix.conflicting_pairs();
        for (i, j) in &conflicts {
            violations.push(ConstraintViolation {
                constraint_name: "Segregation".into(),
                severity: ConstraintSeverity::Warning,
                message: format!("Streams {} and {} not perceptually segregated", i, j),
                margin: -1.0,
            });
        }

        // 5. Cognitive load
        let cog_report = self.model.cognitive_model.evaluate(cog_descriptors);
        if !cog_report.within_budget {
            violations.push(ConstraintViolation {
                constraint_name: "CognitiveLoad".into(),
                severity: ConstraintSeverity::Error,
                message: format!(
                    "Cognitive load {:.2} exceeds budget {:.2}",
                    cog_report.composed_load, cog_report.budget
                ),
                margin: cog_report.budget - cog_report.composed_load,
            });
        } else if cog_report.utilization > 0.85 {
            violations.push(ConstraintViolation {
                constraint_name: "CognitiveLoad".into(),
                severity: ConstraintSeverity::Warning,
                message: format!(
                    "Cognitive load utilization high: {:.0}%",
                    cog_report.utilization * 100.0
                ),
                margin: cog_report.budget - cog_report.composed_load,
            });
        }

        let n_errors = violations
            .iter()
            .filter(|v| v.severity == ConstraintSeverity::Error)
            .count();
        let n_warnings = violations
            .iter()
            .filter(|v| v.severity == ConstraintSeverity::Warning)
            .count();
        let all_satisfied = n_errors == 0;

        let summary = if all_satisfied {
            format!(
                "All constraints satisfied ({} warnings)",
                n_warnings
            )
        } else {
            format!("{} errors, {} warnings", n_errors, n_warnings)
        };

        ConstraintReport {
            all_satisfied,
            n_errors,
            n_warnings,
            violations,
            summary,
        }
    }

    /// Quick single-spectrum check against silence background.
    pub fn check_single_spectrum(&self, spectrum_db: &[f64]) -> ConstraintReport {
        let bg: Vec<(f64, bool)> = vec![(-120.0, false); spectrum_db.len()];
        let analysis = self.model.analyze_spectrum(spectrum_db, &bg);

        let mut violations = Vec::new();
        if analysis.total_loudness_sone < 0.1 {
            violations.push(ConstraintViolation {
                constraint_name: "MinLoudness".into(),
                severity: ConstraintSeverity::Warning,
                message: format!(
                    "Very low loudness: {:.3} sone",
                    analysis.total_loudness_sone
                ),
                margin: analysis.total_loudness_sone - 0.1,
            });
        }

        let n_errors = violations
            .iter()
            .filter(|v| v.severity == ConstraintSeverity::Error)
            .count();
        let n_warnings = violations
            .iter()
            .filter(|v| v.severity == ConstraintSeverity::Warning)
            .count();

        ConstraintReport {
            all_satisfied: n_errors == 0,
            n_errors,
            n_warnings,
            violations,
            summary: format!("Single spectrum: {:.1} sone", analysis.total_loudness_sone),
        }
    }
}

impl Default for PerceptualConstraintChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Cross-model comparison utilities
// ---------------------------------------------------------------------------

/// Compare Zwicker and Stevens loudness models on the same spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoudnessComparison {
    pub zwicker_sone: f64,
    pub zwicker_phon: f64,
    pub stevens_sone: f64,
    pub ratio: f64,
}

/// Compare loudness predictions from different models.
pub fn compare_loudness_models(band_levels_db: &[f64]) -> LoudnessComparison {
    let zwicker = ZwickerLoudnessModel::new();
    let stevens = StevensLoudnessModel::new();

    let mut bark_array = [0.0_f64; NUM_BARK_BANDS];
    for (i, &v) in band_levels_db.iter().enumerate().take(NUM_BARK_BANDS) {
        bark_array[i] = v;
    }
    let zr = zwicker.compute_loudness_from_spectrum(&bark_array);
    let avg_db = band_levels_db.iter().sum::<f64>() / band_levels_db.len().max(1) as f64;
    let stevens_sone = stevens.loudness_from_db(avg_db);

    let ratio = if stevens_sone > 0.0 {
        zr.total_loudness_sone / stevens_sone
    } else {
        f64::NAN
    };

    LoudnessComparison {
        zwicker_sone: zr.total_loudness_sone,
        zwicker_phon: zr.loudness_level_phon,
        stevens_sone,
        ratio,
    }
}

/// Perceptual quality score combining multiple metrics.
///
/// Weighted combination of:
///  - audibility (masking margin)
///  - discriminability (JND margins)
///  - segregation confidence
///  - cognitive load headroom
pub fn perceptual_quality_score(
    masking_fraction_audible: f64,
    min_jnd_margin: f64,
    segregation_fraction: f64,
    cognitive_headroom: f64,
) -> f64 {
    let w_mask = 0.30;
    let w_jnd = 0.25;
    let w_seg = 0.25;
    let w_cog = 0.20;

    let s_mask = masking_fraction_audible.clamp(0.0, 1.0);
    let s_jnd = (min_jnd_margin / 3.0).clamp(0.0, 1.0); // 3 JNDs → 1.0
    let s_seg = segregation_fraction.clamp(0.0, 1.0);
    let s_cog = cognitive_headroom.clamp(0.0, 1.0);

    w_mask * s_mask + w_jnd * s_jnd + w_seg * s_seg + w_cog * s_cog
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params(freq: f64, level: f64, onset: f64, az: f64) -> PerceptualParams {
        PerceptualParams::basic(freq, level, onset, az)
    }

    #[test]
    fn test_integrated_model_creates() {
        let model = IntegratedPsychoacousticModel::new();
        assert!(model.jnd_validator.pitch.optimal_fraction > 0.0);
    }

    #[test]
    fn test_analyze_spectrum_basic() {
        let model = IntegratedPsychoacousticModel::new();
        let signal = vec![60.0; NUM_BARK_BANDS];
        let bg: Vec<(f64, bool)> = vec![(-120.0, false); NUM_BARK_BANDS];
        let result = model.analyze_spectrum(&signal, &bg);
        assert!(result.total_loudness_sone > 0.0);
        assert!(result.quality_score > 0.0);
    }

    #[test]
    fn test_d_prime_well_separated() {
        let model = IntegratedPsychoacousticModel::new();
        let p1 = make_params(300.0, 50.0, 0.0, -30.0);
        let p2 = make_params(2000.0, 70.0, 200.0, 30.0);
        let dp = model.d_prime(&p1, &p2);
        assert!(dp > 3.0, "Well-separated streams should have large d', got {dp}");
    }

    #[test]
    fn test_d_prime_similar() {
        let model = IntegratedPsychoacousticModel::new();
        let p1 = make_params(1000.0, 60.0, 100.0, 0.0);
        let p2 = make_params(1001.0, 60.0, 100.0, 0.0);
        let dp = model.d_prime(&p1, &p2);
        assert!(dp < 3.0, "Similar streams: low d', got {dp}");
    }

    #[test]
    fn test_p_correct_high_d_prime() {
        let model = IntegratedPsychoacousticModel::new();
        let p1 = make_params(300.0, 40.0, 0.0, -45.0);
        let p2 = make_params(4000.0, 80.0, 300.0, 45.0);
        let pc = model.p_correct(&p1, &p2);
        assert!(pc > 0.9, "Very different: P(correct) should be >0.9, got {pc}");
    }

    #[test]
    fn test_constraint_checker_creates() {
        let checker = PerceptualConstraintChecker::new();
        assert_eq!(checker.max_streams, 5);
    }

    #[test]
    fn test_constraint_checker_single_spectrum() {
        let checker = PerceptualConstraintChecker::new();
        let spec = vec![60.0; NUM_BARK_BANDS];
        let report = checker.check_single_spectrum(&spec);
        assert!(report.all_satisfied);
    }

    #[test]
    fn test_constraint_too_many_streams() {
        let checker = PerceptualConstraintChecker::new().with_budget(1.0);
        let spectra: Vec<Vec<f64>> = (0..8).map(|_| vec![60.0; NUM_BARK_BANDS]).collect();
        let params: Vec<PerceptualParams> = (0..8)
            .map(|i| make_params(200.0 + i as f64 * 400.0, 60.0, i as f64 * 50.0, 0.0))
            .collect();
        let seg: Vec<SegStreamDescriptor> = (0..8)
            .map(|i| SegStreamDescriptor {
                id: i,
                onset_ms: i as f64 * 50.0,
                fundamental_freq: 200.0 + i as f64 * 400.0,
                harmonics: vec![],
                spectral_centroid_hz: 200.0 + i as f64 * 400.0,
                am_rate: 0.0,
                fm_rate: 0.0,
                azimuth_deg: 0.0,
                level_db: 60.0,
            })
            .collect();
        let cog: Vec<CogStreamDescriptor> = (0..8)
            .map(|i| CogStreamDescriptor {
                id: i,
                name: format!("s{i}"),
                information_rate_bits_per_sec: 10.0,
                stream_complexity: 0.5,
                update_rate_hz: 2.0,
                spectral_complexity: 0.5,
                temporal_regularity: 0.5,
                familiarity: 0.5,
                priority: 0.5,
            })
            .collect();
        let report = checker.check(&spectra, &params, &seg, &cog);
        // Should flag too many streams
        assert!(
            report.violations.iter().any(|v| v.constraint_name == "MaxStreams"),
            "Should report MaxStreams violation"
        );
    }

    #[test]
    fn test_compare_loudness_models() {
        let levels = vec![60.0; NUM_BARK_BANDS];
        let cmp = compare_loudness_models(&levels);
        assert!(cmp.zwicker_sone > 0.0);
        assert!(cmp.stevens_sone > 0.0);
    }

    #[test]
    fn test_quality_score_range() {
        let score = perceptual_quality_score(1.0, 3.0, 1.0, 0.5);
        assert!(score > 0.0 && score <= 1.0, "Score out of range: {score}");
    }

    #[test]
    fn test_quality_score_zero_inputs() {
        let score = perceptual_quality_score(0.0, 0.0, 0.0, 0.0);
        assert!((score - 0.0).abs() < 0.01, "All-zero inputs: score ~ 0");
    }

    #[test]
    fn test_quality_score_perfect() {
        let score = perceptual_quality_score(1.0, 3.0, 1.0, 1.0);
        assert!(score > 0.9, "Perfect inputs: score near 1, got {score}");
    }

    #[test]
    fn test_constraint_report_summary() {
        let checker = PerceptualConstraintChecker::new();
        let spec = vec![60.0; NUM_BARK_BANDS];
        let report = checker.check_single_spectrum(&spec);
        assert!(!report.summary.is_empty());
    }
}
