//! Core fault localization engine implementing the causal-differential algorithm.
//!
//! This is the primary entry point for running fault localization on NLP pipelines.
//! It combines SBFL metrics, causal intervention analysis, and iterative peeling
//! to pinpoint the exact pipeline stage that introduced or amplified a fault.

use crate::{
    CausalVerdict, FaultClassification, InterventionResult, LocalizationResult,
    StageLocalizationResult, SuspiciousnessEntry, SuspiciousnessRanking,
    TransformationStageData,
};
use shared_types::error::{LocalizerError, Result};
use shared_types::ir::IntermediateRepresentation;
use shared_types::types::StageId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ── Engine configuration ────────────────────────────────────────────────────

/// Configuration for the fault localization engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizationConfig {
    /// Maximum number of iterative peeling rounds.
    pub max_peeling_rounds: usize,
    /// Minimum suspiciousness score to consider a stage faulty.
    pub suspiciousness_threshold: f64,
    /// Whether to perform causal intervention analysis.
    pub enable_causal_analysis: bool,
    /// Whether to compute the discriminability matrix pre-check.
    pub enable_discriminability_check: bool,
    /// Number of calibration samples for discriminability estimation.
    pub calibration_sample_count: usize,
    /// SBFL metric to use for initial ranking.
    pub sbfl_metric: SBFLMetric,
    /// Maximum pipeline stages to investigate causally.
    pub max_causal_stages: usize,
    /// Significance level for hypothesis testing.
    pub significance_level: f64,
    /// Whether to use parallel execution for test runs.
    pub parallel_execution: bool,
    /// Budget in seconds per test case.
    pub per_test_budget_secs: f64,
}

impl Default for LocalizationConfig {
    fn default() -> Self {
        Self {
            max_peeling_rounds: 10,
            suspiciousness_threshold: 0.3,
            enable_causal_analysis: true,
            enable_discriminability_check: true,
            calibration_sample_count: 200,
            sbfl_metric: SBFLMetric::Ochiai,
            max_causal_stages: 5,
            significance_level: 0.05,
            parallel_execution: true,
            per_test_budget_secs: 5.0,
        }
    }
}

/// SBFL metric used for initial stage ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SBFLMetric {
    Ochiai,
    Tarantula,
    DStar,
    Barinel,
    /// Weighted combination of multiple metrics.
    Ensemble,
}

// ── Per-stage spectrum data ─────────────────────────────────────────────────

/// Spectrum data for a single pipeline stage accumulated across test cases.
#[derive(Debug, Clone)]
pub struct StageSpectrum {
    pub stage_id: StageId,
    pub stage_name: String,
    /// Number of test cases where this stage showed high differential and the test failed.
    pub ef: usize,
    /// Number of test cases where this stage showed high differential and the test passed.
    pub ep: usize,
    /// Number of test cases where this stage showed low differential and the test failed.
    pub nf: usize,
    /// Number of test cases where this stage showed low differential and the test passed.
    pub np: usize,
    /// Raw differential values per test case.
    pub differentials: Vec<f64>,
    /// Differential values per transformation type.
    pub per_transformation: HashMap<String, Vec<f64>>,
}

impl StageSpectrum {
    pub fn new(stage_id: StageId, stage_name: String) -> Self {
        Self {
            stage_id,
            stage_name,
            ef: 0,
            ep: 0,
            nf: 0,
            np: 0,
            differentials: Vec::new(),
            per_transformation: HashMap::new(),
        }
    }

    /// Total number of test cases observed.
    pub fn total(&self) -> usize {
        self.ef + self.ep + self.nf + self.np
    }

    /// Total failing test cases.
    pub fn total_failed(&self) -> usize {
        self.ef + self.nf
    }

    /// Total passing test cases.
    pub fn total_passed(&self) -> usize {
        self.ep + self.np
    }

    /// Record a test observation.
    pub fn record(&mut self, differential: f64, violation: bool, threshold: f64) {
        self.differentials.push(differential);
        let high_diff = differential > threshold;
        match (high_diff, violation) {
            (true, true) => self.ef += 1,
            (true, false) => self.ep += 1,
            (false, true) => self.nf += 1,
            (false, false) => self.np += 1,
        }
    }

    /// Record a test observation with transformation name.
    pub fn record_with_transformation(
        &mut self,
        differential: f64,
        violation: bool,
        threshold: f64,
        transformation: &str,
    ) {
        self.record(differential, violation, threshold);
        self.per_transformation
            .entry(transformation.to_string())
            .or_default()
            .push(differential);
    }

    /// Compute mean differential.
    pub fn mean_differential(&self) -> f64 {
        if self.differentials.is_empty() {
            return 0.0;
        }
        self.differentials.iter().sum::<f64>() / self.differentials.len() as f64
    }

    /// Compute variance of differentials.
    pub fn variance(&self) -> f64 {
        if self.differentials.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_differential();
        let sum_sq: f64 = self.differentials.iter().map(|d| (d - mean).powi(2)).sum();
        sum_sq / (self.differentials.len() - 1) as f64
    }

    /// Compute standard deviation of differentials.
    pub fn std_deviation(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Compute the Ochiai suspiciousness score.
    pub fn ochiai_score(&self) -> f64 {
        let total_failed = (self.ef + self.nf) as f64;
        let total_executed = (self.ef + self.ep) as f64;
        let denominator = (total_failed * total_executed).sqrt();
        if denominator < f64::EPSILON {
            return 0.0;
        }
        self.ef as f64 / denominator
    }

    /// Compute the Tarantula suspiciousness score.
    pub fn tarantula_score(&self) -> f64 {
        let total_failed = (self.ef + self.nf) as f64;
        let total_passed = (self.ep + self.np) as f64;
        if total_failed < f64::EPSILON {
            return 0.0;
        }
        let fail_ratio = self.ef as f64 / total_failed;
        let pass_ratio = if total_passed < f64::EPSILON {
            0.0
        } else {
            self.ep as f64 / total_passed
        };
        let denominator = fail_ratio + pass_ratio;
        if denominator < f64::EPSILON {
            return 0.0;
        }
        fail_ratio / denominator
    }

    /// Compute the DStar suspiciousness score with parameter p.
    pub fn dstar_score(&self, p: f64) -> f64 {
        let denominator = self.ep as f64 + self.nf as f64;
        if denominator < f64::EPSILON {
            return 0.0;
        }
        (self.ef as f64).powf(p) / denominator
    }

    /// Compute the Barinel suspiciousness score.
    pub fn barinel_score(&self) -> f64 {
        let total_executed = (self.ef + self.ep) as f64;
        if total_executed < f64::EPSILON {
            return 0.0;
        }
        1.0 - (self.ep as f64 / total_executed)
    }
}

// ── Test observation ────────────────────────────────────────────────────────

/// A single test observation recording per-stage differentials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestObservation {
    pub test_id: String,
    pub transformation_name: String,
    pub input_text: String,
    pub transformed_text: String,
    pub violation_detected: bool,
    pub violation_magnitude: f64,
    pub per_stage_differentials: HashMap<String, f64>,
    pub execution_time_ms: f64,
}

// ── Causal analysis types ───────────────────────────────────────────────────

/// Result of interventional analysis on a single stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageIntervention {
    pub stage_id: StageId,
    pub stage_name: String,
    pub original_violation_magnitude: f64,
    pub post_intervention_magnitude: f64,
    pub attenuation_ratio: f64,
    pub verdict: CausalVerdict,
}

/// Iterative peeling state for multi-fault localization.
#[derive(Debug, Clone)]
pub struct PeelingState {
    /// Stages already identified and intervened on.
    pub identified_stages: Vec<StageId>,
    /// Current residual violation magnitude.
    pub residual_magnitude: f64,
    /// History of peeling rounds.
    pub rounds: Vec<PeelingRound>,
}

/// A single round of iterative peeling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeelingRound {
    pub round_number: usize,
    pub suspected_stage: String,
    pub intervention_result: StageIntervention,
    pub residual_after: f64,
    pub stages_remaining: usize,
}

// ── The engine ──────────────────────────────────────────────────────────────

/// The main fault localization engine.
pub struct LocalizationEngine {
    config: LocalizationConfig,
    spectra: HashMap<String, StageSpectrum>,
    observations: Vec<TestObservation>,
    stage_names: Vec<String>,
    stage_ids: Vec<StageId>,
    differential_threshold: f64,
}

impl LocalizationEngine {
    /// Create a new engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(LocalizationConfig::default())
    }

    /// Create a new engine with custom configuration.
    pub fn with_config(config: LocalizationConfig) -> Self {
        Self {
            config,
            spectra: HashMap::new(),
            observations: Vec::new(),
            stage_names: Vec::new(),
            stage_ids: Vec::new(),
            differential_threshold: 0.1,
        }
    }

    /// Register pipeline stages for analysis.
    pub fn register_stages(&mut self, stages: Vec<(StageId, String)>) {
        for (id, name) in stages {
            self.stage_ids.push(id);
            self.stage_names.push(name.clone());
            self.spectra
                .insert(name.clone(), StageSpectrum::new(id, name));
        }
    }

    /// Set the differential threshold for classifying high/low differential.
    pub fn set_differential_threshold(&mut self, threshold: f64) {
        self.differential_threshold = threshold;
    }

    /// Record a test observation from running a single metamorphic test case.
    pub fn record_observation(&mut self, obs: TestObservation) {
        for (stage_name, &differential) in &obs.per_stage_differentials {
            if let Some(spectrum) = self.spectra.get_mut(stage_name) {
                spectrum.record_with_transformation(
                    differential,
                    obs.violation_detected,
                    self.differential_threshold,
                    &obs.transformation_name,
                );
            }
        }
        self.observations.push(obs);
    }

    /// Record multiple observations at once.
    pub fn record_observations(&mut self, observations: Vec<TestObservation>) {
        for obs in observations {
            self.record_observation(obs);
        }
    }

    /// Compute suspiciousness scores for all stages using the configured metric.
    pub fn compute_suspiciousness(&self) -> SuspiciousnessRanking {
        let mut entries: Vec<SuspiciousnessEntry> = self
            .spectra
            .values()
            .map(|spectrum| {
                let score = match self.config.sbfl_metric {
                    SBFLMetric::Ochiai => spectrum.ochiai_score(),
                    SBFLMetric::Tarantula => spectrum.tarantula_score(),
                    SBFLMetric::DStar => spectrum.dstar_score(2.0),
                    SBFLMetric::Barinel => spectrum.barinel_score(),
                    SBFLMetric::Ensemble => self.ensemble_score(spectrum),
                };
                SuspiciousnessEntry {
                    stage_name: spectrum.stage_name.clone(),
                    score,
                    rank: 0,
                }
            })
            .collect();

        entries.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, entry) in entries.iter_mut().enumerate() {
            entry.rank = i + 1;
        }

        SuspiciousnessRanking { rankings: entries }
    }

    /// Compute an ensemble suspiciousness score combining multiple metrics.
    fn ensemble_score(&self, spectrum: &StageSpectrum) -> f64 {
        let ochiai = spectrum.ochiai_score();
        let tarantula = spectrum.tarantula_score();
        let dstar = spectrum.dstar_score(2.0);
        let barinel = spectrum.barinel_score();
        // Weighted average favoring Ochiai (most robust in literature).
        0.4 * ochiai + 0.25 * dstar + 0.2 * tarantula + 0.15 * barinel
    }

    /// Perform causal intervention analysis on a suspected stage.
    ///
    /// Replaces the stage's output with the original execution's output
    /// and observes whether the downstream violation disappears.
    pub fn analyze_causal_intervention(
        &self,
        stage_name: &str,
        original_magnitude: f64,
        post_intervention_magnitude: f64,
    ) -> StageIntervention {
        let stage_id = self
            .spectra
            .get(stage_name)
            .map(|s| s.stage_id)
            .unwrap_or_else(|| StageId::new(stage_name));

        let attenuation = if original_magnitude > f64::EPSILON {
            1.0 - (post_intervention_magnitude / original_magnitude)
        } else {
            0.0
        };

        let verdict = if post_intervention_magnitude < f64::EPSILON {
            CausalVerdict::Introduced
        } else if attenuation > 0.5 {
            CausalVerdict::Amplified {
                amplification_factor: original_magnitude / post_intervention_magnitude.max(f64::EPSILON),
            }
        } else if attenuation > 0.1 {
            CausalVerdict::Contributing
        } else {
            CausalVerdict::NotCausal
        };

        StageIntervention {
            stage_id,
            stage_name: stage_name.to_string(),
            original_violation_magnitude: original_magnitude,
            post_intervention_magnitude,
            attenuation_ratio: attenuation,
            verdict,
        }
    }

    /// Run iterative peeling to identify multiple faulty stages.
    ///
    /// Implements the multi-fault extension from Theorem 1 (M4):
    /// After identifying k₁*, replace k₁*'s output, re-run localization
    /// on the residual pipeline to identify k₂*.
    pub fn iterative_peeling(
        &self,
        initial_magnitude: f64,
        intervention_results: &[(String, f64)],
    ) -> PeelingState {
        let mut state = PeelingState {
            identified_stages: Vec::new(),
            residual_magnitude: initial_magnitude,
            rounds: Vec::new(),
        };

        let mut remaining_stages: Vec<&String> =
            intervention_results.iter().map(|(name, _)| name).collect();

        for round_num in 0..self.config.max_peeling_rounds {
            if state.residual_magnitude < f64::EPSILON {
                break;
            }
            if remaining_stages.is_empty() {
                break;
            }

            // Find the stage whose intervention causes the largest attenuation.
            let mut best_stage = None;
            let mut best_attenuation = 0.0f64;

            for (name, post_mag) in intervention_results {
                if state.identified_stages.iter().any(|id| {
                    self.spectra
                        .get(name)
                        .map(|s| s.stage_id == *id)
                        .unwrap_or(false)
                }) {
                    continue;
                }
                let attenuation = state.residual_magnitude - post_mag;
                if attenuation > best_attenuation {
                    best_attenuation = attenuation;
                    best_stage = Some(name.clone());
                }
            }

            let stage_name = match best_stage {
                Some(name) => name,
                None => break,
            };

            let post_mag = intervention_results
                .iter()
                .find(|(n, _)| n == &stage_name)
                .map(|(_, m)| *m)
                .unwrap_or(state.residual_magnitude);

            let intervention = self.analyze_causal_intervention(
                &stage_name,
                state.residual_magnitude,
                post_mag,
            );

            let stage_id = intervention.stage_id;
            state.identified_stages.push(stage_id);

            remaining_stages.retain(|s| **s != stage_name);

            state.residual_magnitude = post_mag;

            state.rounds.push(PeelingRound {
                round_number: round_num + 1,
                suspected_stage: stage_name,
                intervention_result: intervention,
                residual_after: state.residual_magnitude,
                stages_remaining: remaining_stages.len(),
            });

            if state.residual_magnitude < self.config.suspiciousness_threshold * initial_magnitude {
                break;
            }
        }

        state
    }

    /// Run the full localization analysis and produce a complete result.
    pub fn run_analysis(&self) -> Result<LocalizationResult> {
        if self.observations.is_empty() {
            return Err(LocalizerError::validation(
                "localization",
                "no test observations recorded",
            ));
        }

        let ranking = self.compute_suspiciousness();
        let violation_count = self
            .observations
            .iter()
            .filter(|o| o.violation_detected)
            .count();

        let transformations_used: Vec<String> = {
            let mut t: Vec<String> = self
                .observations
                .iter()
                .map(|o| o.transformation_name.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            t.sort();
            t
        };

        let stage_results: Vec<StageLocalizationResult> = ranking
            .rankings
            .iter()
            .map(|entry| {
                let spectrum = self.spectra.get(&entry.stage_name);
                let per_transformation = spectrum
                    .map(|s| {
                        s.per_transformation
                            .iter()
                            .map(|(name, diffs)| {
                                let mean = if diffs.is_empty() {
                                    0.0
                                } else {
                                    diffs.iter().sum::<f64>() / diffs.len() as f64
                                };
                                let violations = self
                                    .observations
                                    .iter()
                                    .filter(|o| {
                                        o.transformation_name == *name && o.violation_detected
                                    })
                                    .count();
                                (
                                    name.clone(),
                                    TransformationStageData {
                                        transformation_name: name.clone(),
                                        mean_differential: mean,
                                        sample_count: diffs.len(),
                                        violation_count: violations,
                                    },
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                StageLocalizationResult {
                    stage_name: entry.stage_name.clone(),
                    stage_id: spectrum
                        .map(|s| s.stage_id)
                        .unwrap_or_else(|| StageId::new(&entry.stage_name)),
                    suspiciousness: entry.score,
                    rank: entry.rank,
                    fault_type: self.classify_fault(entry),
                    evidence: self.gather_evidence(&entry.stage_name),
                    differential_data: spectrum
                        .map(|s| s.differentials.clone())
                        .unwrap_or_default(),
                    per_transformation,
                }
            })
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert("sbfl_metric".to_string(), format!("{:?}", self.config.sbfl_metric));
        metadata.insert("test_count".to_string(), self.observations.len().to_string());
        metadata.insert(
            "differential_threshold".to_string(),
            self.differential_threshold.to_string(),
        );

        Ok(LocalizationResult {
            pipeline_name: String::new(),
            stage_results,
            test_count: self.observations.len(),
            violation_count,
            transformations_used,
            metadata,
        })
    }

    /// Classify the fault type based on stage characteristics.
    fn classify_fault(&self, entry: &SuspiciousnessEntry) -> Option<String> {
        if entry.score < self.config.suspiciousness_threshold {
            return None;
        }
        let spectrum = self.spectra.get(&entry.stage_name)?;

        // Check for cascading amplification pattern: high mean + high variance
        let mean = spectrum.mean_differential();
        let std = spectrum.std_deviation();
        if mean > 0.5 && std > 0.3 {
            return Some("cascading_amplification".to_string());
        }

        // Check for consistent introduction pattern: high mean + low variance
        if mean > 0.3 && std < 0.15 {
            return Some("consistent_introduction".to_string());
        }

        // Check for transformation-specific fault
        let max_transform_mean = spectrum
            .per_transformation
            .values()
            .filter(|diffs| !diffs.is_empty())
            .map(|diffs| diffs.iter().sum::<f64>() / diffs.len() as f64)
            .fold(0.0f64, f64::max);

        if max_transform_mean > 2.0 * mean && max_transform_mean > 0.4 {
            return Some("transformation_specific".to_string());
        }

        Some("general_fault".to_string())
    }

    /// Gather evidence strings for a stage.
    fn gather_evidence(&self, stage_name: &str) -> Vec<String> {
        let mut evidence = Vec::new();

        if let Some(spectrum) = self.spectra.get(stage_name) {
            evidence.push(format!(
                "ef={}, ep={}, nf={}, np={} (total={})",
                spectrum.ef,
                spectrum.ep,
                spectrum.nf,
                spectrum.np,
                spectrum.total()
            ));
            evidence.push(format!(
                "mean_differential={:.4}, std={:.4}",
                spectrum.mean_differential(),
                spectrum.std_deviation()
            ));
            evidence.push(format!(
                "ochiai={:.4}, tarantula={:.4}, dstar={:.4}, barinel={:.4}",
                spectrum.ochiai_score(),
                spectrum.tarantula_score(),
                spectrum.dstar_score(2.0),
                spectrum.barinel_score()
            ));

            // Top transformations by mean differential
            let mut transform_means: Vec<(String, f64)> = spectrum
                .per_transformation
                .iter()
                .map(|(name, diffs)| {
                    let mean = if diffs.is_empty() {
                        0.0
                    } else {
                        diffs.iter().sum::<f64>() / diffs.len() as f64
                    };
                    (name.clone(), mean)
                })
                .collect();
            transform_means.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (name, mean) in transform_means.iter().take(3) {
                evidence.push(format!("transformation '{}': mean_diff={:.4}", name, mean));
            }
        }

        evidence
    }

    /// Get a snapshot of the current spectra for external analysis.
    pub fn get_spectra(&self) -> &HashMap<String, StageSpectrum> {
        &self.spectra
    }

    /// Get the observations collected so far.
    pub fn get_observations(&self) -> &[TestObservation] {
        &self.observations
    }

    /// Clear all recorded data and start fresh.
    pub fn reset(&mut self) {
        self.observations.clear();
        for spectrum in self.spectra.values_mut() {
            *spectrum = StageSpectrum::new(spectrum.stage_id, spectrum.stage_name.clone());
        }
    }
}

// ── Discriminability pre-check ──────────────────────────────────────────────

/// Discriminability analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscriminabilityReport {
    pub matrix_rank: usize,
    pub stage_count: usize,
    pub transformation_count: usize,
    pub is_fully_discriminable: bool,
    pub indistinguishable_groups: Vec<Vec<String>>,
    pub suggested_transformations: Vec<String>,
    pub condition_number: f64,
}

/// Compute the Stage Discriminability Matrix M ∈ ℝⁿˣᵐ and analyze its rank.
///
/// M_{k,j} = E_{x~G}[Δₖ(x, τⱼ)]
///
/// - Full rank (rank = n) means all stages can be uniquely localized.
/// - Rank < n means some stages are indistinguishable.
pub fn compute_discriminability(
    spectra: &HashMap<String, StageSpectrum>,
    stage_order: &[String],
    transformation_order: &[String],
) -> DiscriminabilityReport {
    let n = stage_order.len();
    let m = transformation_order.len();

    if n == 0 || m == 0 {
        return DiscriminabilityReport {
            matrix_rank: 0,
            stage_count: n,
            transformation_count: m,
            is_fully_discriminable: false,
            indistinguishable_groups: vec![stage_order.to_vec()],
            suggested_transformations: Vec::new(),
            condition_number: f64::INFINITY,
        };
    }

    // Build the matrix M[k][j] = mean differential of stage k under transformation j
    let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(n);
    for stage_name in stage_order {
        let mut row = Vec::with_capacity(m);
        for transform_name in transformation_order {
            let mean = spectra
                .get(stage_name)
                .and_then(|s| s.per_transformation.get(transform_name))
                .map(|diffs| {
                    if diffs.is_empty() {
                        0.0
                    } else {
                        diffs.iter().sum::<f64>() / diffs.len() as f64
                    }
                })
                .unwrap_or(0.0);
            row.push(mean);
        }
        matrix.push(row);
    }

    // Compute rank via SVD-like approach (simplified: Gaussian elimination).
    let rank = compute_matrix_rank(&matrix, 1e-6);

    // Find indistinguishable stage groups.
    let groups = find_indistinguishable_groups(&matrix, stage_order, 1e-4);

    // Compute condition number estimate.
    let condition = estimate_condition_number(&matrix);

    // Suggest additional transformations if rank-deficient.
    let suggestions = if rank < n {
        suggest_transformations(&groups)
    } else {
        Vec::new()
    };

    DiscriminabilityReport {
        matrix_rank: rank,
        stage_count: n,
        transformation_count: m,
        is_fully_discriminable: rank >= n,
        indistinguishable_groups: groups,
        suggested_transformations: suggestions,
        condition_number: condition,
    }
}

/// Compute matrix rank via Gaussian elimination with partial pivoting.
fn compute_matrix_rank(matrix: &[Vec<f64>], tolerance: f64) -> usize {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0;
    }

    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut work: Vec<Vec<f64>> = matrix.to_vec();
    let mut rank = 0;
    let mut pivot_col = 0;

    for row in 0..rows {
        if pivot_col >= cols {
            break;
        }

        // Find pivot with largest absolute value.
        let mut max_val = work[row][pivot_col].abs();
        let mut max_row = row;
        for r in (row + 1)..rows {
            let val = work[r][pivot_col].abs();
            if val > max_val {
                max_val = val;
                max_row = r;
            }
        }

        if max_val < tolerance {
            pivot_col += 1;
            continue;
        }

        // Swap rows.
        work.swap(row, max_row);

        // Eliminate below.
        let pivot = work[row][pivot_col];
        for r in (row + 1)..rows {
            let factor = work[r][pivot_col] / pivot;
            work[r][pivot_col] = 0.0;
            for c in (pivot_col + 1)..cols {
                work[r][c] -= factor * work[row][c];
            }
        }

        rank += 1;
        pivot_col += 1;
    }

    rank
}

/// Find groups of stages whose differential profiles are indistinguishable.
fn find_indistinguishable_groups(
    matrix: &[Vec<f64>],
    stage_names: &[String],
    tolerance: f64,
) -> Vec<Vec<String>> {
    let n = matrix.len();
    let mut visited = vec![false; n];
    let mut groups = Vec::new();

    for i in 0..n {
        if visited[i] {
            continue;
        }
        let mut group = vec![stage_names[i].clone()];
        visited[i] = true;

        for j in (i + 1)..n {
            if visited[j] {
                continue;
            }
            // Check if rows i and j are linearly dependent.
            let diff_norm: f64 = matrix[i]
                .iter()
                .zip(&matrix[j])
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            let norm_i: f64 = matrix[i].iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
            let norm_j: f64 = matrix[j].iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
            let max_norm = norm_i.max(norm_j);

            if max_norm < tolerance || diff_norm / max_norm < tolerance {
                group.push(stage_names[j].clone());
                visited[j] = true;
            }
        }

        if group.len() > 1 {
            groups.push(group);
        }
    }

    groups
}

/// Estimate the condition number of the matrix (ratio of largest to smallest singular value).
fn estimate_condition_number(matrix: &[Vec<f64>]) -> f64 {
    if matrix.is_empty() || matrix[0].is_empty() {
        return f64::INFINITY;
    }

    // Power iteration for largest singular value.
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut v = vec![1.0 / (cols as f64).sqrt(); cols];

    for _ in 0..50 {
        // u = M * v
        let mut u = vec![0.0; rows];
        for i in 0..rows {
            for j in 0..cols {
                u[i] += matrix[i][j] * v[j];
            }
        }
        // Normalize u
        let norm_u: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_u < f64::EPSILON {
            return f64::INFINITY;
        }
        for x in &mut u {
            *x /= norm_u;
        }
        // v = M^T * u
        v = vec![0.0; cols];
        for j in 0..cols {
            for i in 0..rows {
                v[j] += matrix[i][j] * u[i];
            }
        }
        let norm_v: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_v < f64::EPSILON {
            return f64::INFINITY;
        }
        for x in &mut v {
            *x /= norm_v;
        }
    }

    // Compute largest singular value.
    let mut mv = vec![0.0; rows];
    for i in 0..rows {
        for j in 0..cols {
            mv[i] += matrix[i][j] * v[j];
        }
    }
    let sigma_max: f64 = mv.iter().map(|x| x * x).sum::<f64>().sqrt();

    // Rough estimate of smallest singular value via inverse iteration heuristic.
    // For a proper implementation we would use full SVD; this suffices for a diagnostic.
    let sigma_min_estimate = {
        let frobenius: f64 = matrix
            .iter()
            .flat_map(|row| row.iter())
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let rank = compute_matrix_rank(matrix, 1e-10);
        if rank == 0 {
            f64::EPSILON
        } else {
            // Lower bound: frobenius / sqrt(rank) is an upper bound on the largest SV,
            // so sigma_min ≈ trace(M^T M) / (rank * sigma_max) is a rough estimate.
            let trace_mtm: f64 = matrix
                .iter()
                .flat_map(|row| row.iter())
                .map(|x| x * x)
                .sum();
            (trace_mtm / (rank as f64 * sigma_max)).max(f64::EPSILON)
        }
    };

    sigma_max / sigma_min_estimate
}

/// Suggest transformations that might help distinguish indistinguishable stage groups.
fn suggest_transformations(groups: &[Vec<String>]) -> Vec<String> {
    let mut suggestions = Vec::new();

    for group in groups {
        if group.iter().any(|s| s.contains("tagger") || s.contains("pos")) {
            suggestions.push("agreement_perturbation".to_string());
            suggestions.push("tense_change".to_string());
        }
        if group.iter().any(|s| s.contains("parser") || s.contains("dep")) {
            suggestions.push("topicalization".to_string());
            suggestions.push("pp_attachment".to_string());
            suggestions.push("embedding_depth_change".to_string());
        }
        if group.iter().any(|s| s.contains("ner") || s.contains("entity")) {
            suggestions.push("synonym_substitution".to_string());
            suggestions.push("clefting".to_string());
        }
        if group.iter().any(|s| s.contains("token")) {
            suggestions.push("negation_insertion".to_string());
        }
    }

    suggestions.sort();
    suggestions.dedup();
    suggestions
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stage_id(name: &str) -> StageId {
        StageId::new(name)
    }

    #[test]
    fn test_stage_spectrum_scoring() {
        let mut spectrum = StageSpectrum::new(make_stage_id("tagger"), "tagger".to_string());
        // Simulate 10 test cases: 7 where tagger shows high diff + violation
        for _ in 0..7 {
            spectrum.record(0.8, true, 0.5);
        }
        // 2 where tagger shows low diff + no violation
        for _ in 0..2 {
            spectrum.record(0.1, false, 0.5);
        }
        // 1 where tagger shows high diff + no violation
        spectrum.record(0.6, false, 0.5);

        assert_eq!(spectrum.total(), 10);
        assert_eq!(spectrum.ef, 7);
        assert_eq!(spectrum.ep, 1);
        assert_eq!(spectrum.nf, 0);
        assert_eq!(spectrum.np, 2);

        let ochiai = spectrum.ochiai_score();
        assert!(ochiai > 0.8, "Expected high Ochiai score, got {}", ochiai);

        let tarantula = spectrum.tarantula_score();
        assert!(tarantula > 0.7, "Expected high Tarantula score, got {}", tarantula);
    }

    #[test]
    fn test_stage_spectrum_mean_variance() {
        let mut spectrum = StageSpectrum::new(make_stage_id("parser"), "parser".to_string());
        spectrum.record(0.5, true, 0.3);
        spectrum.record(0.7, true, 0.3);
        spectrum.record(0.3, false, 0.3);

        let mean = spectrum.mean_differential();
        assert!((mean - 0.5).abs() < 0.01);

        let var = spectrum.variance();
        assert!(var > 0.0);
    }

    #[test]
    fn test_engine_basic_flow() {
        let mut engine = LocalizationEngine::new();
        engine.register_stages(vec![
            (make_stage_id("tokenizer"), "tokenizer".to_string()),
            (make_stage_id("tagger"), "tagger".to_string()),
            (make_stage_id("parser"), "parser".to_string()),
            (make_stage_id("ner"), "ner".to_string()),
        ]);

        // Simulate observations where the tagger is the faulty stage.
        for i in 0..20 {
            let violation = i % 3 != 0;
            let mut diffs = HashMap::new();
            diffs.insert("tokenizer".to_string(), 0.05);
            diffs.insert("tagger".to_string(), if violation { 0.85 } else { 0.1 });
            diffs.insert("parser".to_string(), if violation { 0.6 } else { 0.08 });
            diffs.insert("ner".to_string(), if violation { 0.4 } else { 0.05 });

            engine.record_observation(TestObservation {
                test_id: format!("test_{}", i),
                transformation_name: "passivization".to_string(),
                input_text: format!("The cat sat on the mat {}", i),
                transformed_text: format!("The mat was sat on by the cat {}", i),
                violation_detected: violation,
                violation_magnitude: if violation { 0.7 } else { 0.0 },
                per_stage_differentials: diffs,
                execution_time_ms: 5.0,
            });
        }

        let result = engine.run_analysis().expect("analysis should succeed");
        assert_eq!(result.test_count, 20);
        assert!(result.violation_count > 0);
        // The tagger should be ranked highest.
        assert_eq!(result.stage_results[0].stage_name, "tagger");
    }

    #[test]
    fn test_causal_intervention_introduced() {
        let engine = LocalizationEngine::new();
        let intervention =
            engine.analyze_causal_intervention("tagger", 0.8, 0.0);
        assert!(matches!(intervention.verdict, CausalVerdict::Introduced));
        assert!((intervention.attenuation_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_causal_intervention_amplified() {
        let engine = LocalizationEngine::new();
        let intervention =
            engine.analyze_causal_intervention("parser", 0.8, 0.2);
        assert!(matches!(intervention.verdict, CausalVerdict::Amplified { .. }));
        assert!(intervention.attenuation_ratio > 0.5);
    }

    #[test]
    fn test_causal_intervention_not_causal() {
        let engine = LocalizationEngine::new();
        let intervention =
            engine.analyze_causal_intervention("tokenizer", 0.8, 0.78);
        assert!(matches!(intervention.verdict, CausalVerdict::NotCausal));
    }

    #[test]
    fn test_iterative_peeling() {
        let engine = LocalizationEngine::new();

        let results = vec![
            ("tagger".to_string(), 0.3),
            ("parser".to_string(), 0.5),
            ("ner".to_string(), 0.7),
        ];

        let state = engine.iterative_peeling(0.8, &results);
        assert!(!state.rounds.is_empty());
        // The tagger causes the largest attenuation (0.8 - 0.3 = 0.5)
        assert_eq!(state.rounds[0].suspected_stage, "tagger");
    }

    #[test]
    fn test_discriminability_full_rank() {
        let mut spectra = HashMap::new();

        let mut s1 = StageSpectrum::new(make_stage_id("tagger"), "tagger".to_string());
        s1.per_transformation
            .insert("passivization".to_string(), vec![0.8, 0.7, 0.9]);
        s1.per_transformation
            .insert("clefting".to_string(), vec![0.2, 0.1, 0.3]);
        spectra.insert("tagger".to_string(), s1);

        let mut s2 = StageSpectrum::new(make_stage_id("parser"), "parser".to_string());
        s2.per_transformation
            .insert("passivization".to_string(), vec![0.3, 0.2, 0.4]);
        s2.per_transformation
            .insert("clefting".to_string(), vec![0.9, 0.8, 0.7]);
        spectra.insert("parser".to_string(), s2);

        let report = compute_discriminability(
            &spectra,
            &["tagger".to_string(), "parser".to_string()],
            &["passivization".to_string(), "clefting".to_string()],
        );

        assert!(report.is_fully_discriminable);
        assert_eq!(report.matrix_rank, 2);
        assert!(report.indistinguishable_groups.is_empty());
    }

    #[test]
    fn test_discriminability_rank_deficient() {
        let mut spectra = HashMap::new();

        // Two stages with identical differential profiles → rank 1.
        let mut s1 = StageSpectrum::new(make_stage_id("tagger"), "tagger".to_string());
        s1.per_transformation
            .insert("passivization".to_string(), vec![0.5, 0.5, 0.5]);
        spectra.insert("tagger".to_string(), s1);

        let mut s2 = StageSpectrum::new(make_stage_id("parser"), "parser".to_string());
        s2.per_transformation
            .insert("passivization".to_string(), vec![0.5, 0.5, 0.5]);
        spectra.insert("parser".to_string(), s2);

        let report = compute_discriminability(
            &spectra,
            &["tagger".to_string(), "parser".to_string()],
            &["passivization".to_string()],
        );

        assert!(!report.is_fully_discriminable);
        assert_eq!(report.matrix_rank, 1);
        assert!(!report.indistinguishable_groups.is_empty());
    }

    #[test]
    fn test_matrix_rank() {
        // Full rank 3x3.
        let m = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        assert_eq!(compute_matrix_rank(&m, 1e-6), 3);

        // Rank 2 (third row is sum of first two).
        let m2 = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
        ];
        assert_eq!(compute_matrix_rank(&m2, 1e-6), 2);

        // Rank 1.
        let m3 = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert_eq!(compute_matrix_rank(&m3, 1e-6), 1);
    }

    #[test]
    fn test_ensemble_scoring() {
        let config = LocalizationConfig {
            sbfl_metric: SBFLMetric::Ensemble,
            ..Default::default()
        };
        let mut engine = LocalizationEngine::with_config(config);
        engine.register_stages(vec![
            (make_stage_id("a"), "a".to_string()),
            (make_stage_id("b"), "b".to_string()),
        ]);

        let mut diffs = HashMap::new();
        diffs.insert("a".to_string(), 0.9);
        diffs.insert("b".to_string(), 0.1);

        engine.record_observation(TestObservation {
            test_id: "t1".to_string(),
            transformation_name: "passivization".to_string(),
            input_text: "x".to_string(),
            transformed_text: "y".to_string(),
            violation_detected: true,
            violation_magnitude: 0.5,
            per_stage_differentials: diffs,
            execution_time_ms: 1.0,
        });

        let ranking = engine.compute_suspiciousness();
        assert_eq!(ranking.rankings[0].stage_name, "a");
    }

    #[test]
    fn test_empty_engine_returns_error() {
        let engine = LocalizationEngine::new();
        let result = engine.run_analysis();
        assert!(result.is_err());
    }

    #[test]
    fn test_reset() {
        let mut engine = LocalizationEngine::new();
        engine.register_stages(vec![(make_stage_id("a"), "a".to_string())]);

        let mut diffs = HashMap::new();
        diffs.insert("a".to_string(), 0.5);

        engine.record_observation(TestObservation {
            test_id: "t".to_string(),
            transformation_name: "pass".to_string(),
            input_text: "x".to_string(),
            transformed_text: "y".to_string(),
            violation_detected: true,
            violation_magnitude: 0.5,
            per_stage_differentials: diffs,
            execution_time_ms: 1.0,
        });

        assert_eq!(engine.get_observations().len(), 1);
        engine.reset();
        assert_eq!(engine.get_observations().len(), 0);
    }
}
