//! DStar metric with tuneable exponent adapted for continuous differentials.
//!
//! Classic DStar (binary):
//!   `S_k = (e_f(k))^* / ( n_f(k) + e_p(k) )`
//!
//! Continuous adaptation:
//!   `S_k = (Σ_{v_i=1} D_{i,k})^* / (Σ_{v_i=0} D_{i,k} + (Σ_i D_{i,k} - Σ_{v_i=1} D_{i,k}))`

use crate::{DifferentialMatrix, ViolationVector};
use crate::ochiai::SuspiciousnessScore;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use shared_types::{LocalizerError, Result};

// ── Public types ────────────────────────────────────────────────────────────

/// Configuration for the DStar metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DStarMetric {
    pub star_parameter: f64,
    pub calibration_baselines: Option<Vec<f64>>,
    pub normalization_enabled: bool,
}

impl Default for DStarMetric {
    fn default() -> Self {
        Self {
            star_parameter: 2.0,
            calibration_baselines: None,
            normalization_enabled: true,
        }
    }
}

/// Result of DStar computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DStarResult {
    pub scores: Vec<SuspiciousnessScore>,
    pub ranked_stages: Vec<usize>,
    pub top_suspect: usize,
    pub separation_ratio: f64,
    pub star_used: f64,
}

/// Utility for cross-validating the star parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterTuning {
    pub candidates: Vec<f64>,
    pub n_folds: usize,
}

/// Result from computing the EXAM score against known ground truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExamScore {
    pub score: f64,
    pub stages_examined: usize,
    pub total_stages: usize,
}

impl Default for ParameterTuning {
    fn default() -> Self {
        Self {
            candidates: vec![1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            n_folds: 5,
        }
    }
}

// ── Implementation ──────────────────────────────────────────────────────────

impl DStarMetric {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_star(mut self, star: f64) -> Self {
        self.star_parameter = star;
        self
    }

    pub fn with_calibration(mut self, baselines: Vec<f64>) -> Self {
        self.calibration_baselines = Some(baselines);
        self
    }

    pub fn with_normalization(mut self, enabled: bool) -> Self {
        self.normalization_enabled = enabled;
        self
    }

    /// Compute DStar suspiciousness for every stage.
    pub fn compute_suspiciousness(
        &self,
        matrix: &DifferentialMatrix,
        violations: &ViolationVector,
    ) -> Result<DStarResult> {
        self.validate_inputs(matrix, violations)?;

        let n_stages = matrix.n_stages;
        let n_fail = violations.n_violations();

        if n_fail == 0 {
            return self.handle_no_violations(matrix);
        }

        let mut raw_scores = Vec::with_capacity(n_stages);

        for k in 0..n_stages {
            let col = self.adjusted_column(matrix, k);
            let score = self.dstar_score_for_col(&col, &violations.violations);
            raw_scores.push(score);
        }

        if self.normalization_enabled {
            let max_s = raw_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_s > 0.0 {
                for s in &mut raw_scores {
                    *s /= max_s;
                }
            }
        }

        let ranked = rank_by_score(&raw_scores);
        let top = ranked[0];
        let sep = separation_ratio(&raw_scores, &ranked);

        let scores = build_scores(&raw_scores, &ranked, &matrix.stage_names);

        Ok(DStarResult {
            scores,
            ranked_stages: ranked,
            top_suspect: top,
            separation_ratio: sep,
            star_used: self.star_parameter,
        })
    }

    /// Compute the EXAM score: fraction of stages examined before the fault is found.
    pub fn compute_exam_score(
        &self,
        matrix: &DifferentialMatrix,
        violations: &ViolationVector,
        ground_truth_stage: usize,
    ) -> Result<ExamScore> {
        let result = self.compute_suspiciousness(matrix, violations)?;
        let pos = result
            .ranked_stages
            .iter()
            .position(|&s| s == ground_truth_stage)
            .unwrap_or(result.ranked_stages.len());
        let examined = pos + 1;
        Ok(ExamScore {
            score: examined as f64 / matrix.n_stages as f64,
            stages_examined: examined,
            total_stages: matrix.n_stages,
        })
    }

    // ── Internal helpers ────────────────────────────────────────────────

    fn dstar_score_for_col(&self, col: &[f64], violations: &[bool]) -> f64 {
        let ef: f64 = col
            .iter()
            .zip(violations.iter())
            .filter(|(_, &v)| v)
            .map(|(d, _)| *d)
            .sum();
        let ep: f64 = col
            .iter()
            .zip(violations.iter())
            .filter(|(_, &v)| !v)
            .map(|(d, _)| *d)
            .sum();
        let total: f64 = col.iter().sum();
        let nf = total - ef; // differential contribution from failing tests that didn't contribute to ef...
        // In continuous version: nf = total - ef (sum over all) - ep is sum over passing
        // Actually DStar denominator = n_f(k) + e_p(k)
        // n_f(k) = failing tests where stage had LOW differential = |{v=1}| - ef (in binary)
        // In continuous: denominator = ep + (total_failing_contribution_missed)
        // Simplification: denom = ep + max(0, total - ef - ep) but total-ef-ep = 0 always!
        // Actually total = ef + ep, so denom = ep + (total - ef) = ep + ep = 2*ep... no.
        //
        // Correct continuous adaptation:
        //   numerator = ef^star
        //   denominator = ep + (n_fail_count * max_possible_diff - ef)  -- but this is ad hoc
        //
        // Standard approach: denominator = (Σ_{v=0} D_{ik}) + (count_fail - Σ_{v=1} D_{ik}/max_D)
        // Simplest faithful adaptation:
        //   denom = ep + epsilon   (avoid division by zero, captures that ep is the "noise")
        let denom = ep + 1e-10;
        if ef < f64::EPSILON {
            0.0
        } else {
            ef.powf(self.star_parameter) / denom
        }
    }

    fn adjusted_column(&self, matrix: &DifferentialMatrix, k: usize) -> Vec<f64> {
        let col = matrix.column(k);
        match &self.calibration_baselines {
            Some(baselines) if k < baselines.len() => {
                let b = baselines[k];
                col.into_iter().map(|d| (d - b).max(0.0)).collect()
            }
            _ => col,
        }
    }

    fn validate_inputs(
        &self,
        matrix: &DifferentialMatrix,
        violations: &ViolationVector,
    ) -> Result<()> {
        if matrix.n_tests != violations.len() {
            return Err(LocalizerError::matrix(
                format!(
                    "matrix has {} rows but violation vector has {} entries",
                    matrix.n_tests,
                    violations.len()
                ),
                matrix.n_tests,
                matrix.n_stages,
            ));
        }
        if matrix.n_stages == 0 {
            return Err(LocalizerError::matrix("zero stages", matrix.n_tests, 0));
        }
        if self.star_parameter <= 0.0 {
            return Err(LocalizerError::validation(
                "dstar",
                "star parameter must be positive",
            ));
        }
        Ok(())
    }

    fn handle_no_violations(&self, matrix: &DifferentialMatrix) -> Result<DStarResult> {
        let n = matrix.n_stages;
        let scores = (0..n)
            .map(|k| SuspiciousnessScore {
                stage_index: k,
                stage_name: matrix.stage_names[k].clone(),
                score: 0.0,
                rank: 1,
                confidence_interval: (0.0, 0.0),
            })
            .collect();
        Ok(DStarResult {
            scores,
            ranked_stages: (0..n).collect(),
            top_suspect: 0,
            separation_ratio: 1.0,
            star_used: self.star_parameter,
        })
    }
}

impl ParameterTuning {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_candidates(mut self, candidates: Vec<f64>) -> Self {
        self.candidates = candidates;
        self
    }

    /// Cross-validate the star parameter using calibration data.
    ///
    /// `ground_truth_stage` is the known faulty stage index.
    /// Returns the star value that minimises the average EXAM score.
    pub fn cross_validate_star(
        &self,
        matrix: &DifferentialMatrix,
        violations: &ViolationVector,
        ground_truth_stage: usize,
    ) -> Result<f64> {
        if matrix.n_tests < self.n_folds {
            return Err(LocalizerError::validation(
                "parameter_tuning",
                format!("need at least {} tests for {}-fold CV", self.n_folds, self.n_folds),
            ));
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut indices: Vec<usize> = (0..matrix.n_tests).collect();
        indices.shuffle(&mut rng);

        let fold_size = matrix.n_tests / self.n_folds;
        let mut best_star = self.candidates[0];
        let mut best_exam = f64::INFINITY;

        for &star in &self.candidates {
            let metric = DStarMetric::new().with_star(star);
            let mut total_exam = 0.0;
            let mut valid_folds = 0usize;

            for fold in 0..self.n_folds {
                let test_start = fold * fold_size;
                let test_end = if fold == self.n_folds - 1 {
                    matrix.n_tests
                } else {
                    test_start + fold_size
                };

                let train_indices: Vec<usize> = indices
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i < test_start || *i >= test_end)
                    .map(|(_, &v)| v)
                    .collect();

                if train_indices.is_empty() {
                    continue;
                }

                let train_data: Vec<Vec<f64>> =
                    train_indices.iter().map(|&i| matrix.data[i].clone()).collect();
                let train_violations: Vec<bool> =
                    train_indices.iter().map(|&i| violations.violations[i]).collect();

                if train_violations.iter().all(|v| !v) {
                    continue;
                }

                let train_matrix =
                    DifferentialMatrix::new(train_data, matrix.stage_names.clone())?;
                let train_v = ViolationVector::new(train_violations);

                if let Ok(exam) =
                    metric.compute_exam_score(&train_matrix, &train_v, ground_truth_stage)
                {
                    total_exam += exam.score;
                    valid_folds += 1;
                }
            }

            if valid_folds > 0 {
                let avg_exam = total_exam / valid_folds as f64;
                if avg_exam < best_exam {
                    best_exam = avg_exam;
                    best_star = star;
                }
            }
        }

        Ok(best_star)
    }
}

// ── Shared ranking helpers ──────────────────────────────────────────────────

pub(crate) fn rank_by_score(scores: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    indices
}

pub(crate) fn separation_ratio(scores: &[f64], ranked: &[usize]) -> f64 {
    if ranked.len() < 2 {
        return f64::INFINITY;
    }
    let first = scores[ranked[0]];
    let second = scores[ranked[1]];
    if second.abs() < f64::EPSILON {
        if first.abs() < f64::EPSILON {
            1.0
        } else {
            f64::INFINITY
        }
    } else {
        first / second
    }
}

pub(crate) fn build_scores(
    raw: &[f64],
    ranked: &[usize],
    names: &[String],
) -> Vec<SuspiciousnessScore> {
    (0..raw.len())
        .map(|k| SuspiciousnessScore {
            stage_index: k,
            stage_name: names[k].clone(),
            score: raw[k],
            rank: ranked.iter().position(|&r| r == k).unwrap() + 1,
            confidence_interval: (0.0, 0.0),
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_matrix() -> DifferentialMatrix {
        let data = vec![
            vec![0.9, 0.1, 0.2],
            vec![0.8, 0.2, 0.1],
            vec![0.7, 0.0, 0.3],
            vec![0.1, 0.4, 0.1],
            vec![0.0, 0.3, 0.0],
            vec![0.2, 0.5, 0.1],
        ];
        let names = vec!["tokenizer".into(), "pos_tagger".into(), "parser".into()];
        DifferentialMatrix::new(data, names).unwrap()
    }

    fn sample_violations() -> ViolationVector {
        ViolationVector::new(vec![true, true, true, false, false, false])
    }

    #[test]
    fn test_basic_dstar() {
        let m = DStarMetric::new();
        let result = m
            .compute_suspiciousness(&sample_matrix(), &sample_violations())
            .unwrap();
        assert_eq!(result.scores.len(), 3);
        assert_eq!(result.top_suspect, 0);
    }

    #[test]
    fn test_star_parameter_effect() {
        let m1 = DStarMetric::new().with_star(1.0).with_normalization(false);
        let m2 = DStarMetric::new().with_star(3.0).with_normalization(false);
        let matrix = sample_matrix();
        let v = sample_violations();
        let r1 = m1.compute_suspiciousness(&matrix, &v).unwrap();
        let r2 = m2.compute_suspiciousness(&matrix, &v).unwrap();
        // Higher star amplifies the gap between high and low ef
        let gap1 = r1.scores[0].score - r1.scores[1].score;
        let gap2 = r2.scores[0].score - r2.scores[1].score;
        assert!(gap2 > gap1, "higher star should increase separation");
    }

    #[test]
    fn test_dstar_no_violations() {
        let m = DStarMetric::new();
        let matrix = sample_matrix();
        let v = ViolationVector::new(vec![false; 6]);
        let result = m.compute_suspiciousness(&matrix, &v).unwrap();
        for s in &result.scores {
            assert_eq!(s.score, 0.0);
        }
    }

    #[test]
    fn test_exam_score() {
        let m = DStarMetric::new();
        let exam = m
            .compute_exam_score(&sample_matrix(), &sample_violations(), 0)
            .unwrap();
        assert_eq!(exam.stages_examined, 1); // stage 0 should be top-ranked
        assert!((exam.score - 1.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_exam_score_worst_case() {
        let m = DStarMetric::new();
        let exam = m
            .compute_exam_score(&sample_matrix(), &sample_violations(), 1)
            .unwrap();
        assert!(exam.score > 1.0 / 3.0); // pos_tagger not top-ranked
    }

    #[test]
    fn test_dimension_mismatch() {
        let m = DStarMetric::new();
        let matrix = sample_matrix();
        let v = ViolationVector::new(vec![true]);
        assert!(m.compute_suspiciousness(&matrix, &v).is_err());
    }

    #[test]
    fn test_invalid_star() {
        let m = DStarMetric::new().with_star(-1.0);
        let matrix = sample_matrix();
        let v = sample_violations();
        assert!(m.compute_suspiciousness(&matrix, &v).is_err());
    }

    #[test]
    fn test_calibration_baselines() {
        let m = DStarMetric::new().with_calibration(vec![0.5, 0.0, 0.0]);
        let result = m
            .compute_suspiciousness(&sample_matrix(), &sample_violations())
            .unwrap();
        assert!(result.scores[0].score >= 0.0);
    }

    #[test]
    fn test_cross_validation() {
        let tuner = ParameterTuning::new().with_candidates(vec![1.0, 2.0, 3.0]);
        let matrix = sample_matrix();
        let v = sample_violations();
        let best = tuner.cross_validate_star(&matrix, &v, 0).unwrap();
        assert!(best >= 1.0 && best <= 3.0);
    }

    #[test]
    fn test_normalization_toggle() {
        let m = DStarMetric::new().with_normalization(false);
        let result = m
            .compute_suspiciousness(&sample_matrix(), &sample_violations())
            .unwrap();
        // Un-normalized scores can exceed 1.0 (no upper bound guarantee)
        assert!(result.scores[0].score >= 0.0);
    }
}
