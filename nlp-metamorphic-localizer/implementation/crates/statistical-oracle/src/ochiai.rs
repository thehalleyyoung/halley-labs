//! Ochiai coefficient adapted for continuous-valued pipeline-stage differentials.
//!
//! Classic Ochiai (binary):  `S_k = e_f(k) / sqrt(e_f(k) + n_f(k)) * sqrt(e_f(k) + e_p(k))`
//!
//! Continuous adaptation:
//!   `S_k = Σ_{v_i=1} D_{i,k}  /  sqrt( (Σ_i D_{i,k}) · |{v_i=1}| )`

use crate::{DifferentialMatrix, ViolationVector};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use shared_types::{LocalizerError, Result};

// ── Public types ────────────────────────────────────────────────────────────

/// Per-stage suspiciousness score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspiciousnessScore {
    pub stage_index: usize,
    pub stage_name: String,
    pub score: f64,
    pub rank: usize,
    pub confidence_interval: (f64, f64),
}

/// Full result from Ochiai computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OchiaiResult {
    pub scores: Vec<SuspiciousnessScore>,
    pub ranked_stages: Vec<usize>,
    pub top_suspect: usize,
    pub separation_ratio: f64,
}

/// Configuration for the Ochiai metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OchiaiMetric {
    pub calibration_baselines: Option<Vec<f64>>,
    pub normalization_enabled: bool,
    pub bootstrap_resamples: usize,
    pub confidence_level: f64,
}

impl Default for OchiaiMetric {
    fn default() -> Self {
        Self {
            calibration_baselines: None,
            normalization_enabled: true,
            bootstrap_resamples: 1000,
            confidence_level: 0.95,
        }
    }
}

impl OchiaiMetric {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_calibration(mut self, baselines: Vec<f64>) -> Self {
        self.calibration_baselines = Some(baselines);
        self
    }

    pub fn with_normalization(mut self, enabled: bool) -> Self {
        self.normalization_enabled = enabled;
        self
    }

    pub fn with_bootstrap(mut self, n: usize, level: f64) -> Self {
        self.bootstrap_resamples = n;
        self.confidence_level = level;
        self
    }

    // ── Core computation ────────────────────────────────────────────────

    /// Compute Ochiai suspiciousness for every stage.
    pub fn compute_suspiciousness(
        &self,
        matrix: &DifferentialMatrix,
        violations: &ViolationVector,
    ) -> Result<OchiaiResult> {
        self.validate_inputs(matrix, violations)?;

        let n_stages = matrix.n_stages;
        let n_fail = violations.n_violations() as f64;

        if n_fail == 0.0 {
            return self.handle_no_violations(matrix);
        }

        let mut raw_scores: Vec<f64> = Vec::with_capacity(n_stages);

        for k in 0..n_stages {
            let col = self.adjusted_column(matrix, k);

            let ef: f64 = col
                .iter()
                .zip(violations.violations.iter())
                .filter(|(_, &v)| v)
                .map(|(&d, _)| d)
                .sum();

            let total: f64 = col.iter().sum();

            let score = self.ochiai_score(ef, total, n_fail);
            raw_scores.push(score);
        }

        if self.normalization_enabled {
            let max_score = raw_scores
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            if max_score > 0.0 {
                for s in &mut raw_scores {
                    *s /= max_score;
                }
            }
        }

        let ranked = self.rank_stages(&raw_scores);
        let cis = self.compute_all_cis(matrix, violations, &raw_scores);

        let scores: Vec<SuspiciousnessScore> = (0..n_stages)
            .map(|k| SuspiciousnessScore {
                stage_index: k,
                stage_name: matrix.stage_names[k].clone(),
                score: raw_scores[k],
                rank: ranked.iter().position(|&r| r == k).unwrap() + 1,
                confidence_interval: cis[k],
            })
            .collect();

        let top = ranked[0];
        let sep = self.separation_ratio(&raw_scores, &ranked);

        Ok(OchiaiResult {
            scores,
            ranked_stages: ranked,
            top_suspect: top,
            separation_ratio: sep,
        })
    }

    // ── Internal helpers ────────────────────────────────────────────────

    fn ochiai_score(&self, ef: f64, total_col: f64, n_fail: f64) -> f64 {
        let denom = (total_col * n_fail).sqrt();
        if denom < f64::EPSILON {
            0.0
        } else {
            ef / denom
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
            return Err(LocalizerError::matrix(
                "matrix has zero stages",
                matrix.n_tests,
                0,
            ));
        }
        Ok(())
    }

    fn handle_no_violations(&self, matrix: &DifferentialMatrix) -> Result<OchiaiResult> {
        let n = matrix.n_stages;
        let scores: Vec<SuspiciousnessScore> = (0..n)
            .map(|k| SuspiciousnessScore {
                stage_index: k,
                stage_name: matrix.stage_names[k].clone(),
                score: 0.0,
                rank: 1,
                confidence_interval: (0.0, 0.0),
            })
            .collect();
        Ok(OchiaiResult {
            scores,
            ranked_stages: (0..n).collect(),
            top_suspect: 0,
            separation_ratio: 1.0,
        })
    }

    /// Sort stages by descending score, breaking ties by index.
    pub fn rank_stages(&self, scores: &[f64]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        indices
    }

    /// Ratio of the top score to the second-best score.
    pub fn separation_ratio(&self, scores: &[f64], ranked: &[usize]) -> f64 {
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

    // ── Bootstrap confidence intervals ──────────────────────────────────

    fn compute_all_cis(
        &self,
        matrix: &DifferentialMatrix,
        violations: &ViolationVector,
        _point_estimates: &[f64],
    ) -> Vec<(f64, f64)> {
        if self.bootstrap_resamples == 0 || matrix.n_tests < 3 {
            return vec![(0.0, 1.0); matrix.n_stages];
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let b = self.bootstrap_resamples;
        let n_stages = matrix.n_stages;
        let n_tests = matrix.n_tests;
        let n_fail = violations.n_violations() as f64;

        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(b); n_stages];

        let indices: Vec<usize> = (0..n_tests).collect();
        for _ in 0..b {
            let resampled: Vec<usize> = (0..n_tests)
                .map(|_| *indices.choose(&mut rng).unwrap())
                .collect();

            let boot_nfail: f64 = resampled.iter().filter(|&&i| violations.violations[i]).count() as f64;
            if boot_nfail < f64::EPSILON {
                for k in 0..n_stages {
                    samples[k].push(0.0);
                }
                continue;
            }

            for k in 0..n_stages {
                let col = self.adjusted_column(matrix, k);
                let ef: f64 = resampled
                    .iter()
                    .filter(|&&i| violations.violations[i])
                    .map(|&i| col[i])
                    .sum();
                let total: f64 = resampled.iter().map(|&i| col[i]).sum();
                let score = self.ochiai_score(ef, total, boot_nfail);
                samples[k].push(score);
            }
        }

        let alpha = 1.0 - self.confidence_level;
        samples
            .iter_mut()
            .map(|s| {
                s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let lo_idx = ((alpha / 2.0) * s.len() as f64).floor() as usize;
                let hi_idx = ((1.0 - alpha / 2.0) * s.len() as f64).ceil() as usize;
                let lo = s.get(lo_idx).copied().unwrap_or(0.0);
                let hi = s.get(hi_idx.min(s.len() - 1)).copied().unwrap_or(1.0);
                (lo, hi)
            })
            .collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_matrix() -> DifferentialMatrix {
        // 6 tests × 3 stages
        let data = vec![
            vec![0.8, 0.1, 0.2],
            vec![0.7, 0.2, 0.1],
            vec![0.9, 0.0, 0.3],
            vec![0.1, 0.1, 0.1],
            vec![0.0, 0.3, 0.0],
            vec![0.2, 0.2, 0.1],
        ];
        let names = vec!["tokenizer".into(), "pos_tagger".into(), "parser".into()];
        DifferentialMatrix::new(data, names).unwrap()
    }

    fn sample_violations() -> ViolationVector {
        ViolationVector::new(vec![true, true, true, false, false, false])
    }

    #[test]
    fn test_basic_ochiai() {
        let m = OchiaiMetric::new();
        let result = m
            .compute_suspiciousness(&sample_matrix(), &sample_violations())
            .unwrap();
        assert_eq!(result.scores.len(), 3);
        assert_eq!(result.top_suspect, 0); // tokenizer has highest differential on violations
    }

    #[test]
    fn test_ranking_order() {
        let m = OchiaiMetric::new();
        let result = m
            .compute_suspiciousness(&sample_matrix(), &sample_violations())
            .unwrap();
        assert_eq!(result.ranked_stages[0], 0);
        // Score of rank 1 >= score of rank 2
        let s0 = result.scores[result.ranked_stages[0]].score;
        let s1 = result.scores[result.ranked_stages[1]].score;
        assert!(s0 >= s1);
    }

    #[test]
    fn test_separation_ratio_basic() {
        let m = OchiaiMetric::new();
        let result = m
            .compute_suspiciousness(&sample_matrix(), &sample_violations())
            .unwrap();
        assert!(result.separation_ratio >= 1.0);
    }

    #[test]
    fn test_no_violations() {
        let m = OchiaiMetric::new();
        let matrix = sample_matrix();
        let v = ViolationVector::new(vec![false; 6]);
        let result = m.compute_suspiciousness(&matrix, &v).unwrap();
        for s in &result.scores {
            assert_eq!(s.score, 0.0);
        }
    }

    #[test]
    fn test_all_violations() {
        let m = OchiaiMetric::new();
        let matrix = sample_matrix();
        let v = ViolationVector::new(vec![true; 6]);
        let result = m.compute_suspiciousness(&matrix, &v).unwrap();
        // All violations ⟹ ef == total_col for each stage, score = sqrt(n_fail/total)
        for s in &result.scores {
            assert!(s.score >= 0.0);
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let m = OchiaiMetric::new();
        let matrix = sample_matrix();
        let v = ViolationVector::new(vec![true, false]);
        assert!(m.compute_suspiciousness(&matrix, &v).is_err());
    }

    #[test]
    fn test_calibration_baselines() {
        let m = OchiaiMetric::new().with_calibration(vec![0.5, 0.0, 0.0]);
        let result = m
            .compute_suspiciousness(&sample_matrix(), &sample_violations())
            .unwrap();
        // With baseline subtraction, stage 0 raw scores are reduced
        assert!(result.scores[0].score >= 0.0);
    }

    #[test]
    fn test_confidence_intervals() {
        let m = OchiaiMetric::new().with_bootstrap(200, 0.95);
        let result = m
            .compute_suspiciousness(&sample_matrix(), &sample_violations())
            .unwrap();
        for s in &result.scores {
            let (lo, hi) = s.confidence_interval;
            assert!(lo <= hi, "CI lower {} > upper {}", lo, hi);
        }
    }

    #[test]
    fn test_normalization_disabled() {
        let m = OchiaiMetric::new().with_normalization(false);
        let result = m
            .compute_suspiciousness(&sample_matrix(), &sample_violations())
            .unwrap();
        // Without normalization scores are the raw Ochiai values (≤ 1 anyway for well-behaved data)
        assert!(result.scores[0].score >= 0.0);
    }

    #[test]
    fn test_rank_stages_with_ties() {
        let m = OchiaiMetric::new();
        let scores = vec![0.5, 0.5, 0.3, 0.9];
        let ranked = m.rank_stages(&scores);
        assert_eq!(ranked[0], 3); // highest
        // ties broken by index
        assert_eq!(ranked[1], 0);
        assert_eq!(ranked[2], 1);
        assert_eq!(ranked[3], 2);
    }

    #[test]
    fn test_single_stage() {
        let data = vec![vec![0.9], vec![0.1]];
        let names = vec!["only_stage".into()];
        let matrix = DifferentialMatrix::new(data, names).unwrap();
        let v = ViolationVector::new(vec![true, false]);
        let m = OchiaiMetric::new();
        let result = m.compute_suspiciousness(&matrix, &v).unwrap();
        assert_eq!(result.scores.len(), 1);
        assert_eq!(result.top_suspect, 0);
    }

    #[test]
    fn test_zero_column() {
        let data = vec![
            vec![0.0, 0.5],
            vec![0.0, 0.3],
            vec![0.0, 0.8],
        ];
        let names = vec!["dead_stage".into(), "live_stage".into()];
        let matrix = DifferentialMatrix::new(data, names).unwrap();
        let v = ViolationVector::new(vec![true, false, true]);
        let m = OchiaiMetric::new();
        let result = m.compute_suspiciousness(&matrix, &v).unwrap();
        assert_eq!(result.scores[0].score, 0.0);
        assert!(result.scores[1].score > 0.0);
    }
}
