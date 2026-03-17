//! Tarantula metric adapted for continuous-valued differentials.
//!
//! Classic Tarantula (binary):
//!   `S_k = (e_f / total_f) / ( e_f / total_f + e_p / total_p )`
//!
//! Continuous adaptation:
//!   `S_k = (Σ_{v=1} D_{ik} / |{v=1}|) / (Σ_{v=1} D_{ik}/|{v=1}| + Σ_{v=0} D_{ik}/|{v=0}|)`

use crate::dstar::{build_scores, rank_by_score, separation_ratio};
use crate::ochiai::SuspiciousnessScore;
use crate::{DifferentialMatrix, ViolationVector};
use serde::{Deserialize, Serialize};
use shared_types::{LocalizerError, Result};

// ── Public types ────────────────────────────────────────────────────────────

/// Configuration for the Tarantula metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TarantulaMetric {
    pub calibration_baselines: Option<Vec<f64>>,
    pub normalization_enabled: bool,
}

impl Default for TarantulaMetric {
    fn default() -> Self {
        Self {
            calibration_baselines: None,
            normalization_enabled: true,
        }
    }
}

/// Full Tarantula result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TarantulaResult {
    pub scores: Vec<SuspiciousnessScore>,
    pub ranked_stages: Vec<usize>,
    pub top_suspect: usize,
    pub separation_ratio: f64,
    pub colors: Vec<SuspiciousnessColor>,
}

/// RGB color representing suspiciousness (red = high, green = low).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SuspiciousnessColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl SuspiciousnessColor {
    /// Map a suspiciousness value in [0, 1] to a colour on the green→yellow→red
    /// spectrum using hue interpolation.
    pub fn from_score(score: f64) -> Self {
        let s = score.clamp(0.0, 1.0);
        // Hue goes from 120° (green) at s=0 to 0° (red) at s=1.
        let hue = 120.0 * (1.0 - s);
        let (r, g, b) = hsv_to_rgb(hue, 0.9, 0.95);
        Self { r, g, b }
    }

    pub fn to_hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

/// Correlation between two ranking vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankCorrelation {
    pub spearman_rho: f64,
    pub kendall_tau: f64,
    pub top_k_overlap: f64,
}

// ── Implementation ──────────────────────────────────────────────────────────

impl TarantulaMetric {
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

    /// Compute Tarantula suspiciousness for every stage.
    pub fn compute_suspiciousness(
        &self,
        matrix: &DifferentialMatrix,
        violations: &ViolationVector,
    ) -> Result<TarantulaResult> {
        self.validate(matrix, violations)?;

        let n_stages = matrix.n_stages;
        let n_fail = violations.n_violations() as f64;
        let n_pass = violations.n_passing() as f64;

        if n_fail < f64::EPSILON {
            return self.handle_no_violations(matrix);
        }

        let mut raw_scores = Vec::with_capacity(n_stages);

        for k in 0..n_stages {
            let col = self.adjusted_column(matrix, k);
            let score = self.tarantula_score(&col, &violations.violations, n_fail, n_pass);
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
        let colors: Vec<SuspiciousnessColor> = raw_scores
            .iter()
            .map(|&s| SuspiciousnessColor::from_score(s))
            .collect();

        Ok(TarantulaResult {
            scores,
            ranked_stages: ranked,
            top_suspect: top,
            separation_ratio: sep,
            colors,
        })
    }

    /// Compare the Tarantula ranking with an Ochiai ranking.
    pub fn compare_with_ochiai(
        tarantula_ranked: &[usize],
        ochiai_ranked: &[usize],
    ) -> RankCorrelation {
        let n = tarantula_ranked.len().min(ochiai_ranked.len());
        if n == 0 {
            return RankCorrelation {
                spearman_rho: 0.0,
                kendall_tau: 0.0,
                top_k_overlap: 0.0,
            };
        }

        // Build rank vectors (position in ranking)
        let mut t_rank = vec![0usize; n];
        let mut o_rank = vec![0usize; n];
        for (pos, &stage) in tarantula_ranked.iter().enumerate() {
            if stage < n {
                t_rank[stage] = pos;
            }
        }
        for (pos, &stage) in ochiai_ranked.iter().enumerate() {
            if stage < n {
                o_rank[stage] = pos;
            }
        }

        let spearman = spearman_rho(&t_rank, &o_rank);
        let kendall = kendall_tau(&t_rank, &o_rank);

        // Top-k overlap (k = ceil(n/2))
        let k = (n + 1) / 2;
        let top_t: std::collections::HashSet<usize> =
            tarantula_ranked.iter().take(k).cloned().collect();
        let top_o: std::collections::HashSet<usize> =
            ochiai_ranked.iter().take(k).cloned().collect();
        let overlap = top_t.intersection(&top_o).count() as f64 / k as f64;

        RankCorrelation {
            spearman_rho: spearman,
            kendall_tau: kendall,
            top_k_overlap: overlap,
        }
    }

    // ── Internal helpers ────────────────────────────────────────────────

    fn tarantula_score(
        &self,
        col: &[f64],
        violations: &[bool],
        n_fail: f64,
        n_pass: f64,
    ) -> f64 {
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

        let fail_ratio = if n_fail > f64::EPSILON {
            ef / n_fail
        } else {
            0.0
        };

        let pass_ratio = if n_pass > f64::EPSILON {
            ep / n_pass
        } else {
            0.0
        };

        let denom = fail_ratio + pass_ratio;
        if denom < f64::EPSILON {
            0.0
        } else {
            fail_ratio / denom
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

    fn validate(
        &self,
        matrix: &DifferentialMatrix,
        violations: &ViolationVector,
    ) -> Result<()> {
        if matrix.n_tests != violations.len() {
            return Err(LocalizerError::matrix(
                "row/violation length mismatch",
                matrix.n_tests,
                matrix.n_stages,
            ));
        }
        if matrix.n_stages == 0 {
            return Err(LocalizerError::matrix("zero stages", matrix.n_tests, 0));
        }
        Ok(())
    }

    fn handle_no_violations(&self, matrix: &DifferentialMatrix) -> Result<TarantulaResult> {
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
        let colors = vec![SuspiciousnessColor::from_score(0.0); n];
        Ok(TarantulaResult {
            scores,
            ranked_stages: (0..n).collect(),
            top_suspect: 0,
            separation_ratio: 1.0,
            colors,
        })
    }
}

// ── Rank correlation helpers ────────────────────────────────────────────────

fn spearman_rho(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let d_sq_sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai as f64 - bi as f64;
            d * d
        })
        .sum();
    1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1.0))
}

fn kendall_tau(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let a_diff = a[i] as i64 - a[j] as i64;
            let b_diff = b[i] as i64 - b[j] as i64;
            let product = a_diff * b_diff;
            if product > 0 {
                concordant += 1;
            } else if product < 0 {
                discordant += 1;
            }
        }
    }
    let pairs = (n * (n - 1)) as f64 / 2.0;
    if pairs < f64::EPSILON {
        0.0
    } else {
        (concordant - discordant) as f64 / pairs
    }
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let (r1, g1, b1) = if h_prime < 1.0 {
        (c, x, 0.0)
    } else if h_prime < 2.0 {
        (x, c, 0.0)
    } else if h_prime < 3.0 {
        (0.0, c, x)
    } else if h_prime < 4.0 {
        (0.0, x, c)
    } else if h_prime < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = v - c;
    (
        ((r1 + m) * 255.0).round() as u8,
        ((g1 + m) * 255.0).round() as u8,
        ((b1 + m) * 255.0).round() as u8,
    )
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_matrix() -> DifferentialMatrix {
        let data = vec![
            vec![0.9, 0.1, 0.2],
            vec![0.8, 0.15, 0.1],
            vec![0.85, 0.05, 0.25],
            vec![0.1, 0.4, 0.05],
            vec![0.05, 0.35, 0.0],
            vec![0.15, 0.45, 0.1],
        ];
        let names = vec!["tokenizer".into(), "pos_tagger".into(), "parser".into()];
        DifferentialMatrix::new(data, names).unwrap()
    }

    fn sample_violations() -> ViolationVector {
        ViolationVector::new(vec![true, true, true, false, false, false])
    }

    #[test]
    fn test_basic_tarantula() {
        let m = TarantulaMetric::new();
        let result = m
            .compute_suspiciousness(&sample_matrix(), &sample_violations())
            .unwrap();
        assert_eq!(result.scores.len(), 3);
        assert_eq!(result.top_suspect, 0);
    }

    #[test]
    fn test_color_mapping_red_for_suspicious() {
        let m = TarantulaMetric::new();
        let result = m
            .compute_suspiciousness(&sample_matrix(), &sample_violations())
            .unwrap();
        let top_color = result.colors[result.top_suspect];
        // Top suspect should be reddish (high r, low g)
        assert!(
            top_color.r > top_color.g,
            "expected red > green for top suspect, got r={} g={}",
            top_color.r,
            top_color.g
        );
    }

    #[test]
    fn test_color_mapping_green_for_safe() {
        let c = SuspiciousnessColor::from_score(0.0);
        // Score=0 → pure green region
        assert!(c.g > c.r, "score 0 should be green, got r={} g={}", c.r, c.g);
    }

    #[test]
    fn test_color_mapping_red_for_max() {
        let c = SuspiciousnessColor::from_score(1.0);
        assert!(c.r > c.g, "score 1 should be red, got r={} g={}", c.r, c.g);
    }

    #[test]
    fn test_no_violations() {
        let m = TarantulaMetric::new();
        let v = ViolationVector::new(vec![false; 6]);
        let result = m
            .compute_suspiciousness(&sample_matrix(), &v)
            .unwrap();
        for s in &result.scores {
            assert_eq!(s.score, 0.0);
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let m = TarantulaMetric::new();
        let matrix = sample_matrix();
        let v = ViolationVector::new(vec![true, false]);
        assert!(m.compute_suspiciousness(&matrix, &v).is_err());
    }

    #[test]
    fn test_compare_with_ochiai() {
        let t_ranked = vec![0, 2, 1];
        let o_ranked = vec![0, 1, 2];
        let corr = TarantulaMetric::compare_with_ochiai(&t_ranked, &o_ranked);
        // Same top element → some agreement
        assert!(corr.spearman_rho > 0.0);
        assert!(corr.top_k_overlap > 0.0);
    }

    #[test]
    fn test_hex_color() {
        let c = SuspiciousnessColor { r: 255, g: 0, b: 128 };
        assert_eq!(c.to_hex(), "#ff0080");
    }

    #[test]
    fn test_scores_in_unit_interval() {
        let m = TarantulaMetric::new();
        let result = m
            .compute_suspiciousness(&sample_matrix(), &sample_violations())
            .unwrap();
        for s in &result.scores {
            assert!(
                (0.0..=1.0).contains(&s.score),
                "score out of [0,1]: {}",
                s.score
            );
        }
    }

    #[test]
    fn test_spearman_identical() {
        let a = vec![0, 1, 2, 3];
        let b = vec![0, 1, 2, 3];
        let rho = spearman_rho(&a, &b);
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kendall_identical() {
        let a = vec![0, 1, 2];
        let b = vec![0, 1, 2];
        let tau = kendall_tau(&a, &b);
        assert!((tau - 1.0).abs() < 1e-10);
    }
}
