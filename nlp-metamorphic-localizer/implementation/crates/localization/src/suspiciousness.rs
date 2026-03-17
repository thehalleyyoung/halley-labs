//! Suspiciousness scoring dispatch and comparison.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageScore {
    pub stage_index: usize,
    pub stage_name: String,
    pub score: f64,
    pub rank: usize,
}

pub trait SuspiciousnessComputer {
    fn compute(&self, matrix: &[Vec<f64>], violations: &[bool], stage_names: &[String]) -> Vec<StageScore>;
    fn name(&self) -> &str;
}

pub struct OchiaiComputer;

impl SuspiciousnessComputer for OchiaiComputer {
    fn compute(&self, matrix: &[Vec<f64>], violations: &[bool], stage_names: &[String]) -> Vec<StageScore> {
        let n_tests = matrix.len();
        let n_stages = stage_names.len();
        let viol_count = violations.iter().filter(|&&v| v).count() as f64;
        if viol_count == 0.0 {
            return stage_names.iter().enumerate().map(|(i, name)| StageScore { stage_index: i, stage_name: name.clone(), score: 0.0, rank: i + 1 }).collect();
        }
        let mut scores: Vec<(usize, f64)> = (0..n_stages).map(|k| {
            let mut sum_fail = 0.0;
            let mut sum_total = 0.0;
            for i in 0..n_tests {
                let d = matrix[i].get(k).copied().unwrap_or(0.0).abs();
                sum_total += d;
                if violations[i] { sum_fail += d; }
            }
            let denom = (sum_total * viol_count).sqrt();
            (k, if denom > 0.0 { sum_fail / denom } else { 0.0 })
        }).collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.iter().enumerate().map(|(rank, &(idx, score))| StageScore {
            stage_index: idx, stage_name: stage_names.get(idx).cloned().unwrap_or_default(), score, rank: rank + 1,
        }).collect()
    }
    fn name(&self) -> &str { "Ochiai" }
}

pub struct DStarComputer { pub star_param: f64 }

impl Default for DStarComputer {
    fn default() -> Self { Self { star_param: 2.0 } }
}

impl SuspiciousnessComputer for DStarComputer {
    fn compute(&self, matrix: &[Vec<f64>], violations: &[bool], stage_names: &[String]) -> Vec<StageScore> {
        let n_tests = matrix.len();
        let n_stages = stage_names.len();
        let mut scores: Vec<(usize, f64)> = (0..n_stages).map(|k| {
            let mut sum_fail = 0.0;
            let mut sum_pass = 0.0;
            let mut sum_total = 0.0;
            for i in 0..n_tests {
                let d = matrix[i].get(k).copied().unwrap_or(0.0).abs();
                sum_total += d;
                if violations[i] { sum_fail += d; } else { sum_pass += d; }
            }
            let numerator = sum_fail.powf(self.star_param);
            let denominator = sum_pass + (sum_total - sum_fail);
            (k, if denominator > 0.0 { numerator / denominator } else { 0.0 })
        }).collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.iter().enumerate().map(|(rank, &(idx, score))| StageScore {
            stage_index: idx, stage_name: stage_names.get(idx).cloned().unwrap_or_default(), score, rank: rank + 1,
        }).collect()
    }
    fn name(&self) -> &str { "DStar" }
}

pub struct TarantulaComputer;

impl SuspiciousnessComputer for TarantulaComputer {
    fn compute(&self, matrix: &[Vec<f64>], violations: &[bool], stage_names: &[String]) -> Vec<StageScore> {
        let n_tests = matrix.len();
        let n_stages = stage_names.len();
        let n_fail = violations.iter().filter(|&&v| v).count() as f64;
        let n_pass = violations.iter().filter(|&&v| !v).count() as f64;
        if n_fail == 0.0 || n_pass == 0.0 {
            return stage_names.iter().enumerate().map(|(i, name)| StageScore { stage_index: i, stage_name: name.clone(), score: 0.5, rank: i + 1 }).collect();
        }
        let mut scores: Vec<(usize, f64)> = (0..n_stages).map(|k| {
            let mut sum_fail = 0.0;
            let mut sum_pass = 0.0;
            for i in 0..n_tests {
                let d = matrix[i].get(k).copied().unwrap_or(0.0).abs();
                if violations[i] { sum_fail += d; } else { sum_pass += d; }
            }
            let fail_ratio = sum_fail / n_fail;
            let pass_ratio = sum_pass / n_pass;
            let denom = fail_ratio + pass_ratio;
            (k, if denom > 0.0 { fail_ratio / denom } else { 0.5 })
        }).collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.iter().enumerate().map(|(rank, &(idx, score))| StageScore {
            stage_index: idx, stage_name: stage_names.get(idx).cloned().unwrap_or_default(), score, rank: rank + 1,
        }).collect()
    }
    fn name(&self) -> &str { "Tarantula" }
}

pub struct EnsembleComputer {
    pub computers: Vec<Box<dyn SuspiciousnessComputer>>,
}

impl EnsembleComputer {
    pub fn new() -> Self {
        Self { computers: vec![Box::new(OchiaiComputer), Box::new(DStarComputer::default()), Box::new(TarantulaComputer)] }
    }

    pub fn compute(&self, matrix: &[Vec<f64>], violations: &[bool], stage_names: &[String]) -> Vec<StageScore> {
        let all_rankings: Vec<Vec<StageScore>> = self.computers.iter().map(|c| c.compute(matrix, violations, stage_names)).collect();
        borda_aggregate(&all_rankings, stage_names)
    }
}

pub fn borda_aggregate(rankings: &[Vec<StageScore>], stage_names: &[String]) -> Vec<StageScore> {
    let n = stage_names.len();
    let mut borda_scores = vec![0.0; n];

    for ranking in rankings {
        for score in ranking {
            if score.stage_index < n {
                borda_scores[score.stage_index] += (n - score.rank) as f64;
            }
        }
    }

    let mut indexed: Vec<(usize, f64)> = borda_scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    indexed.iter().enumerate().map(|(rank, &(idx, score))| StageScore {
        stage_index: idx, stage_name: stage_names.get(idx).cloned().unwrap_or_default(), score, rank: rank + 1,
    }).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    pub metric_a: String,
    pub metric_b: String,
    pub kendall_tau: f64,
    pub top_k_agreement: f64,
}

impl MetricComparison {
    pub fn compute(name_a: &str, ranking_a: &[StageScore], name_b: &str, ranking_b: &[StageScore]) -> Self {
        let n = ranking_a.len().min(ranking_b.len());
        if n <= 1 {
            return Self { metric_a: name_a.into(), metric_b: name_b.into(), kendall_tau: 1.0, top_k_agreement: 1.0 };
        }

        // Kendall tau: count concordant and discordant pairs
        let rank_a: HashMap<usize, usize> = ranking_a.iter().map(|s| (s.stage_index, s.rank)).collect();
        let rank_b: HashMap<usize, usize> = ranking_b.iter().map(|s| (s.stage_index, s.rank)).collect();
        let indices: Vec<usize> = rank_a.keys().copied().collect();

        let mut concordant = 0;
        let mut discordant = 0;
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                let a_diff = rank_a.get(&indices[i]).unwrap_or(&0).cmp(rank_a.get(&indices[j]).unwrap_or(&0));
                let b_diff = rank_b.get(&indices[i]).unwrap_or(&0).cmp(rank_b.get(&indices[j]).unwrap_or(&0));
                if a_diff == b_diff { concordant += 1; } else { discordant += 1; }
            }
        }
        let total = concordant + discordant;
        let tau = if total > 0 { (concordant as f64 - discordant as f64) / total as f64 } else { 1.0 };

        // Top-1 agreement
        let top_a = ranking_a.iter().min_by_key(|s| s.rank).map(|s| s.stage_index);
        let top_b = ranking_b.iter().min_by_key(|s| s.rank).map(|s| s.stage_index);
        let agreement = if top_a == top_b { 1.0 } else { 0.0 };

        Self { metric_a: name_a.into(), metric_b: name_b.into(), kendall_tau: tau, top_k_agreement: agreement }
    }
}

pub fn select_best_metric(
    matrix: &[Vec<f64>],
    violations: &[bool],
    stage_names: &[String],
    ground_truth_fault: usize,
) -> String {
    let computers: Vec<(&str, Box<dyn SuspiciousnessComputer>)> = vec![
        ("Ochiai", Box::new(OchiaiComputer)),
        ("DStar", Box::new(DStarComputer::default())),
        ("Tarantula", Box::new(TarantulaComputer)),
    ];

    let mut best_name = "Ochiai".to_string();
    let mut best_rank = usize::MAX;

    for (name, computer) in &computers {
        let scores = computer.compute(matrix, violations, stage_names);
        if let Some(fault_score) = scores.iter().find(|s| s.stage_index == ground_truth_fault) {
            if fault_score.rank < best_rank {
                best_rank = fault_score.rank;
                best_name = name.to_string();
            }
        }
    }

    best_name
}

pub struct SuspiciousnessFactory;

impl SuspiciousnessFactory {
    pub fn create(metric_name: &str) -> Box<dyn SuspiciousnessComputer> {
        match metric_name.to_lowercase().as_str() {
            "ochiai" => Box::new(OchiaiComputer),
            "dstar" => Box::new(DStarComputer::default()),
            "tarantula" => Box::new(TarantulaComputer),
            _ => Box::new(OchiaiComputer),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data() -> (Vec<Vec<f64>>, Vec<bool>, Vec<String>) {
        let matrix = vec![
            vec![0.1, 0.8, 0.2],
            vec![0.0, 0.7, 0.1],
            vec![0.1, 0.9, 0.3],
            vec![0.0, 0.1, 0.0],
            vec![0.1, 0.0, 0.1],
        ];
        let violations = vec![true, true, true, false, false];
        let names = vec!["tok".into(), "tag".into(), "parse".into()];
        (matrix, violations, names)
    }

    #[test]
    fn test_ochiai() {
        let (m, v, n) = make_data();
        let scores = OchiaiComputer.compute(&m, &v, &n);
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].rank, 1);
    }

    #[test]
    fn test_dstar() {
        let (m, v, n) = make_data();
        let scores = DStarComputer::default().compute(&m, &v, &n);
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_tarantula() {
        let (m, v, n) = make_data();
        let scores = TarantulaComputer.compute(&m, &v, &n);
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_ensemble() {
        let (m, v, n) = make_data();
        let ensemble = EnsembleComputer::new();
        let scores = ensemble.compute(&m, &v, &n);
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].rank, 1);
    }

    #[test]
    fn test_metric_comparison() {
        let (m, v, n) = make_data();
        let ochiai = OchiaiComputer.compute(&m, &v, &n);
        let dstar = DStarComputer::default().compute(&m, &v, &n);
        let comp = MetricComparison::compute("Ochiai", &ochiai, "DStar", &dstar);
        assert!(comp.kendall_tau >= -1.0 && comp.kendall_tau <= 1.0);
    }

    #[test]
    fn test_select_best_metric() {
        let (m, v, n) = make_data();
        let best = select_best_metric(&m, &v, &n, 1);
        assert!(!best.is_empty());
    }

    #[test]
    fn test_factory() {
        let comp = SuspiciousnessFactory::create("ochiai");
        assert_eq!(comp.name(), "Ochiai");
        let comp2 = SuspiciousnessFactory::create("unknown");
        assert_eq!(comp2.name(), "Ochiai"); // Fallback
    }

    #[test]
    fn test_borda_aggregate() {
        let (m, v, n) = make_data();
        let r1 = OchiaiComputer.compute(&m, &v, &n);
        let r2 = DStarComputer::default().compute(&m, &v, &n);
        let agg = borda_aggregate(&[r1, r2], &n);
        assert_eq!(agg.len(), 3);
    }

    #[test]
    fn test_no_violations() {
        let matrix = vec![vec![0.1, 0.2, 0.3]];
        let violations = vec![false];
        let names = vec!["a".into(), "b".into(), "c".into()];
        let scores = OchiaiComputer.compute(&matrix, &violations, &names);
        assert!(scores.iter().all(|s| s.score == 0.0));
    }
}
