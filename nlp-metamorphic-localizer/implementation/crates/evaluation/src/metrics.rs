//! Metrics for evaluating fault localization effectiveness.
//!
//! Implements standard SBFL evaluation metrics: Top-k accuracy, EXAM score,
//! wasted effort, mean first rank, and more.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete accuracy metrics for a localization evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    pub top_k: Vec<TopKAccuracy>,
    pub exam_score: EXAM,
    pub wasted_effort: WastedEffort,
    pub mean_first_rank: MeanFirstRank,
    pub precision_at_k: Vec<(usize, f64)>,
    pub recall_at_k: Vec<(usize, f64)>,
    pub map_score: f64,
}

/// Top-k accuracy: fraction of faults appearing in the top k positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopKAccuracy {
    pub k: usize,
    pub accuracy: f64,
    pub count: usize,
    pub total: usize,
}

/// EXAM score: fraction of the program that must be examined to find the fault.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EXAM {
    pub score: f64,
    pub absolute_rank: usize,
    pub total_elements: usize,
}

/// Wasted effort: number of non-faulty elements examined before finding the fault.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WastedEffort {
    pub absolute: usize,
    pub relative: f64,
}

/// Mean first rank across multiple localization runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanFirstRank {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: usize,
    pub max: usize,
    pub ranks: Vec<usize>,
}

/// Ranking quality metrics for a single localization result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingMetrics {
    pub ndcg: f64,
    pub kendall_tau: f64,
    pub spearman_rho: f64,
    pub auc: f64,
}

/// Localization accuracy result combining all metrics for one evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizationAccuracy {
    pub scenario_name: String,
    pub faulty_stages: Vec<String>,
    pub predicted_ranking: Vec<String>,
    pub first_fault_rank: usize,
    pub metrics: AccuracyMetrics,
}

// ── Metric computation ──────────────────────────────────────────────────────

/// Compute Top-k accuracy for k = 1, 3, 5.
pub fn compute_top_k(
    predicted_ranking: &[String],
    actual_faulty: &[String],
    k_values: &[usize],
) -> Vec<TopKAccuracy> {
    k_values
        .iter()
        .map(|&k| {
            let top_k: Vec<&String> = predicted_ranking.iter().take(k).collect();
            let found = actual_faulty
                .iter()
                .filter(|f| top_k.contains(f))
                .count();
            TopKAccuracy {
                k,
                accuracy: if actual_faulty.is_empty() {
                    1.0
                } else {
                    found as f64 / actual_faulty.len() as f64
                },
                count: found,
                total: actual_faulty.len(),
            }
        })
        .collect()
}

/// Compute the EXAM score (fraction of program examined to find the first fault).
pub fn compute_exam(predicted_ranking: &[String], actual_faulty: &[String]) -> EXAM {
    let first_rank = predicted_ranking
        .iter()
        .enumerate()
        .find(|(_, name)| actual_faulty.contains(name))
        .map(|(i, _)| i + 1)
        .unwrap_or(predicted_ranking.len());

    EXAM {
        score: if predicted_ranking.is_empty() {
            1.0
        } else {
            first_rank as f64 / predicted_ranking.len() as f64
        },
        absolute_rank: first_rank,
        total_elements: predicted_ranking.len(),
    }
}

/// Compute wasted effort (non-faulty elements examined before first fault).
pub fn compute_wasted_effort(
    predicted_ranking: &[String],
    actual_faulty: &[String],
) -> WastedEffort {
    let first_rank = predicted_ranking
        .iter()
        .enumerate()
        .find(|(_, name)| actual_faulty.contains(name))
        .map(|(i, _)| i)
        .unwrap_or(predicted_ranking.len());

    WastedEffort {
        absolute: first_rank,
        relative: if predicted_ranking.is_empty() {
            1.0
        } else {
            first_rank as f64 / predicted_ranking.len() as f64
        },
    }
}

/// Compute Mean First Rank from multiple evaluation runs.
pub fn compute_mean_first_rank(ranks: &[usize]) -> MeanFirstRank {
    if ranks.is_empty() {
        return MeanFirstRank {
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 0,
            max: 0,
            ranks: Vec::new(),
        };
    }

    let mut sorted = ranks.to_vec();
    sorted.sort();

    let mean = sorted.iter().sum::<usize>() as f64 / sorted.len() as f64;
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) as f64 / 2.0
    } else {
        sorted[sorted.len() / 2] as f64
    };
    let variance: f64 = sorted.iter().map(|&r| (r as f64 - mean).powi(2)).sum::<f64>()
        / sorted.len() as f64;

    MeanFirstRank {
        mean,
        median,
        std_dev: variance.sqrt(),
        min: sorted[0],
        max: *sorted.last().unwrap(),
        ranks: sorted,
    }
}

/// Compute Normalized Discounted Cumulative Gain.
pub fn compute_ndcg(predicted_ranking: &[String], actual_faulty: &[String]) -> f64 {
    if actual_faulty.is_empty() {
        return 1.0;
    }

    // Compute DCG for the predicted ranking.
    let dcg: f64 = predicted_ranking
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let relevance = if actual_faulty.contains(name) {
                1.0
            } else {
                0.0
            };
            relevance / (2.0 + i as f64).log2()
        })
        .sum();

    // Compute ideal DCG (all faulty stages at the top).
    let ideal_dcg: f64 = (0..actual_faulty.len())
        .map(|i| 1.0 / (2.0 + i as f64).log2())
        .sum();

    if ideal_dcg < f64::EPSILON {
        return 0.0;
    }

    dcg / ideal_dcg
}

/// Compute Kendall's tau correlation between predicted and actual ranking.
pub fn compute_kendall_tau(ranking_a: &[String], ranking_b: &[String]) -> f64 {
    let common: Vec<String> = ranking_a
        .iter()
        .filter(|s| ranking_b.contains(s))
        .cloned()
        .collect();

    if common.len() < 2 {
        return 0.0;
    }

    let n = common.len();
    let mut concordant = 0i64;
    let mut discordant = 0i64;

    let rank_in_a: HashMap<String, usize> = ranking_a
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();
    let rank_in_b: HashMap<String, usize> = ranking_b
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();

    for i in 0..n {
        for j in (i + 1)..n {
            let a_i = rank_in_a.get(&common[i]).copied().unwrap_or(0);
            let a_j = rank_in_a.get(&common[j]).copied().unwrap_or(0);
            let b_i = rank_in_b.get(&common[i]).copied().unwrap_or(0);
            let b_j = rank_in_b.get(&common[j]).copied().unwrap_or(0);

            if (a_i < a_j && b_i < b_j) || (a_i > a_j && b_i > b_j) {
                concordant += 1;
            } else if (a_i < a_j && b_i > b_j) || (a_i > a_j && b_i < b_j) {
                discordant += 1;
            }
        }
    }

    let total_pairs = n * (n - 1) / 2;
    if total_pairs == 0 {
        return 0.0;
    }

    (concordant - discordant) as f64 / total_pairs as f64
}

/// Compute Spearman's rank correlation coefficient.
pub fn compute_spearman_rho(ranking_a: &[String], ranking_b: &[String]) -> f64 {
    let common: Vec<String> = ranking_a
        .iter()
        .filter(|s| ranking_b.contains(s))
        .cloned()
        .collect();

    if common.len() < 2 {
        return 0.0;
    }

    let n = common.len();
    let rank_in_a: HashMap<String, usize> = ranking_a
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();
    let rank_in_b: HashMap<String, usize> = ranking_b
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();

    let d_squared_sum: f64 = common
        .iter()
        .map(|item| {
            let ra = rank_in_a.get(item).copied().unwrap_or(0) as f64;
            let rb = rank_in_b.get(item).copied().unwrap_or(0) as f64;
            (ra - rb).powi(2)
        })
        .sum();

    1.0 - (6.0 * d_squared_sum) / (n as f64 * (n as f64 * n as f64 - 1.0))
}

/// Compute precision@k.
pub fn compute_precision_at_k(
    predicted: &[String],
    actual_faulty: &[String],
    k: usize,
) -> f64 {
    let top_k: Vec<&String> = predicted.iter().take(k).collect();
    let relevant = top_k
        .iter()
        .filter(|s| actual_faulty.contains(s))
        .count();
    if k == 0 {
        return 0.0;
    }
    relevant as f64 / k as f64
}

/// Compute recall@k.
pub fn compute_recall_at_k(
    predicted: &[String],
    actual_faulty: &[String],
    k: usize,
) -> f64 {
    if actual_faulty.is_empty() {
        return 1.0;
    }
    let top_k: Vec<&String> = predicted.iter().take(k).collect();
    let relevant = top_k
        .iter()
        .filter(|s| actual_faulty.contains(s))
        .count();
    relevant as f64 / actual_faulty.len() as f64
}

/// Compute Mean Average Precision.
pub fn compute_map(predicted: &[String], actual_faulty: &[String]) -> f64 {
    if actual_faulty.is_empty() {
        return 1.0;
    }

    let mut sum_precision = 0.0;
    let mut relevant_found = 0;

    for (i, name) in predicted.iter().enumerate() {
        if actual_faulty.contains(name) {
            relevant_found += 1;
            sum_precision += relevant_found as f64 / (i + 1) as f64;
        }
    }

    if relevant_found == 0 {
        return 0.0;
    }

    sum_precision / actual_faulty.len() as f64
}

/// Compute complete accuracy metrics for a single evaluation scenario.
pub fn compute_full_metrics(
    predicted: &[String],
    actual_faulty: &[String],
) -> AccuracyMetrics {
    let k_values = vec![1, 3, 5];
    let top_k = compute_top_k(predicted, actual_faulty, &k_values);
    let exam = compute_exam(predicted, actual_faulty);
    let wasted = compute_wasted_effort(predicted, actual_faulty);

    let precision_at_k: Vec<(usize, f64)> = k_values
        .iter()
        .map(|&k| (k, compute_precision_at_k(predicted, actual_faulty, k)))
        .collect();
    let recall_at_k: Vec<(usize, f64)> = k_values
        .iter()
        .map(|&k| (k, compute_recall_at_k(predicted, actual_faulty, k)))
        .collect();
    let map_score = compute_map(predicted, actual_faulty);

    AccuracyMetrics {
        top_k,
        exam_score: exam,
        wasted_effort: wasted,
        mean_first_rank: MeanFirstRank {
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 0,
            max: 0,
            ranks: Vec::new(),
        },
        precision_at_k,
        recall_at_k,
        map_score,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_accuracy() {
        let predicted = vec!["tagger".into(), "parser".into(), "ner".into(), "tokenizer".into()];
        let actual = vec!["tagger".to_string()];

        let top_k = compute_top_k(&predicted, &actual, &[1, 3, 5]);
        assert!((top_k[0].accuracy - 1.0).abs() < f64::EPSILON); // top-1 = 100%
        assert!((top_k[1].accuracy - 1.0).abs() < f64::EPSILON); // top-3 = 100%
    }

    #[test]
    fn test_top_k_miss() {
        let predicted = vec!["parser".into(), "ner".into(), "tagger".into(), "tokenizer".into()];
        let actual = vec!["tagger".to_string()];

        let top_k = compute_top_k(&predicted, &actual, &[1, 3]);
        assert!((top_k[0].accuracy).abs() < f64::EPSILON); // top-1 = 0%
        assert!((top_k[1].accuracy - 1.0).abs() < f64::EPSILON); // top-3 = 100%
    }

    #[test]
    fn test_exam_score() {
        let predicted = vec!["parser".into(), "tagger".into(), "ner".into()];
        let actual = vec!["tagger".to_string()];

        let exam = compute_exam(&predicted, &actual);
        assert!((exam.score - 2.0 / 3.0).abs() < 0.01);
        assert_eq!(exam.absolute_rank, 2);
    }

    #[test]
    fn test_wasted_effort() {
        let predicted = vec!["parser".into(), "tagger".into(), "ner".into()];
        let actual = vec!["tagger".to_string()];

        let wasted = compute_wasted_effort(&predicted, &actual);
        assert_eq!(wasted.absolute, 1); // examined 'parser' before finding 'tagger'
    }

    #[test]
    fn test_mean_first_rank() {
        let ranks = vec![1, 3, 2, 1, 5];
        let mfr = compute_mean_first_rank(&ranks);
        assert!((mfr.mean - 2.4).abs() < 0.01);
        assert!((mfr.median - 2.0).abs() < 0.01);
        assert_eq!(mfr.min, 1);
        assert_eq!(mfr.max, 5);
    }

    #[test]
    fn test_ndcg_perfect() {
        let predicted = vec!["a".into(), "b".into(), "c".into()];
        let actual = vec!["a".to_string()];
        let ndcg = compute_ndcg(&predicted, &actual);
        assert!((ndcg - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ndcg_imperfect() {
        let predicted = vec!["c".into(), "a".into(), "b".into()];
        let actual = vec!["a".to_string()];
        let ndcg = compute_ndcg(&predicted, &actual);
        assert!(ndcg < 1.0);
        assert!(ndcg > 0.0);
    }

    #[test]
    fn test_kendall_tau_perfect() {
        let a = vec!["x".into(), "y".into(), "z".into()];
        let b = vec!["x".into(), "y".into(), "z".into()];
        let tau = compute_kendall_tau(&a, &b);
        assert!((tau - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_kendall_tau_reversed() {
        let a = vec!["x".into(), "y".into(), "z".into()];
        let b = vec!["z".into(), "y".into(), "x".into()];
        let tau = compute_kendall_tau(&a, &b);
        assert!((tau - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_spearman_rho_perfect() {
        let a = vec!["x".into(), "y".into(), "z".into()];
        let b = vec!["x".into(), "y".into(), "z".into()];
        let rho = compute_spearman_rho(&a, &b);
        assert!((rho - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_precision_recall_at_k() {
        let predicted = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let actual = vec!["a".to_string(), "c".to_string()];

        let p1 = compute_precision_at_k(&predicted, &actual, 1);
        assert!((p1 - 1.0).abs() < f64::EPSILON); // a is relevant

        let p2 = compute_precision_at_k(&predicted, &actual, 2);
        assert!((p2 - 0.5).abs() < f64::EPSILON); // only a in top-2

        let r2 = compute_recall_at_k(&predicted, &actual, 2);
        assert!((r2 - 0.5).abs() < f64::EPSILON); // found 1 of 2

        let r3 = compute_recall_at_k(&predicted, &actual, 3);
        assert!((r3 - 1.0).abs() < f64::EPSILON); // found both
    }

    #[test]
    fn test_map() {
        let predicted = vec!["a".into(), "b".into(), "c".into()];
        let actual = vec!["a".to_string(), "c".to_string()];
        let map = compute_map(&predicted, &actual);
        // AP = (1/1 + 2/3) / 2 = 0.833...
        assert!((map - 0.833).abs() < 0.01);
    }

    #[test]
    fn test_full_metrics() {
        let predicted = vec!["tagger".into(), "parser".into(), "ner".into()];
        let actual = vec!["tagger".to_string()];
        let metrics = compute_full_metrics(&predicted, &actual);

        assert!((metrics.top_k[0].accuracy - 1.0).abs() < f64::EPSILON);
        assert_eq!(metrics.exam_score.absolute_rank, 1);
        assert_eq!(metrics.wasted_effort.absolute, 0);
    }
}
