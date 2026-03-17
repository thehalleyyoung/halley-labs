//! Evaluation harness: end-to-end orchestration of evaluation runs.
//!
//! Combines ground truth, fault injection, metric computation, and reporting.

use crate::benchmarks::{BenchmarkConfig, BenchmarkResult, BenchmarkSuite};
use crate::fault_injection::{FaultInjector, FaultProfile};
use crate::ground_truth::{GroundTruth, GroundTruthEntry};
use crate::metrics::{self, AccuracyMetrics, LocalizationAccuracy};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for an evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub name: String,
    pub ground_truth_name: String,
    pub fault_profiles: Vec<String>,
    pub metrics_to_compute: Vec<String>,
    pub k_values: Vec<usize>,
    pub output_format: OutputFormat,
    pub include_per_scenario: bool,
    pub include_aggregate: bool,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            name: "default_evaluation".to_string(),
            ground_truth_name: String::new(),
            fault_profiles: Vec::new(),
            metrics_to_compute: vec![
                "top_k".to_string(),
                "exam".to_string(),
                "wasted_effort".to_string(),
                "ndcg".to_string(),
                "map".to_string(),
            ],
            k_values: vec![1, 3, 5],
            output_format: OutputFormat::Json,
            include_per_scenario: true,
            include_aggregate: true,
        }
    }
}

/// Output format for evaluation reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Csv,
    Markdown,
    Html,
}

/// Complete evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub config: EvaluationConfig,
    pub per_scenario: Vec<ScenarioEvaluation>,
    pub aggregate: AggregateMetrics,
    pub timing: TimingReport,
    pub metadata: HashMap<String, String>,
}

/// Evaluation result for a single scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioEvaluation {
    pub scenario_id: String,
    pub pipeline_name: String,
    pub faulty_stages: Vec<String>,
    pub predicted_ranking: Vec<String>,
    pub accuracy: AccuracyMetrics,
    pub first_fault_rank: usize,
    pub localization_time: Duration,
}

/// Aggregate metrics across all scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateMetrics {
    pub scenario_count: usize,
    pub mean_top1_accuracy: f64,
    pub mean_top3_accuracy: f64,
    pub mean_top5_accuracy: f64,
    pub mean_exam_score: f64,
    pub mean_wasted_effort: f64,
    pub mean_first_rank: f64,
    pub median_first_rank: f64,
    pub mean_map: f64,
    pub mean_ndcg: f64,
    pub perfect_localization_rate: f64,
}

/// Timing report for the evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingReport {
    pub total_time: Duration,
    pub mean_per_scenario: Duration,
    pub mean_per_test_case: Duration,
    pub total_test_cases: usize,
}

/// The main evaluation harness.
pub struct EvaluationHarness {
    config: EvaluationConfig,
    ground_truth: GroundTruth,
    scenario_results: Vec<ScenarioEvaluation>,
}

impl EvaluationHarness {
    pub fn new(config: EvaluationConfig, ground_truth: GroundTruth) -> Self {
        Self {
            config,
            ground_truth,
            scenario_results: Vec::new(),
        }
    }

    /// Record the result of evaluating one scenario.
    pub fn record_scenario(
        &mut self,
        scenario_id: &str,
        predicted_ranking: Vec<String>,
        localization_time: Duration,
    ) {
        let entry = match self.ground_truth.get_entry(scenario_id) {
            Some(e) => e,
            None => return,
        };

        let accuracy = metrics::compute_full_metrics(&predicted_ranking, &entry.faulty_stages);
        let first_rank = predicted_ranking
            .iter()
            .enumerate()
            .find(|(_, name)| entry.faulty_stages.contains(name))
            .map(|(i, _)| i + 1)
            .unwrap_or(predicted_ranking.len() + 1);

        self.scenario_results.push(ScenarioEvaluation {
            scenario_id: scenario_id.to_string(),
            pipeline_name: entry.pipeline_name.clone(),
            faulty_stages: entry.faulty_stages.clone(),
            predicted_ranking,
            accuracy,
            first_fault_rank: first_rank,
            localization_time,
        });
    }

    /// Compute aggregate metrics across all recorded scenarios.
    pub fn compute_aggregate(&self) -> AggregateMetrics {
        let n = self.scenario_results.len();
        if n == 0 {
            return AggregateMetrics {
                scenario_count: 0,
                mean_top1_accuracy: 0.0,
                mean_top3_accuracy: 0.0,
                mean_top5_accuracy: 0.0,
                mean_exam_score: 0.0,
                mean_wasted_effort: 0.0,
                mean_first_rank: 0.0,
                median_first_rank: 0.0,
                mean_map: 0.0,
                mean_ndcg: 0.0,
                perfect_localization_rate: 0.0,
            };
        }

        let mean_top1 = self
            .scenario_results
            .iter()
            .filter_map(|s| s.accuracy.top_k.first())
            .map(|t| t.accuracy)
            .sum::<f64>()
            / n as f64;

        let mean_top3 = self
            .scenario_results
            .iter()
            .filter_map(|s| s.accuracy.top_k.get(1))
            .map(|t| t.accuracy)
            .sum::<f64>()
            / n as f64;

        let mean_top5 = self
            .scenario_results
            .iter()
            .filter_map(|s| s.accuracy.top_k.get(2))
            .map(|t| t.accuracy)
            .sum::<f64>()
            / n as f64;

        let mean_exam = self
            .scenario_results
            .iter()
            .map(|s| s.accuracy.exam_score.score)
            .sum::<f64>()
            / n as f64;

        let mean_wasted = self
            .scenario_results
            .iter()
            .map(|s| s.accuracy.wasted_effort.relative)
            .sum::<f64>()
            / n as f64;

        let mut ranks: Vec<usize> = self
            .scenario_results
            .iter()
            .map(|s| s.first_fault_rank)
            .collect();
        ranks.sort();

        let mean_rank = ranks.iter().sum::<usize>() as f64 / n as f64;
        let median_rank = if n % 2 == 0 {
            (ranks[n / 2 - 1] + ranks[n / 2]) as f64 / 2.0
        } else {
            ranks[n / 2] as f64
        };

        let mean_map = self
            .scenario_results
            .iter()
            .map(|s| s.accuracy.map_score)
            .sum::<f64>()
            / n as f64;

        let perfect_count = self
            .scenario_results
            .iter()
            .filter(|s| s.first_fault_rank == 1)
            .count();

        AggregateMetrics {
            scenario_count: n,
            mean_top1_accuracy: mean_top1,
            mean_top3_accuracy: mean_top3,
            mean_top5_accuracy: mean_top5,
            mean_exam_score: mean_exam,
            mean_wasted_effort: mean_wasted,
            mean_first_rank: mean_rank,
            median_first_rank: median_rank,
            mean_map,
            mean_ndcg: 0.0,
            perfect_localization_rate: perfect_count as f64 / n as f64,
        }
    }

    /// Generate the full evaluation report.
    pub fn generate_report(&self, total_time: Duration, total_tests: usize) -> EvaluationReport {
        let aggregate = self.compute_aggregate();
        let n = self.scenario_results.len();

        let timing = TimingReport {
            total_time,
            mean_per_scenario: if n > 0 {
                total_time / n as u32
            } else {
                Duration::ZERO
            },
            mean_per_test_case: if total_tests > 0 {
                total_time / total_tests as u32
            } else {
                Duration::ZERO
            },
            total_test_cases: total_tests,
        };

        EvaluationReport {
            config: self.config.clone(),
            per_scenario: self.scenario_results.clone(),
            aggregate,
            timing,
            metadata: HashMap::new(),
        }
    }

    /// Format the report as JSON.
    pub fn report_json(&self, total_time: Duration, total_tests: usize) -> String {
        let report = self.generate_report(total_time, total_tests);
        serde_json::to_string_pretty(&report).unwrap_or_else(|e| format!("Error: {}", e))
    }

    /// Format the report as Markdown.
    pub fn report_markdown(&self, total_time: Duration, total_tests: usize) -> String {
        let report = self.generate_report(total_time, total_tests);
        let agg = &report.aggregate;

        let mut md = String::new();
        md.push_str(&format!("# Evaluation Report: {}\n\n", self.config.name));
        md.push_str("## Aggregate Metrics\n\n");
        md.push_str(&format!("| Metric | Value |\n"));
        md.push_str(&format!("|--------|-------|\n"));
        md.push_str(&format!("| Scenarios | {} |\n", agg.scenario_count));
        md.push_str(&format!(
            "| Top-1 Accuracy | {:.1}% |\n",
            agg.mean_top1_accuracy * 100.0
        ));
        md.push_str(&format!(
            "| Top-3 Accuracy | {:.1}% |\n",
            agg.mean_top3_accuracy * 100.0
        ));
        md.push_str(&format!(
            "| Top-5 Accuracy | {:.1}% |\n",
            agg.mean_top5_accuracy * 100.0
        ));
        md.push_str(&format!(
            "| Mean EXAM Score | {:.3} |\n",
            agg.mean_exam_score
        ));
        md.push_str(&format!("| Mean First Rank | {:.1} |\n", agg.mean_first_rank));
        md.push_str(&format!(
            "| Median First Rank | {:.1} |\n",
            agg.median_first_rank
        ));
        md.push_str(&format!("| Mean MAP | {:.3} |\n", agg.mean_map));
        md.push_str(&format!(
            "| Perfect Localization Rate | {:.1}% |\n",
            agg.perfect_localization_rate * 100.0
        ));

        if self.config.include_per_scenario {
            md.push_str("\n## Per-Scenario Results\n\n");
            md.push_str(
                "| Scenario | Pipeline | Faulty | Predicted #1 | Rank | EXAM | MAP |\n",
            );
            md.push_str(
                "|----------|----------|--------|--------------|------|------|-----|\n",
            );
            for s in &report.per_scenario {
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {:.3} | {:.3} |\n",
                    s.scenario_id,
                    s.pipeline_name,
                    s.faulty_stages.join(", "),
                    s.predicted_ranking.first().unwrap_or(&"—".to_string()),
                    s.first_fault_rank,
                    s.accuracy.exam_score.score,
                    s.accuracy.map_score,
                ));
            }
        }

        md.push_str(&format!(
            "\n---\n*Total time: {:.2}s, {:.1}ms/scenario, {} test cases*\n",
            report.timing.total_time.as_secs_f64(),
            report.timing.mean_per_scenario.as_secs_f64() * 1000.0,
            report.timing.total_test_cases,
        ));

        md
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ground_truth::GroundTruthBuilder;

    fn make_ground_truth() -> GroundTruth {
        GroundTruthBuilder::new("test_gt")
            .single_fault("spacy", "tagger", "tag_flip", 0.6)
            .single_fault("spacy", "parser", "dep_swap", 0.5)
            .single_fault("spacy", "ner", "span_shift", 0.7)
            .build()
    }

    #[test]
    fn test_harness_perfect_localization() {
        let gt = make_ground_truth();
        let config = EvaluationConfig::default();
        let mut harness = EvaluationHarness::new(config, gt);

        // Perfect prediction: faulty stage ranked first.
        harness.record_scenario(
            "scenario_001",
            vec!["tagger".into(), "parser".into(), "ner".into(), "tokenizer".into()],
            Duration::from_millis(100),
        );
        harness.record_scenario(
            "scenario_002",
            vec!["parser".into(), "tagger".into(), "ner".into(), "tokenizer".into()],
            Duration::from_millis(100),
        );
        harness.record_scenario(
            "scenario_003",
            vec!["ner".into(), "tagger".into(), "parser".into(), "tokenizer".into()],
            Duration::from_millis(100),
        );

        let agg = harness.compute_aggregate();
        assert_eq!(agg.scenario_count, 3);
        assert!((agg.mean_top1_accuracy - 1.0).abs() < f64::EPSILON);
        assert!((agg.perfect_localization_rate - 1.0).abs() < f64::EPSILON);
        assert!((agg.mean_first_rank - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_harness_imperfect_localization() {
        let gt = make_ground_truth();
        let config = EvaluationConfig::default();
        let mut harness = EvaluationHarness::new(config, gt);

        // Tagger is faulty but ranked 3rd.
        harness.record_scenario(
            "scenario_001",
            vec!["parser".into(), "ner".into(), "tagger".into(), "tokenizer".into()],
            Duration::from_millis(100),
        );

        let agg = harness.compute_aggregate();
        assert_eq!(agg.scenario_count, 1);
        assert!((agg.mean_top1_accuracy).abs() < f64::EPSILON); // not in top-1
        assert!((agg.perfect_localization_rate).abs() < f64::EPSILON);
        assert!((agg.mean_first_rank - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_report_generation() {
        let gt = make_ground_truth();
        let config = EvaluationConfig::default();
        let mut harness = EvaluationHarness::new(config, gt);

        harness.record_scenario(
            "scenario_001",
            vec!["tagger".into(), "parser".into()],
            Duration::from_millis(50),
        );

        let report = harness.generate_report(Duration::from_secs(1), 100);
        assert_eq!(report.per_scenario.len(), 1);
        assert_eq!(report.timing.total_test_cases, 100);
    }

    #[test]
    fn test_markdown_report() {
        let gt = make_ground_truth();
        let config = EvaluationConfig {
            name: "markdown_test".to_string(),
            include_per_scenario: true,
            ..Default::default()
        };
        let mut harness = EvaluationHarness::new(config, gt);

        harness.record_scenario(
            "scenario_001",
            vec!["tagger".into(), "parser".into()],
            Duration::from_millis(50),
        );

        let md = harness.report_markdown(Duration::from_secs(1), 50);
        assert!(md.contains("markdown_test"));
        assert!(md.contains("Top-1 Accuracy"));
        assert!(md.contains("scenario_001"));
    }
}
