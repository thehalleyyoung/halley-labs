// Ground truth labeling: compare decomposition results at different time cutoffs,
// assign labels, analyze label stability, and compute consensus labels.

use crate::classifier::traits::DecompositionMethod;
use crate::error::{OracleError, OracleResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Time cutoff for evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TimeCutoff {
    Short,   // 60s
    Medium,  // 300s
    Long,    // 900s
    Full,    // 3600s
}

impl TimeCutoff {
    pub fn all() -> &'static [TimeCutoff] {
        &[
            TimeCutoff::Short,
            TimeCutoff::Medium,
            TimeCutoff::Long,
            TimeCutoff::Full,
        ]
    }

    pub fn seconds(&self) -> u64 {
        match self {
            TimeCutoff::Short => 60,
            TimeCutoff::Medium => 300,
            TimeCutoff::Long => 900,
            TimeCutoff::Full => 3600,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            TimeCutoff::Short => "60s",
            TimeCutoff::Medium => "300s",
            TimeCutoff::Long => "900s",
            TimeCutoff::Full => "3600s",
        }
    }
}

impl std::fmt::Display for TimeCutoff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Result of running a decomposition method on an instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionResult {
    pub method: DecompositionMethod,
    pub dual_bound: f64,
    pub primal_bound: Option<f64>,
    pub gap: Option<f64>,
    pub elapsed_secs: f64,
    pub status: SolveStatus,
}

/// Solve status for a decomposition run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolveStatus {
    Optimal,
    Feasible,
    Infeasible,
    Timeout,
    Error,
}

/// Labeling result for a single instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelingResult {
    pub instance_name: String,
    pub labels_by_cutoff: HashMap<String, DecompositionMethod>,
    pub consensus_label: DecompositionMethod,
    pub flip_rate: f64,
    pub is_stable: bool,
    pub best_dual_bounds: HashMap<String, HashMap<String, f64>>,
    pub improvement_over_none: HashMap<String, f64>,
}

/// Consensus label with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusLabel {
    pub instance_name: String,
    pub label: DecompositionMethod,
    pub confidence: f64,
    pub flip_rate: f64,
    pub method_at_cutoffs: Vec<(TimeCutoff, DecompositionMethod)>,
}

/// Ground truth labeler for decomposition method selection.
#[derive(Debug, Clone)]
pub struct GroundTruthLabeler {
    pub improvement_threshold: f64, // min dual bound improvement to prefer decomposition
    pub flip_rate_threshold: f64,   // max flip rate for stable labels
    pub cutoffs: Vec<TimeCutoff>,
}

impl GroundTruthLabeler {
    pub fn new() -> Self {
        Self {
            improvement_threshold: 0.01, // 1% improvement
            flip_rate_threshold: 0.2,    // 20% flip rate
            cutoffs: TimeCutoff::all().to_vec(),
        }
    }

    pub fn with_improvement_threshold(mut self, threshold: f64) -> Self {
        self.improvement_threshold = threshold;
        self
    }

    pub fn with_flip_rate_threshold(mut self, threshold: f64) -> Self {
        self.flip_rate_threshold = threshold;
        self
    }

    /// Assign a label for an instance at a single time cutoff.
    pub fn label_at_cutoff(
        &self,
        results: &HashMap<DecompositionMethod, DecompositionResult>,
    ) -> DecompositionMethod {
        let none_result = results.get(&DecompositionMethod::None);
        let none_bound = none_result.map(|r| r.dual_bound).unwrap_or(f64::NEG_INFINITY);

        let mut best_method = DecompositionMethod::None;
        let mut best_improvement = 0.0_f64;

        for (&method, result) in results {
            if method == DecompositionMethod::None {
                continue;
            }
            if matches!(result.status, SolveStatus::Infeasible | SolveStatus::Error) {
                continue;
            }

            let improvement = self.compute_improvement(result.dual_bound, none_bound);
            if improvement > self.improvement_threshold && improvement > best_improvement {
                best_improvement = improvement;
                best_method = method;
            }
        }

        best_method
    }

    /// Compute relative improvement of new bound over baseline.
    fn compute_improvement(&self, new_bound: f64, baseline: f64) -> f64 {
        if baseline.abs() < 1e-10 {
            if (new_bound - baseline).abs() < 1e-10 {
                return 0.0;
            }
            return (new_bound - baseline).abs();
        }
        (new_bound - baseline).abs() / baseline.abs()
    }

    /// Assign labels for an instance across all cutoffs and compute consensus.
    pub fn label_instance(
        &self,
        instance_name: &str,
        results_by_cutoff: &HashMap<TimeCutoff, HashMap<DecompositionMethod, DecompositionResult>>,
    ) -> OracleResult<LabelingResult> {
        if results_by_cutoff.is_empty() {
            return Err(OracleError::invalid_input("no results provided"));
        }

        let mut labels_by_cutoff = HashMap::new();
        let mut dual_bounds_by_cutoff = HashMap::new();
        let mut improvement_by_cutoff = HashMap::new();

        for &cutoff in &self.cutoffs {
            if let Some(results) = results_by_cutoff.get(&cutoff) {
                let label = self.label_at_cutoff(results);
                labels_by_cutoff.insert(cutoff.name().to_string(), label);

                // Record dual bounds
                let mut bounds = HashMap::new();
                for (method, result) in results {
                    bounds.insert(method.to_string(), result.dual_bound);
                }
                dual_bounds_by_cutoff.insert(cutoff.name().to_string(), bounds);

                // Record improvement over None
                let none_bound = results
                    .get(&DecompositionMethod::None)
                    .map(|r| r.dual_bound)
                    .unwrap_or(f64::NEG_INFINITY);
                if let Some(best_result) = results.get(&label) {
                    let imp = self.compute_improvement(best_result.dual_bound, none_bound);
                    improvement_by_cutoff.insert(cutoff.name().to_string(), imp);
                }
            }
        }

        // Compute flip rate
        let labels: Vec<DecompositionMethod> = labels_by_cutoff.values().copied().collect();
        let flip_rate = self.compute_flip_rate(&labels);
        let is_stable = flip_rate <= self.flip_rate_threshold;

        // Compute consensus label
        let consensus_label = if is_stable {
            // Use the label from the longest cutoff
            labels_by_cutoff
                .get(self.cutoffs.last().unwrap().name())
                .copied()
                .unwrap_or(DecompositionMethod::None)
        } else {
            // Majority vote
            self.majority_vote(&labels)
        };

        Ok(LabelingResult {
            instance_name: instance_name.to_string(),
            labels_by_cutoff,
            consensus_label,
            flip_rate,
            is_stable,
            best_dual_bounds: dual_bounds_by_cutoff,
            improvement_over_none: improvement_by_cutoff,
        })
    }

    /// Compute flip rate: fraction of adjacent cutoff pairs where the label changes.
    fn compute_flip_rate(&self, labels: &[DecompositionMethod]) -> f64 {
        if labels.len() <= 1 {
            return 0.0;
        }
        let flips = labels.windows(2).filter(|w| w[0] != w[1]).count();
        flips as f64 / (labels.len() - 1) as f64
    }

    /// Majority vote across labels.
    fn majority_vote(&self, labels: &[DecompositionMethod]) -> DecompositionMethod {
        let mut counts = HashMap::new();
        for &label in labels {
            *counts.entry(label).or_insert(0usize) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(method, _)| method)
            .unwrap_or(DecompositionMethod::None)
    }

    /// Analyze label stability across a corpus of instances.
    pub fn stability_analysis(
        &self,
        labeling_results: &[LabelingResult],
    ) -> StabilityReport {
        let total = labeling_results.len();
        let stable = labeling_results.iter().filter(|r| r.is_stable).count();
        let unstable = total - stable;

        let flip_rates: Vec<f64> = labeling_results.iter().map(|r| r.flip_rate).collect();
        let mean_flip_rate = if total > 0 {
            flip_rates.iter().sum::<f64>() / total as f64
        } else {
            0.0
        };

        let mut method_stability = HashMap::new();
        for result in labeling_results {
            let entry = method_stability
                .entry(result.consensus_label)
                .or_insert_with(|| MethodStability {
                    method: result.consensus_label,
                    count: 0,
                    stable_count: 0,
                    avg_flip_rate: 0.0,
                });
            entry.count += 1;
            if result.is_stable {
                entry.stable_count += 1;
            }
            entry.avg_flip_rate += result.flip_rate;
        }

        for entry in method_stability.values_mut() {
            if entry.count > 0 {
                entry.avg_flip_rate /= entry.count as f64;
            }
        }

        StabilityReport {
            total_instances: total,
            stable_instances: stable,
            unstable_instances: unstable,
            mean_flip_rate,
            method_stability,
        }
    }

    /// Generate consensus labels for a batch of instances.
    pub fn generate_consensus_labels(
        &self,
        labeling_results: &[LabelingResult],
    ) -> Vec<ConsensusLabel> {
        labeling_results
            .iter()
            .map(|r| {
                let confidence = if r.is_stable { 1.0 } else { 0.6 };
                let method_at_cutoffs: Vec<(TimeCutoff, DecompositionMethod)> = self
                    .cutoffs
                    .iter()
                    .filter_map(|&c| {
                        r.labels_by_cutoff
                            .get(c.name())
                            .map(|&label| (c, label))
                    })
                    .collect();

                ConsensusLabel {
                    instance_name: r.instance_name.clone(),
                    label: r.consensus_label,
                    confidence,
                    flip_rate: r.flip_rate,
                    method_at_cutoffs,
                }
            })
            .collect()
    }
}

/// Per-method stability statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodStability {
    pub method: DecompositionMethod,
    pub count: usize,
    pub stable_count: usize,
    pub avg_flip_rate: f64,
}

/// Stability report for a corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityReport {
    pub total_instances: usize,
    pub stable_instances: usize,
    pub unstable_instances: usize,
    pub mean_flip_rate: f64,
    pub method_stability: HashMap<DecompositionMethod, MethodStability>,
}

impl StabilityReport {
    pub fn stability_fraction(&self) -> f64 {
        if self.total_instances == 0 {
            return 0.0;
        }
        self.stable_instances as f64 / self.total_instances as f64
    }

    pub fn summary(&self) -> String {
        format!(
            "Stability: {}/{} stable ({:.1}%) | mean flip rate: {:.3}",
            self.stable_instances,
            self.total_instances,
            self.stability_fraction() * 100.0,
            self.mean_flip_rate
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_results(
        benders_bound: f64,
        dw_bound: f64,
        none_bound: f64,
    ) -> HashMap<DecompositionMethod, DecompositionResult> {
        let mut results = HashMap::new();
        results.insert(
            DecompositionMethod::Benders,
            DecompositionResult {
                method: DecompositionMethod::Benders,
                dual_bound: benders_bound,
                primal_bound: None,
                gap: None,
                elapsed_secs: 30.0,
                status: SolveStatus::Feasible,
            },
        );
        results.insert(
            DecompositionMethod::DantzigWolfe,
            DecompositionResult {
                method: DecompositionMethod::DantzigWolfe,
                dual_bound: dw_bound,
                primal_bound: None,
                gap: None,
                elapsed_secs: 45.0,
                status: SolveStatus::Feasible,
            },
        );
        results.insert(
            DecompositionMethod::None,
            DecompositionResult {
                method: DecompositionMethod::None,
                dual_bound: none_bound,
                primal_bound: None,
                gap: None,
                elapsed_secs: 60.0,
                status: SolveStatus::Feasible,
            },
        );
        results
    }

    #[test]
    fn test_time_cutoff_seconds() {
        assert_eq!(TimeCutoff::Short.seconds(), 60);
        assert_eq!(TimeCutoff::Medium.seconds(), 300);
        assert_eq!(TimeCutoff::Long.seconds(), 900);
        assert_eq!(TimeCutoff::Full.seconds(), 3600);
    }

    #[test]
    fn test_time_cutoff_display() {
        assert_eq!(TimeCutoff::Short.to_string(), "60s");
    }

    #[test]
    fn test_label_at_cutoff_benders_best() {
        let results = make_results(110.0, 105.0, 100.0);
        let labeler = GroundTruthLabeler::new();
        let label = labeler.label_at_cutoff(&results);
        assert_eq!(label, DecompositionMethod::Benders);
    }

    #[test]
    fn test_label_at_cutoff_no_improvement() {
        let results = make_results(100.0, 100.0, 100.0);
        let labeler = GroundTruthLabeler::new();
        let label = labeler.label_at_cutoff(&results);
        assert_eq!(label, DecompositionMethod::None);
    }

    #[test]
    fn test_label_at_cutoff_dw_best() {
        let results = make_results(105.0, 120.0, 100.0);
        let labeler = GroundTruthLabeler::new();
        let label = labeler.label_at_cutoff(&results);
        assert_eq!(label, DecompositionMethod::DantzigWolfe);
    }

    #[test]
    fn test_label_instance() {
        let labeler = GroundTruthLabeler::new();
        let mut results_by_cutoff = HashMap::new();
        results_by_cutoff.insert(TimeCutoff::Short, make_results(110.0, 105.0, 100.0));
        results_by_cutoff.insert(TimeCutoff::Medium, make_results(115.0, 108.0, 100.0));
        results_by_cutoff.insert(TimeCutoff::Long, make_results(120.0, 110.0, 100.0));
        results_by_cutoff.insert(TimeCutoff::Full, make_results(125.0, 112.0, 100.0));

        let result = labeler.label_instance("test_instance", &results_by_cutoff).unwrap();
        assert_eq!(result.consensus_label, DecompositionMethod::Benders);
        assert!(result.is_stable);
        assert_eq!(result.flip_rate, 0.0);
    }

    #[test]
    fn test_label_instance_unstable() {
        let labeler = GroundTruthLabeler::new();
        let mut results_by_cutoff = HashMap::new();
        results_by_cutoff.insert(TimeCutoff::Short, make_results(110.0, 105.0, 100.0));
        results_by_cutoff.insert(TimeCutoff::Medium, make_results(105.0, 115.0, 100.0));
        results_by_cutoff.insert(TimeCutoff::Long, make_results(120.0, 110.0, 100.0));
        results_by_cutoff.insert(TimeCutoff::Full, make_results(105.0, 125.0, 100.0));

        let result = labeler.label_instance("unstable_instance", &results_by_cutoff).unwrap();
        assert!(result.flip_rate > 0.0);
    }

    #[test]
    fn test_flip_rate_no_flips() {
        let labeler = GroundTruthLabeler::new();
        let labels = vec![DecompositionMethod::Benders; 4];
        assert_eq!(labeler.compute_flip_rate(&labels), 0.0);
    }

    #[test]
    fn test_flip_rate_all_flips() {
        let labeler = GroundTruthLabeler::new();
        let labels = vec![
            DecompositionMethod::Benders,
            DecompositionMethod::DantzigWolfe,
            DecompositionMethod::Benders,
            DecompositionMethod::DantzigWolfe,
        ];
        assert_eq!(labeler.compute_flip_rate(&labels), 1.0);
    }

    #[test]
    fn test_majority_vote() {
        let labeler = GroundTruthLabeler::new();
        let labels = vec![
            DecompositionMethod::Benders,
            DecompositionMethod::Benders,
            DecompositionMethod::DantzigWolfe,
        ];
        assert_eq!(labeler.majority_vote(&labels), DecompositionMethod::Benders);
    }

    #[test]
    fn test_stability_analysis() {
        let labeler = GroundTruthLabeler::new();
        let results = vec![
            LabelingResult {
                instance_name: "a".to_string(),
                labels_by_cutoff: HashMap::new(),
                consensus_label: DecompositionMethod::Benders,
                flip_rate: 0.0,
                is_stable: true,
                best_dual_bounds: HashMap::new(),
                improvement_over_none: HashMap::new(),
            },
            LabelingResult {
                instance_name: "b".to_string(),
                labels_by_cutoff: HashMap::new(),
                consensus_label: DecompositionMethod::DantzigWolfe,
                flip_rate: 0.5,
                is_stable: false,
                best_dual_bounds: HashMap::new(),
                improvement_over_none: HashMap::new(),
            },
        ];

        let report = labeler.stability_analysis(&results);
        assert_eq!(report.stable_instances, 1);
        assert_eq!(report.unstable_instances, 1);
        assert!((report.stability_fraction() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_consensus_labels() {
        let labeler = GroundTruthLabeler::new();
        let results = vec![LabelingResult {
            instance_name: "test".to_string(),
            labels_by_cutoff: {
                let mut m = HashMap::new();
                m.insert("60s".to_string(), DecompositionMethod::Benders);
                m
            },
            consensus_label: DecompositionMethod::Benders,
            flip_rate: 0.0,
            is_stable: true,
            best_dual_bounds: HashMap::new(),
            improvement_over_none: HashMap::new(),
        }];

        let consensus = labeler.generate_consensus_labels(&results);
        assert_eq!(consensus.len(), 1);
        assert_eq!(consensus[0].label, DecompositionMethod::Benders);
        assert_eq!(consensus[0].confidence, 1.0);
    }

    #[test]
    fn test_label_empty_results() {
        let labeler = GroundTruthLabeler::new();
        assert!(labeler.label_instance("test", &HashMap::new()).is_err());
    }

    #[test]
    fn test_improvement_computation() {
        let labeler = GroundTruthLabeler::new();
        let imp = labeler.compute_improvement(110.0, 100.0);
        assert!((imp - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_solve_status() {
        let result = DecompositionResult {
            method: DecompositionMethod::Benders,
            dual_bound: 100.0,
            primal_bound: Some(110.0),
            gap: Some(0.1),
            elapsed_secs: 30.0,
            status: SolveStatus::Optimal,
        };
        assert_eq!(result.status, SolveStatus::Optimal);
    }
}
