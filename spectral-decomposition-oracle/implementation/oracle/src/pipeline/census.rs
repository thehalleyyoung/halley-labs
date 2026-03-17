// Census infrastructure: processing all MIPLIB instances through the oracle pipeline.
// Tiered execution, job tracking, result aggregation, and statistical summaries.

use crate::classifier::traits::DecompositionMethod;
use crate::error::{OracleError, OracleResult};
use crate::pipeline::oracle_pipeline::{BatchStatistics, PipelineResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Census tier controlling the number of instances to process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CensusTier {
    Pilot,    // 50 instances
    Dev,      // 200 instances
    Paper,    // 500 instances
    Artifact, // 1065 instances (full MIPLIB)
}

impl CensusTier {
    pub fn instance_count(&self) -> usize {
        match self {
            CensusTier::Pilot => 50,
            CensusTier::Dev => 200,
            CensusTier::Paper => 500,
            CensusTier::Artifact => 1065,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            CensusTier::Pilot => "pilot",
            CensusTier::Dev => "dev",
            CensusTier::Paper => "paper",
            CensusTier::Artifact => "artifact",
        }
    }
}

impl std::fmt::Display for CensusTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (n={})", self.name(), self.instance_count())
    }
}

/// Status of a single instance in the census.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstanceStatus {
    Pending,
    Running,
    Completed(PipelineResult),
    Failed(String),
    Skipped(String),
}

impl InstanceStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            InstanceStatus::Completed(_) | InstanceStatus::Failed(_) | InstanceStatus::Skipped(_)
        )
    }

    pub fn is_completed(&self) -> bool {
        matches!(self, InstanceStatus::Completed(_))
    }
}

/// Census result: aggregated results from processing all instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CensusResult {
    pub tier: CensusTier,
    pub total_instances: usize,
    pub completed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub results: Vec<PipelineResult>,
    pub method_distribution: HashMap<String, usize>,
    pub coverage: f64,
    pub total_elapsed_secs: f64,
}

impl CensusResult {
    pub fn summary(&self) -> String {
        format!(
            "Census [{}]: {}/{} completed ({} failed, {} skipped) | coverage={:.1}% | time={:.1}s\nMethod distribution: {:?}",
            self.tier.name(),
            self.completed,
            self.total_instances,
            self.failed,
            self.skipped,
            self.coverage * 100.0,
            self.total_elapsed_secs,
            self.method_distribution
        )
    }

    pub fn batch_stats(&self) -> BatchStatistics {
        BatchStatistics::compute(&self.results)
    }
}

/// Job queue for census processing.
#[derive(Debug)]
pub struct JobQueue {
    pub jobs: Vec<Job>,
}

/// A single job in the queue.
#[derive(Debug, Clone)]
pub struct Job {
    pub instance_name: String,
    pub status: InstanceStatus,
    pub attempt: usize,
    pub max_attempts: usize,
}

impl JobQueue {
    pub fn new(instance_names: Vec<String>) -> Self {
        let jobs = instance_names
            .into_iter()
            .map(|name| Job {
                instance_name: name,
                status: InstanceStatus::Pending,
                attempt: 0,
                max_attempts: 2,
            })
            .collect();
        Self { jobs }
    }

    /// Get the next pending job.
    pub fn next_pending(&mut self) -> Option<&mut Job> {
        self.jobs.iter_mut().find(|j| matches!(j.status, InstanceStatus::Pending))
    }

    /// Mark a job as running.
    pub fn mark_running(&mut self, name: &str) {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.instance_name == name) {
            job.status = InstanceStatus::Running;
            job.attempt += 1;
        }
    }

    /// Mark a job as completed with result.
    pub fn mark_completed(&mut self, name: &str, result: PipelineResult) {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.instance_name == name) {
            job.status = InstanceStatus::Completed(result);
        }
    }

    /// Mark a job as failed. If retries remain, reset to pending.
    pub fn mark_failed(&mut self, name: &str, error: &str) {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.instance_name == name) {
            if job.attempt < job.max_attempts {
                job.status = InstanceStatus::Pending; // retry
            } else {
                job.status = InstanceStatus::Failed(error.to_string());
            }
        }
    }

    /// Mark a job as skipped.
    pub fn mark_skipped(&mut self, name: &str, reason: &str) {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.instance_name == name) {
            job.status = InstanceStatus::Skipped(reason.to_string());
        }
    }

    /// Count jobs by status.
    pub fn status_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for job in &self.jobs {
            let key = match &job.status {
                InstanceStatus::Pending => "pending",
                InstanceStatus::Running => "running",
                InstanceStatus::Completed(_) => "completed",
                InstanceStatus::Failed(_) => "failed",
                InstanceStatus::Skipped(_) => "skipped",
            };
            *counts.entry(key.to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Get all completed results.
    pub fn completed_results(&self) -> Vec<PipelineResult> {
        self.jobs
            .iter()
            .filter_map(|j| match &j.status {
                InstanceStatus::Completed(r) => Some(r.clone()),
                _ => None,
            })
            .collect()
    }

    /// Check if all jobs are in a terminal state.
    pub fn all_done(&self) -> bool {
        self.jobs.iter().all(|j| j.status.is_terminal())
    }

    /// Progress as a fraction [0, 1].
    pub fn progress(&self) -> f64 {
        let done = self.jobs.iter().filter(|j| j.status.is_terminal()).count();
        done as f64 / self.jobs.len().max(1) as f64
    }
}

/// Census pipeline: orchestrates processing of MIPLIB instances.
pub struct CensusPipeline {
    pub tier: CensusTier,
    pub queue: JobQueue,
    pub completed_names: std::collections::HashSet<String>,
}

impl CensusPipeline {
    pub fn new(tier: CensusTier, instance_names: Vec<String>) -> OracleResult<Self> {
        let expected = tier.instance_count();
        let actual = instance_names.len();
        if actual == 0 {
            return Err(OracleError::invalid_input("no instances provided"));
        }

        // Take at most the tier count
        let names: Vec<String> = instance_names.into_iter().take(expected).collect();

        Ok(Self {
            tier,
            queue: JobQueue::new(names),
            completed_names: std::collections::HashSet::new(),
        })
    }

    /// Skip instances that have already been processed (idempotent execution).
    pub fn skip_completed(&mut self, completed: &[String]) {
        for name in completed {
            self.completed_names.insert(name.clone());
            self.queue.mark_skipped(name, "already completed");
        }
    }

    /// Process results for a batch of instances.
    pub fn record_results(&mut self, results: Vec<PipelineResult>) {
        for result in results {
            let name = result.instance_name.clone();
            self.completed_names.insert(name.clone());
            self.queue.mark_completed(&name, result);
        }
    }

    /// Record a failure for an instance.
    pub fn record_failure(&mut self, name: &str, error: &str) {
        self.queue.mark_failed(name, error);
    }

    /// Get the next batch of instance names to process.
    pub fn next_batch(&mut self, batch_size: usize) -> Vec<String> {
        let mut batch = Vec::new();
        for job in &mut self.queue.jobs {
            if batch.len() >= batch_size {
                break;
            }
            if matches!(job.status, InstanceStatus::Pending) {
                job.status = InstanceStatus::Running;
                job.attempt += 1;
                batch.push(job.instance_name.clone());
            }
        }
        batch
    }

    /// Generate the census result.
    pub fn finalize(&self) -> CensusResult {
        let results = self.queue.completed_results();
        let counts = self.queue.status_counts();

        let completed = *counts.get("completed").unwrap_or(&0);
        let failed = *counts.get("failed").unwrap_or(&0);
        let skipped = *counts.get("skipped").unwrap_or(&0);
        let total = self.queue.jobs.len();

        let mut method_dist = HashMap::new();
        for result in &results {
            if let Some(method) = result.recommended_method {
                *method_dist.entry(method.to_string()).or_insert(0) += 1;
            }
        }

        let coverage = completed as f64 / total.max(1) as f64;
        let total_elapsed = results.iter().map(|r| r.total_elapsed_secs).sum();

        CensusResult {
            tier: self.tier,
            total_instances: total,
            completed,
            failed,
            skipped,
            results,
            method_distribution: method_dist,
            coverage,
            total_elapsed_secs: total_elapsed,
        }
    }

    /// Coverage metrics: fraction of instances processed per structure type, etc.
    pub fn coverage_report(&self) -> CoverageReport {
        let results = self.queue.completed_results();
        let total = self.queue.jobs.len();
        let completed = results.len();

        let mut method_counts = HashMap::new();
        let mut futile_count = 0usize;
        let mut confidence_sum = 0.0_f64;

        for r in &results {
            if let Some(m) = r.recommended_method {
                *method_counts.entry(m).or_insert(0usize) += 1;
            }
            if r.is_futile == Some(true) {
                futile_count += 1;
            }
            confidence_sum += r.confidence;
        }

        let avg_confidence = if completed > 0 {
            confidence_sum / completed as f64
        } else {
            0.0
        };

        CoverageReport {
            total_instances: total,
            completed_instances: completed,
            coverage_fraction: completed as f64 / total.max(1) as f64,
            method_counts,
            futile_count,
            avg_confidence,
        }
    }
}

/// Coverage report for census progress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub total_instances: usize,
    pub completed_instances: usize,
    pub coverage_fraction: f64,
    pub method_counts: HashMap<DecompositionMethod, usize>,
    pub futile_count: usize,
    pub avg_confidence: f64,
}

/// Export census results to structured format.
pub fn export_results_json(result: &CensusResult) -> OracleResult<String> {
    serde_json::to_string_pretty(result).map_err(|e| OracleError::serialization(e.to_string()))
}

/// Compute statistical summary of census results.
pub fn statistical_summary(results: &[PipelineResult]) -> StatisticalSummary {
    let n = results.len();
    if n == 0 {
        return StatisticalSummary::default();
    }

    let confidences: Vec<f64> = results.iter().map(|r| r.confidence).collect();
    let elapsed: Vec<f64> = results.iter().map(|r| r.total_elapsed_secs).collect();

    let mean_confidence = confidences.iter().sum::<f64>() / n as f64;
    let mean_elapsed = elapsed.iter().sum::<f64>() / n as f64;

    let std_confidence = if n > 1 {
        let var = confidences
            .iter()
            .map(|&c| (c - mean_confidence).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    let mut sorted_conf = confidences.clone();
    sorted_conf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_confidence = sorted_conf[n / 2];

    let complete_rate = results.iter().filter(|r| r.is_complete()).count() as f64 / n as f64;

    StatisticalSummary {
        n_instances: n,
        mean_confidence,
        std_confidence,
        median_confidence,
        mean_elapsed_secs: mean_elapsed,
        complete_rate,
    }
}

/// Statistical summary of census execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub n_instances: usize,
    pub mean_confidence: f64,
    pub std_confidence: f64,
    pub median_confidence: f64,
    pub mean_elapsed_secs: f64,
    pub complete_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(name: &str, method: DecompositionMethod) -> PipelineResult {
        PipelineResult {
            instance_name: name.to_string(),
            recommended_method: Some(method),
            confidence: 0.8,
            class_probabilities: vec![],
            is_futile: Some(false),
            futility_score: None,
            features: None,
            stages: vec![],
            total_elapsed_secs: 1.0,
            partial: false,
        }
    }

    #[test]
    fn test_census_tier_counts() {
        assert_eq!(CensusTier::Pilot.instance_count(), 50);
        assert_eq!(CensusTier::Dev.instance_count(), 200);
        assert_eq!(CensusTier::Paper.instance_count(), 500);
        assert_eq!(CensusTier::Artifact.instance_count(), 1065);
    }

    #[test]
    fn test_census_tier_display() {
        let s = CensusTier::Pilot.to_string();
        assert!(s.contains("pilot"));
        assert!(s.contains("50"));
    }

    #[test]
    fn test_job_queue_basic() {
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut queue = JobQueue::new(names);
        assert!(!queue.all_done());
        assert_eq!(queue.progress(), 0.0);
    }

    #[test]
    fn test_job_queue_next_pending() {
        let names = vec!["a".to_string(), "b".to_string()];
        let mut queue = JobQueue::new(names);
        let job = queue.next_pending().unwrap();
        assert_eq!(job.instance_name, "a");
    }

    #[test]
    fn test_job_queue_lifecycle() {
        let names = vec!["inst1".to_string()];
        let mut queue = JobQueue::new(names);

        queue.mark_running("inst1");
        assert!(!queue.all_done());

        let result = make_result("inst1", DecompositionMethod::Benders);
        queue.mark_completed("inst1", result);
        assert!(queue.all_done());
    }

    #[test]
    fn test_job_queue_retry() {
        let names = vec!["inst1".to_string()];
        let mut queue = JobQueue::new(names);

        queue.mark_running("inst1");
        queue.mark_failed("inst1", "error");
        // Should retry (attempt 1 of 2)
        assert!(!queue.all_done());

        queue.mark_running("inst1");
        queue.mark_failed("inst1", "error again");
        // Now should be terminal
        assert!(queue.all_done());
    }

    #[test]
    fn test_census_pipeline_basic() {
        let names: Vec<String> = (0..10).map(|i| format!("inst_{}", i)).collect();
        let pipeline = CensusPipeline::new(CensusTier::Pilot, names).unwrap();
        assert_eq!(pipeline.tier, CensusTier::Pilot);
    }

    #[test]
    fn test_census_pipeline_empty() {
        let result = CensusPipeline::new(CensusTier::Pilot, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_census_skip_completed() {
        let names: Vec<String> = (0..5).map(|i| format!("inst_{}", i)).collect();
        let mut pipeline = CensusPipeline::new(CensusTier::Pilot, names).unwrap();
        pipeline.skip_completed(&["inst_0".to_string(), "inst_1".to_string()]);

        let counts = pipeline.queue.status_counts();
        assert_eq!(*counts.get("skipped").unwrap_or(&0), 2);
    }

    #[test]
    fn test_census_next_batch() {
        let names: Vec<String> = (0..10).map(|i| format!("inst_{}", i)).collect();
        let mut pipeline = CensusPipeline::new(CensusTier::Pilot, names).unwrap();
        let batch = pipeline.next_batch(3);
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_census_finalize() {
        let names: Vec<String> = (0..3).map(|i| format!("inst_{}", i)).collect();
        let mut pipeline = CensusPipeline::new(CensusTier::Pilot, names).unwrap();

        let results = vec![
            make_result("inst_0", DecompositionMethod::Benders),
            make_result("inst_1", DecompositionMethod::DantzigWolfe),
        ];
        pipeline.record_results(results);
        pipeline.record_failure("inst_2", "timeout");

        let census_result = pipeline.finalize();
        assert_eq!(census_result.completed, 2);
    }

    #[test]
    fn test_statistical_summary() {
        let results = vec![
            make_result("a", DecompositionMethod::Benders),
            make_result("b", DecompositionMethod::DantzigWolfe),
            make_result("c", DecompositionMethod::None),
        ];
        let summary = statistical_summary(&results);
        assert_eq!(summary.n_instances, 3);
        assert!((summary.mean_confidence - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_statistical_summary_empty() {
        let summary = statistical_summary(&[]);
        assert_eq!(summary.n_instances, 0);
    }

    #[test]
    fn test_export_json() {
        let census = CensusResult {
            tier: CensusTier::Pilot,
            total_instances: 1,
            completed: 1,
            failed: 0,
            skipped: 0,
            results: vec![make_result("a", DecompositionMethod::Benders)],
            method_distribution: HashMap::new(),
            coverage: 1.0,
            total_elapsed_secs: 1.0,
        };
        let json = export_results_json(&census).unwrap();
        assert!(json.contains("Pilot"));
    }

    #[test]
    fn test_instance_status_terminal() {
        assert!(!InstanceStatus::Pending.is_terminal());
        assert!(!InstanceStatus::Running.is_terminal());
        assert!(InstanceStatus::Failed("err".to_string()).is_terminal());
        assert!(InstanceStatus::Skipped("reason".to_string()).is_terminal());
    }

    #[test]
    fn test_coverage_report() {
        let names: Vec<String> = (0..5).map(|i| format!("inst_{}", i)).collect();
        let mut pipeline = CensusPipeline::new(CensusTier::Pilot, names).unwrap();
        pipeline.record_results(vec![
            make_result("inst_0", DecompositionMethod::Benders),
            make_result("inst_1", DecompositionMethod::None),
        ]);
        let report = pipeline.coverage_report();
        assert_eq!(report.completed_instances, 2);
        assert!(report.coverage_fraction > 0.0);
    }
}
