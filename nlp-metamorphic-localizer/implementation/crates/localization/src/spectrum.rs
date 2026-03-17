//! Spectrum collection (Phase 1) for CAUSAL_LOCALIZE.

use serde::{Deserialize, Serialize};
use shared_types::{IntermediateRepresentation, Result, StageId, TestCaseId, LocalizerError};
use std::collections::HashMap;
use std::time::Instant;

/// Collected spectrum data: differential matrix and violation vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumData {
    pub differential_matrix: Vec<Vec<f64>>,
    pub violation_vector: Vec<bool>,
    pub test_ids: Vec<String>,
    pub stage_names: Vec<String>,
    pub num_tests: usize,
    pub num_stages: usize,
}

impl SpectrumData {
    pub fn new(stage_names: Vec<String>) -> Self {
        let n = stage_names.len();
        Self {
            differential_matrix: Vec::new(),
            violation_vector: Vec::new(),
            test_ids: Vec::new(),
            stage_names,
            num_tests: 0,
            num_stages: n,
        }
    }

    pub fn add_test(&mut self, test_id: String, differentials: Vec<f64>, violated: bool) {
        self.test_ids.push(test_id);
        self.differential_matrix.push(differentials);
        self.violation_vector.push(violated);
        self.num_tests += 1;
    }

    pub fn violation_count(&self) -> usize {
        self.violation_vector.iter().filter(|&&v| v).count()
    }

    pub fn violation_rate(&self) -> f64 {
        if self.num_tests == 0 { return 0.0; }
        self.violation_count() as f64 / self.num_tests as f64
    }

    pub fn stage_mean_differential(&self, stage_idx: usize) -> f64 {
        if self.num_tests == 0 { return 0.0; }
        let sum: f64 = self.differential_matrix.iter()
            .map(|row| row.get(stage_idx).copied().unwrap_or(0.0))
            .sum();
        sum / self.num_tests as f64
    }

    pub fn stage_std_differential(&self, stage_idx: usize) -> f64 {
        let mean = self.stage_mean_differential(stage_idx);
        if self.num_tests <= 1 { return 0.0; }
        let variance: f64 = self.differential_matrix.iter()
            .map(|row| {
                let d = row.get(stage_idx).copied().unwrap_or(0.0) - mean;
                d * d
            })
            .sum::<f64>() / (self.num_tests - 1) as f64;
        variance.sqrt()
    }

    pub fn get_violation_rows(&self) -> Vec<&Vec<f64>> {
        self.differential_matrix.iter()
            .zip(self.violation_vector.iter())
            .filter(|(_, &v)| v)
            .map(|(row, _)| row)
            .collect()
    }

    pub fn get_passing_rows(&self) -> Vec<&Vec<f64>> {
        self.differential_matrix.iter()
            .zip(self.violation_vector.iter())
            .filter(|(_, &v)| !v)
            .map(|(row, _)| row)
            .collect()
    }
}

/// Result of a single test execution capturing per-stage IRs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecution {
    pub test_id: String,
    pub original_trace: Vec<StagePair>,
    pub per_stage_distances: Vec<f64>,
    pub violation: bool,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagePair {
    pub stage_name: String,
    pub stage_index: usize,
    pub original_output: String,
    pub transformed_output: String,
}

/// Spectrum collector: executes test suite and builds the differential matrix.
pub struct SpectrumCollector {
    pub stage_names: Vec<String>,
    pub cache_enabled: bool,
    cache: HashMap<String, Vec<f64>>,
}

impl SpectrumCollector {
    pub fn new(stage_names: Vec<String>) -> Self {
        Self {
            stage_names,
            cache_enabled: true,
            cache: HashMap::new(),
        }
    }

    /// Collect spectrum from pre-computed test executions.
    pub fn collect_from_executions(&mut self, executions: Vec<TestExecution>) -> SpectrumData {
        let mut data = SpectrumData::new(self.stage_names.clone());
        for exec in executions {
            data.add_test(exec.test_id, exec.per_stage_distances, exec.violation);
        }
        data
    }

    /// Collect spectrum from raw distance data.
    pub fn collect_from_distances(
        &mut self,
        test_ids: Vec<String>,
        distances: Vec<Vec<f64>>,
        violations: Vec<bool>,
    ) -> Result<SpectrumData> {
        if test_ids.len() != distances.len() || test_ids.len() != violations.len() {
            return Err(LocalizerError::ValidationError("Input lengths mismatch".into()));
        }
        let mut data = SpectrumData::new(self.stage_names.clone());
        for ((id, dists), &viol) in test_ids.into_iter().zip(distances).zip(violations.iter()) {
            data.add_test(id, dists, viol);
        }
        Ok(data)
    }
}

/// Statistics about the collected spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumStatistics {
    pub num_tests: usize,
    pub num_violations: usize,
    pub violation_rate: f64,
    pub per_stage_mean: Vec<f64>,
    pub per_stage_std: Vec<f64>,
    pub per_stage_max: Vec<f64>,
    pub max_differential_stage: Option<usize>,
    pub min_differential_stage: Option<usize>,
}

impl SpectrumStatistics {
    pub fn compute(data: &SpectrumData) -> Self {
        let n_stages = data.num_stages;
        let mut per_stage_mean = vec![0.0; n_stages];
        let mut per_stage_max = vec![0.0_f64; n_stages];

        for k in 0..n_stages {
            per_stage_mean[k] = data.stage_mean_differential(k);
            per_stage_max[k] = data.differential_matrix.iter()
                .map(|row| row.get(k).copied().unwrap_or(0.0))
                .fold(0.0_f64, f64::max);
        }

        let per_stage_std: Vec<f64> = (0..n_stages)
            .map(|k| data.stage_std_differential(k))
            .collect();

        let max_stage = per_stage_mean.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);

        let min_stage = per_stage_mean.iter().enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);

        Self {
            num_tests: data.num_tests,
            num_violations: data.violation_count(),
            violation_rate: data.violation_rate(),
            per_stage_mean,
            per_stage_std,
            per_stage_max,
            max_differential_stage: max_stage,
            min_differential_stage: min_stage,
        }
    }

    pub fn is_usable(&self) -> bool {
        self.num_tests >= 5 && self.num_violations >= 1
    }
}

/// Validates whether collected spectrum data is sufficient for localization.
pub fn validate_spectrum(data: &SpectrumData) -> Vec<String> {
    let mut warnings = Vec::new();

    if data.num_tests < 10 {
        warnings.push(format!("Only {} tests collected; recommend at least 10", data.num_tests));
    }

    let vr = data.violation_rate();
    if vr == 0.0 {
        warnings.push("No violations found; localization cannot proceed".into());
    }
    if vr == 1.0 {
        warnings.push("All tests violate; cannot distinguish faulty from non-faulty behavior".into());
    }
    if vr > 0.8 {
        warnings.push(format!("Very high violation rate ({:.1}%); results may be unreliable", vr * 100.0));
    }

    for k in 0..data.num_stages {
        let std = data.stage_std_differential(k);
        if std < 1e-10 {
            warnings.push(format!("Stage '{}' has zero variance; cannot contribute to localization",
                data.stage_names.get(k).unwrap_or(&format!("{}", k))));
        }
    }

    warnings
}

/// Progress tracker for long-running spectrum collection.
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    pub total: usize,
    pub completed: usize,
    pub start_time: Instant,
}

impl ProgressTracker {
    pub fn new(total: usize) -> Self {
        Self { total, completed: 0, start_time: Instant::now() }
    }

    pub fn tick(&mut self) {
        self.completed += 1;
    }

    pub fn progress(&self) -> f64 {
        if self.total == 0 { return 1.0; }
        self.completed as f64 / self.total as f64
    }

    pub fn estimated_remaining_ms(&self) -> u64 {
        let elapsed = self.start_time.elapsed().as_millis() as u64;
        if self.completed == 0 { return 0; }
        let per_item = elapsed / self.completed as u64;
        per_item * (self.total - self.completed) as u64
    }

    pub fn is_complete(&self) -> bool {
        self.completed >= self.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrum_data_creation() {
        let mut data = SpectrumData::new(vec!["tok".into(), "tag".into(), "parse".into()]);
        data.add_test("t1".into(), vec![0.1, 0.5, 0.2], true);
        data.add_test("t2".into(), vec![0.0, 0.1, 0.0], false);
        assert_eq!(data.num_tests, 2);
        assert_eq!(data.violation_count(), 1);
        assert!((data.violation_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_stage_statistics() {
        let mut data = SpectrumData::new(vec!["a".into(), "b".into()]);
        data.add_test("t1".into(), vec![1.0, 2.0], true);
        data.add_test("t2".into(), vec![3.0, 4.0], false);
        assert!((data.stage_mean_differential(0) - 2.0).abs() < 0.01);
        assert!(data.stage_std_differential(0) > 0.0);
    }

    #[test]
    fn test_spectrum_collector() {
        let mut collector = SpectrumCollector::new(vec!["s1".into(), "s2".into()]);
        let data = collector.collect_from_distances(
            vec!["t1".into()],
            vec![vec![0.5, 0.3]],
            vec![true],
        ).unwrap();
        assert_eq!(data.num_tests, 1);
    }

    #[test]
    fn test_spectrum_statistics_compute() {
        let mut data = SpectrumData::new(vec!["a".into()]);
        data.add_test("t1".into(), vec![0.5], true);
        data.add_test("t2".into(), vec![0.1], false);
        let stats = SpectrumStatistics::compute(&data);
        assert_eq!(stats.num_tests, 2);
        assert_eq!(stats.num_violations, 1);
    }

    #[test]
    fn test_validate_spectrum() {
        let mut data = SpectrumData::new(vec!["a".into()]);
        let warnings = validate_spectrum(&data);
        assert!(!warnings.is_empty()); // Should warn about no tests
    }

    #[test]
    fn test_progress_tracker() {
        let mut tracker = ProgressTracker::new(10);
        assert!((tracker.progress() - 0.0).abs() < 0.01);
        tracker.tick();
        assert!((tracker.progress() - 0.1).abs() < 0.01);
        assert!(!tracker.is_complete());
    }

    #[test]
    fn test_violation_rows() {
        let mut data = SpectrumData::new(vec!["a".into()]);
        data.add_test("t1".into(), vec![0.5], true);
        data.add_test("t2".into(), vec![0.1], false);
        data.add_test("t3".into(), vec![0.9], true);
        assert_eq!(data.get_violation_rows().len(), 2);
        assert_eq!(data.get_passing_rows().len(), 1);
    }

    #[test]
    fn test_collector_mismatch() {
        let mut collector = SpectrumCollector::new(vec!["s1".into()]);
        let result = collector.collect_from_distances(
            vec!["t1".into()],
            vec![vec![0.5]],
            vec![],
        );
        assert!(result.is_err());
    }
}
