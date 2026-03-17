use serde::{Deserialize, Serialize};
use shared_types::{ConfidenceInterval, StageId};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSample {
    pub stage_id: StageId,
    pub stage_name: String,
    pub differentials: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBaseline {
    pub stage_name: String,
    pub mean: f64,
    pub std_dev: f64,
    pub threshold: f64,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationCorpus {
    pub sentences: Vec<String>,
}

impl CalibrationCorpus {
    pub fn new(sentences: Vec<String>) -> Self { Self { sentences } }
    pub fn len(&self) -> usize { self.sentences.len() }
    pub fn is_empty(&self) -> bool { self.sentences.is_empty() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub baselines: HashMap<String, CalibrationBaseline>,
    pub sample_count: usize,
    pub quality_score: f64,
    pub confidence_intervals: HashMap<String, ConfidenceInterval>,
}

pub struct CalibrationEngine {
    pub n_runs: usize,
    pub warmup: usize,
}

impl CalibrationEngine {
    pub fn new(n_runs: usize) -> Self { Self { n_runs, warmup: 5 } }
}
