//! Calibration for fault localization.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use shared_types::{LocalizerError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationData {
    pub per_stage_means: Vec<f64>,
    pub per_stage_stds: Vec<f64>,
    pub per_transformation_baselines: HashMap<String, Vec<f64>>,
    pub sample_count: usize,
    pub stage_names: Vec<String>,
}

impl CalibrationData {
    pub fn new(stage_names: Vec<String>) -> Self {
        let n = stage_names.len();
        Self {
            per_stage_means: vec![0.0; n],
            per_stage_stds: vec![1.0; n],
            per_transformation_baselines: HashMap::new(),
            sample_count: 0,
            stage_names,
        }
    }

    pub fn num_stages(&self) -> usize { self.stage_names.len() }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| LocalizerError::SerializationError(e.to_string()))
    }

    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| LocalizerError::SerializationError(e.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationQuality {
    pub per_stage_coefficient_of_variation: Vec<f64>,
    pub overall_stability: f64,
    pub min_samples_per_cell: usize,
    pub sufficient: bool,
    pub recommendations: Vec<String>,
}

pub struct LocalizationCalibrator {
    pub stage_names: Vec<String>,
    pub min_samples: usize,
    pub max_cov: f64,
    running_sums: Vec<f64>,
    running_sum_sq: Vec<f64>,
    count: usize,
    transformation_sums: HashMap<String, Vec<f64>>,
    transformation_counts: HashMap<String, usize>,
}

impl LocalizationCalibrator {
    pub fn new(stage_names: Vec<String>, min_samples: usize) -> Self {
        let n = stage_names.len();
        Self {
            stage_names,
            min_samples,
            max_cov: 0.5,
            running_sums: vec![0.0; n],
            running_sum_sq: vec![0.0; n],
            count: 0,
            transformation_sums: HashMap::new(),
            transformation_counts: HashMap::new(),
        }
    }

    pub fn add_sample(&mut self, differentials: &[f64], transformation: Option<&str>) {
        for (k, &d) in differentials.iter().enumerate() {
            if k < self.running_sums.len() {
                self.running_sums[k] += d;
                self.running_sum_sq[k] += d * d;
            }
        }
        self.count += 1;

        if let Some(t_name) = transformation {
            let entry = self.transformation_sums.entry(t_name.to_string()).or_insert_with(|| vec![0.0; self.stage_names.len()]);
            for (k, &d) in differentials.iter().enumerate() {
                if k < entry.len() { entry[k] += d; }
            }
            *self.transformation_counts.entry(t_name.to_string()).or_insert(0) += 1;
        }
    }

    pub fn run_calibration(&mut self, samples: &[Vec<f64>]) -> CalibrationData {
        for s in samples { self.add_sample(s, None); }
        self.get_calibration_data()
    }

    pub fn get_calibration_data(&self) -> CalibrationData {
        let n = self.stage_names.len();
        let means: Vec<f64> = (0..n).map(|k| {
            if self.count > 0 { self.running_sums[k] / self.count as f64 } else { 0.0 }
        }).collect();

        let stds: Vec<f64> = (0..n).map(|k| {
            if self.count <= 1 { return 1.0; }
            let mean = means[k];
            let variance = self.running_sum_sq[k] / self.count as f64 - mean * mean;
            variance.max(0.0).sqrt().max(0.001)
        }).collect();

        let mut baselines = HashMap::new();
        for (t_name, sums) in &self.transformation_sums {
            let t_count = self.transformation_counts.get(t_name).copied().unwrap_or(1) as f64;
            let t_means: Vec<f64> = sums.iter().map(|s| s / t_count).collect();
            baselines.insert(t_name.clone(), t_means);
        }

        CalibrationData {
            per_stage_means: means,
            per_stage_stds: stds,
            per_transformation_baselines: baselines,
            sample_count: self.count,
            stage_names: self.stage_names.clone(),
        }
    }

    pub fn is_sufficient(&self) -> bool {
        if self.count < self.min_samples { return false; }
        let data = self.get_calibration_data();
        let quality = assess_quality(&data);
        quality.sufficient
    }

    pub fn update_calibration(&mut self, differential: &[f64]) {
        self.add_sample(differential, None);
    }

    pub fn recommend_additional_samples(&self) -> usize {
        if self.count >= self.min_samples { return 0; }
        self.min_samples - self.count
    }
}

pub fn assess_quality(data: &CalibrationData) -> CalibrationQuality {
    let covs: Vec<f64> = data.per_stage_means.iter().zip(data.per_stage_stds.iter()).map(|(&m, &s)| {
        if m.abs() < 1e-10 { 0.0 } else { s / m.abs() }
    }).collect();

    let overall = if covs.is_empty() { 0.0 } else { covs.iter().sum::<f64>() / covs.len() as f64 };
    let sufficient = data.sample_count >= 30 && overall < 0.5;

    let mut recs = Vec::new();
    if data.sample_count < 30 { recs.push(format!("Need at least 30 samples, have {}", data.sample_count)); }
    for (i, &cov) in covs.iter().enumerate() {
        if cov > 1.0 {
            recs.push(format!("Stage '{}' has very high CoV ({:.2})", data.stage_names.get(i).unwrap_or(&format!("{}", i)), cov));
        }
    }

    CalibrationQuality {
        per_stage_coefficient_of_variation: covs,
        overall_stability: 1.0 - overall.min(1.0),
        min_samples_per_cell: data.sample_count,
        sufficient,
        recommendations: recs,
    }
}

pub fn apply_to_spectrum(
    spectrum_matrix: &mut [Vec<f64>],
    calibration: &CalibrationData,
) {
    for row in spectrum_matrix.iter_mut() {
        for (k, val) in row.iter_mut().enumerate() {
            let mean = calibration.per_stage_means.get(k).copied().unwrap_or(0.0);
            let std = calibration.per_stage_stds.get(k).copied().unwrap_or(1.0);
            *val = (*val - mean) / std.max(0.001);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibrator_basic() {
        let mut cal = LocalizationCalibrator::new(vec!["a".into(), "b".into()], 10);
        for _ in 0..20 {
            cal.add_sample(&[0.5, 0.3], None);
        }
        let data = cal.get_calibration_data();
        assert_eq!(data.sample_count, 20);
        assert!((data.per_stage_means[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_calibration_data_json() {
        let data = CalibrationData::new(vec!["a".into(), "b".into()]);
        let json = data.to_json().unwrap();
        let parsed = CalibrationData::from_json(&json).unwrap();
        assert_eq!(parsed.stage_names, data.stage_names);
    }

    #[test]
    fn test_is_sufficient() {
        let mut cal = LocalizationCalibrator::new(vec!["a".into()], 10);
        assert!(!cal.is_sufficient());
        for _ in 0..50 {
            cal.add_sample(&[0.5], None);
        }
        assert!(cal.is_sufficient());
    }

    #[test]
    fn test_assess_quality() {
        let data = CalibrationData {
            per_stage_means: vec![0.5, 0.3],
            per_stage_stds: vec![0.1, 0.05],
            per_transformation_baselines: HashMap::new(),
            sample_count: 100,
            stage_names: vec!["a".into(), "b".into()],
        };
        let quality = assess_quality(&data);
        assert!(quality.sufficient);
        assert!(quality.overall_stability > 0.5);
    }

    #[test]
    fn test_apply_to_spectrum() {
        let cal = CalibrationData {
            per_stage_means: vec![1.0, 2.0],
            per_stage_stds: vec![0.5, 0.5],
            per_transformation_baselines: HashMap::new(),
            sample_count: 50,
            stage_names: vec!["a".into(), "b".into()],
        };
        let mut matrix = vec![vec![1.5, 2.5]];
        apply_to_spectrum(&mut matrix, &cal);
        assert!((matrix[0][0] - 1.0).abs() < 0.01);
        assert!((matrix[0][1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_recommend_samples() {
        let cal = LocalizationCalibrator::new(vec!["a".into()], 50);
        assert_eq!(cal.recommend_additional_samples(), 50);
    }

    #[test]
    fn test_transformation_baselines() {
        let mut cal = LocalizationCalibrator::new(vec!["a".into(), "b".into()], 5);
        cal.add_sample(&[0.5, 0.3], Some("passivize"));
        cal.add_sample(&[0.7, 0.1], Some("passivize"));
        cal.add_sample(&[0.2, 0.8], Some("cleft"));
        let data = cal.get_calibration_data();
        assert!(data.per_transformation_baselines.contains_key("passivize"));
        assert!(data.per_transformation_baselines.contains_key("cleft"));
    }

    #[test]
    fn test_update_calibration() {
        let mut cal = LocalizationCalibrator::new(vec!["a".into()], 5);
        cal.update_calibration(&[0.3]);
        cal.update_calibration(&[0.7]);
        let data = cal.get_calibration_data();
        assert_eq!(data.sample_count, 2);
    }
}
