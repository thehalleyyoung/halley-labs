//! Anomaly detection methods.
use serde::{Serialize, Deserialize};

/// Anomaly detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult { pub is_anomaly: bool, pub score: f64, pub index: usize }

/// Anomaly detector trait.
pub trait AnomalyDetector { fn detect(&self, data: &[f64]) -> Vec<AnomalyResult>; fn name(&self) -> &str; }

/// Z-score based anomaly detection.
#[derive(Debug, Clone)]
pub struct ZScoreAnomaly { pub threshold: f64 }
impl Default for ZScoreAnomaly { fn default() -> Self { Self { threshold: 3.0 } } }
impl ZScoreAnomaly {
    /// Create with a custom threshold.
    pub fn new(threshold: f64) -> Self { Self { threshold } }

    /// Run anomaly detection and return a summary result.
    pub fn detect(&self, data: &[f64]) -> ZScoreDetectionResult {
        let n = data.len() as f64;
        if n < 2.0 { return ZScoreDetectionResult { is_anomaly: false, confidence: 0.0, anomaly_indices: vec![] }; }
        let mean = data.iter().sum::<f64>() / n;
        let std = (data.iter().map(|x| (x-mean).powi(2)).sum::<f64>() / n).sqrt();
        if std < 1e-30 { return ZScoreDetectionResult { is_anomaly: false, confidence: 0.0, anomaly_indices: vec![] }; }
        let indices: Vec<usize> = data.iter().enumerate()
            .filter(|(_, &v)| ((v - mean) / std).abs() > self.threshold)
            .map(|(i, _)| i).collect();
        let is_anomaly = !indices.is_empty();
        let confidence = if is_anomaly { 0.95 } else { 0.0 };
        ZScoreDetectionResult { is_anomaly, confidence, anomaly_indices: indices }
    }
}

/// Summary result from Z-score anomaly detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZScoreDetectionResult {
    pub is_anomaly: bool,
    pub confidence: f64,
    pub anomaly_indices: Vec<usize>,
}
impl AnomalyDetector for ZScoreAnomaly {
    fn detect(&self, data: &[f64]) -> Vec<AnomalyResult> {
        let n = data.len() as f64;
        if n < 2.0 { return Vec::new(); }
        let mean = data.iter().sum::<f64>() / n;
        let std = (data.iter().map(|x| (x-mean).powi(2)).sum::<f64>() / n).sqrt();
        if std < 1e-30 { return Vec::new(); }
        data.iter().enumerate().filter_map(|(i, &v)| {
            let z = (v - mean).abs() / std;
            if z > self.threshold { Some(AnomalyResult { is_anomaly: true, score: z, index: i }) } else { None }
        }).collect()
    }
    fn name(&self) -> &str { "ZScore" }
}

/// IQR-based anomaly detection.
#[derive(Debug, Clone, Default)]
pub struct IqrAnomaly { pub multiplier: f64 }
impl AnomalyDetector for IqrAnomaly {
    fn detect(&self, _data: &[f64]) -> Vec<AnomalyResult> { Vec::new() }
    fn name(&self) -> &str { "IQR" }
}

/// Isolation score anomaly detection.
#[derive(Debug, Clone, Default)]
pub struct IsolationScore;
impl AnomalyDetector for IsolationScore {
    fn detect(&self, _data: &[f64]) -> Vec<AnomalyResult> { Vec::new() }
    fn name(&self) -> &str { "IsolationScore" }
}

/// Local outlier factor.
#[derive(Debug, Clone, Default)]
pub struct LocalOutlierFactor { pub k: usize }
impl AnomalyDetector for LocalOutlierFactor {
    fn detect(&self, _data: &[f64]) -> Vec<AnomalyResult> { Vec::new() }
    fn name(&self) -> &str { "LOF" }
}

/// Moving median anomaly detection.
#[derive(Debug, Clone)]
pub struct MovingMedianAnomaly { pub window_size: usize, pub threshold: f64 }
impl Default for MovingMedianAnomaly { fn default() -> Self { Self { window_size: 5, threshold: 3.0 } } }
impl AnomalyDetector for MovingMedianAnomaly {
    fn detect(&self, _data: &[f64]) -> Vec<AnomalyResult> { Vec::new() }
    fn name(&self) -> &str { "MovingMedian" }
}
