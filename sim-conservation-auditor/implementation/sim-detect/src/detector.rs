//! Violation detection pipeline: orchestrates preprocessing, statistical tests,
//! classification, and localization of conservation law violations.

use serde::{Deserialize, Serialize};
use sim_types::{ConservationKind, TimeSeries, Tolerance, ViolationSeverity};
use crate::drift_detect::DriftDetector;

// ---------------------------------------------------------------------------
// Core Detector trait
// ---------------------------------------------------------------------------

/// Trait implemented by all detection algorithms.
pub trait Detector {
    /// Run detection on a time-series of conservation-quantity errors.
    fn detect(&self, errors: &[f64]) -> DetectionResult;

    /// Human-readable name of the detector.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Detection result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub detected: bool,
    pub confidence: f64,
    pub p_value: Option<f64>,
    pub violation_kind: Option<ConservationKind>,
    pub violation_severity: Option<ViolationSeverity>,
    pub details: String,
    pub change_point_index: Option<usize>,
    pub test_statistic: Option<f64>,
}

impl DetectionResult {
    pub fn no_violation() -> Self {
        Self {
            detected: false,
            confidence: 0.0,
            p_value: None,
            violation_kind: None,
            violation_severity: None,
            details: "No violation detected".to_string(),
            change_point_index: None,
            test_statistic: None,
        }
    }

    pub fn violation(confidence: f64, p_value: f64, detail: &str) -> Self {
        Self {
            detected: true,
            confidence,
            p_value: Some(p_value),
            violation_kind: None,
            violation_severity: None,
            details: detail.to_string(),
            change_point_index: None,
            test_statistic: None,
        }
    }

    pub fn with_kind(mut self, kind: ConservationKind) -> Self {
        self.violation_kind = Some(kind);
        self
    }

    pub fn with_severity(mut self, severity: ViolationSeverity) -> Self {
        self.violation_severity = Some(severity);
        self
    }

    pub fn with_change_point(mut self, idx: usize) -> Self {
        self.change_point_index = Some(idx);
        self
    }

    pub fn with_statistic(mut self, stat: f64) -> Self {
        self.test_statistic = Some(stat);
        self
    }
}

// ---------------------------------------------------------------------------
// Detection configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    pub significance_level: f64,
    pub min_samples: usize,
    pub tolerance: Tolerance,
    pub check_energy: bool,
    pub check_momentum: bool,
    pub check_angular_momentum: bool,
    pub check_mass: bool,
    pub warmup_samples: usize,
    pub sliding_window_size: usize,
    pub enable_drift_detection: bool,
    pub enable_anomaly_detection: bool,
    pub enable_change_point_detection: bool,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            significance_level: 0.05,
            min_samples: 30,
            tolerance: Tolerance::absolute(1e-10),
            check_energy: true,
            check_momentum: true,
            check_angular_momentum: true,
            check_mass: true,
            warmup_samples: 20,
            sliding_window_size: 50,
            enable_drift_detection: true,
            enable_anomaly_detection: true,
            enable_change_point_detection: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-law detection result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLawResult {
    pub results: Vec<(ConservationKind, DetectionResult)>,
    pub overall_detected: bool,
    pub worst_p_value: Option<f64>,
}

impl MultiLawResult {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            overall_detected: false,
            worst_p_value: None,
        }
    }

    pub fn add(&mut self, kind: ConservationKind, result: DetectionResult) {
        if result.detected {
            self.overall_detected = true;
        }
        if let Some(pv) = result.p_value {
            match self.worst_p_value {
                None => self.worst_p_value = Some(pv),
                Some(current) => {
                    if pv < current {
                        self.worst_p_value = Some(pv);
                    }
                }
            }
        }
        self.results.push((kind, result));
    }

    pub fn violations(&self) -> Vec<&(ConservationKind, DetectionResult)> {
        self.results.iter().filter(|(_, r)| r.detected).collect()
    }
}

impl Default for MultiLawResult {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Preprocessing utilities
// ---------------------------------------------------------------------------

/// Compute running differences: errors[i] = values[i] - reference
pub fn compute_errors(values: &[f64], reference: f64) -> Vec<f64> {
    values.iter().map(|&v| v - reference).collect()
}

/// Compute relative errors: (values[i] - reference) / |reference|
pub fn compute_relative_errors(values: &[f64], reference: f64) -> Vec<f64> {
    if reference.abs() < 1e-30 {
        return values.iter().map(|&v| v).collect();
    }
    values.iter().map(|&v| (v - reference) / reference.abs()).collect()
}

/// Detrend a time series by subtracting a linear fit.
pub fn detrend(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n < 2 {
        return data.to_vec();
    }
    let n_f = n as f64;
    let sum_x: f64 = (0..n).map(|i| i as f64).sum();
    let sum_y: f64 = data.iter().sum();
    let sum_xy: f64 = data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..n).map(|i| (i as f64).powi(2)).sum();

    let denom = n_f * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return data.to_vec();
    }
    let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n_f;

    data.iter()
        .enumerate()
        .map(|(i, &y)| y - (slope * i as f64 + intercept))
        .collect()
}

/// Apply a moving-average smoother with the given window size.
pub fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.is_empty() {
        return data.to_vec();
    }
    let w = window.min(data.len());
    let mut result = Vec::with_capacity(data.len());
    let mut sum: f64 = data[..w].iter().sum();
    // center the window for first w/2 elements
    for i in 0..data.len() {
        let lo = i.saturating_sub(w / 2);
        let hi = (i + w / 2 + 1).min(data.len());
        let s: f64 = data[lo..hi].iter().sum();
        result.push(s / (hi - lo) as f64);
    }
    let _ = sum; // suppress unused warning
    result
}

/// Compute the autocorrelation function up to `max_lag` lags.
pub fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    if n < 2 {
        return vec![1.0];
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();
    if var.abs() < 1e-30 {
        return vec![1.0; max_lag + 1];
    }

    let lags = max_lag.min(n - 1);
    let mut acf = Vec::with_capacity(lags + 1);
    for lag in 0..=lags {
        let mut c = 0.0;
        for i in 0..n - lag {
            c += (data[i] - mean) * (data[i + lag] - mean);
        }
        acf.push(c / var);
    }
    acf
}

/// Compute the power spectral density via a simple periodogram (no FFT).
/// Returns (frequencies, power) vectors.
pub fn periodogram(data: &[f64], dt: f64) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();

    let n_freq = n / 2 + 1;
    let mut freqs = Vec::with_capacity(n_freq);
    let mut power = Vec::with_capacity(n_freq);

    for k in 0..n_freq {
        let freq = k as f64 / (n as f64 * dt);
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (j, &x) in centered.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        freqs.push(freq);
        power.push((re * re + im * im) / n as f64);
    }
    (freqs, power)
}

/// Apply a Hanning window to the data.
pub fn hanning_window(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }
    data.iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1).max(1) as f64).cos());
            x * w
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ViolationDetector – orchestration
// ---------------------------------------------------------------------------

/// The main violation detector that orchestrates the full detection pipeline.
pub struct ViolationDetector {
    pub config: DetectionConfig,
}

impl ViolationDetector {
    pub fn new(config: DetectionConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self {
            config: DetectionConfig::default(),
        }
    }

    /// Run the full detection pipeline on a conservation-quantity time series.
    /// `values` are the computed conservation quantity at each timestep.
    pub fn detect_violation(&self, values: &[f64]) -> DetectionResult {
        if values.len() < self.config.min_samples {
            return DetectionResult {
                detected: false,
                confidence: 0.0,
                p_value: None,
                violation_kind: None,
                violation_severity: None,
                details: format!(
                    "Insufficient samples: {} < {}",
                    values.len(),
                    self.config.min_samples
                ),
                change_point_index: None,
                test_statistic: None,
            };
        }

        let reference = values[0];
        let errors = compute_errors(values, reference);
        let max_abs_error = errors.iter().fold(0.0_f64, |acc, &e| acc.max(e.abs()));
        if self
            .config
            .tolerance
            .check(reference, reference + max_abs_error)
        {
            return DetectionResult::no_violation();
        }

        // Stage 1: basic statistical test (t-test for mean != 0)
        let t_result = self.run_ttest(&errors);
        if !t_result.detected {
            return t_result;
        }

        // Stage 2: drift detection
        let drift_result = if self.config.enable_drift_detection {
            self.run_drift_check(&errors)
        } else {
            DetectionResult::no_violation()
        };

        // Stage 3: anomaly check
        let anomaly_result = if self.config.enable_anomaly_detection {
            self.run_anomaly_check(&errors)
        } else {
            DetectionResult::no_violation()
        };

        // Stage 4: change-point detection
        let cp_result = if self.config.enable_change_point_detection {
            self.run_change_point(&errors)
        } else {
            DetectionResult::no_violation()
        };

        // Combine results: take the most severe
        self.combine_results(&[t_result, drift_result, anomaly_result, cp_result])
    }

    /// Run multi-law detection on a trajectory-like structure given per-law
    /// time-series. Each entry is (ConservationKind, values).
    pub fn detect_multi_law(
        &self,
        law_series: &[(ConservationKind, Vec<f64>)],
    ) -> MultiLawResult {
        let mut multi = MultiLawResult::new();
        for (kind, values) in law_series {
            let result = self.detect_violation(values).with_kind(*kind);
            multi.add(*kind, result);
        }
        multi
    }

    // ----- internal helpers -----

    fn run_ttest(&self, errors: &[f64]) -> DetectionResult {
        let test = crate::statistical::TTest::new(0.0);
        let result = test.test(errors);
        if result.reject {
            DetectionResult::violation(
                1.0 - result.p_value,
                result.p_value,
                &format!("T-test: mean error significantly != 0 (t={:.4}, p={:.4e})",
                    result.statistic, result.p_value),
            )
            .with_statistic(result.statistic)
        } else {
            DetectionResult::no_violation()
        }
    }

    fn run_drift_check(&self, errors: &[f64]) -> DetectionResult {
        let ph = crate::drift_detect::PageHinkley::new(0.005, 50.0, 30);
        let mut detector = ph;
        for (i, &e) in errors.iter().enumerate() {
            if let crate::drift_detect::DriftStatus::Drift = detector.update(e) {
                return DetectionResult::violation(
                    0.9,
                    0.01,
                    &format!("Page-Hinkley drift detected at sample {}", i),
                )
                .with_change_point(i);
            }
        }
        DetectionResult::no_violation()
    }

    fn run_anomaly_check(&self, errors: &[f64]) -> DetectionResult {
        let z = crate::anomaly::ZScoreAnomaly::new(3.0);
        let result = z.detect(errors);
        if result.is_anomaly {
            DetectionResult::violation(
                result.confidence,
                0.001,
                &format!("Z-score anomaly: {} anomalous points detected", result.anomaly_indices.len()),
            )
        } else {
            DetectionResult::no_violation()
        }
    }

    fn run_change_point(&self, errors: &[f64]) -> DetectionResult {
        let cusum = crate::cusum::Cusum::new(0.0, 5.0);
        let result = cusum.run(errors);
        if result.alarm {
            DetectionResult::violation(
                0.95,
                0.005,
                &format!("CUSUM alarm at sample {}", result.alarm_index.unwrap_or(0)),
            )
            .with_change_point(result.alarm_index.unwrap_or(0))
        } else {
            DetectionResult::no_violation()
        }
    }

    fn combine_results(&self, results: &[DetectionResult]) -> DetectionResult {
        let detected_results: Vec<&DetectionResult> =
            results.iter().filter(|r| r.detected).collect();

        if detected_results.is_empty() {
            return DetectionResult::no_violation();
        }

        // Pick the result with the lowest p-value (strongest evidence)
        let best = detected_results
            .iter()
            .min_by(|a, b| {
                let pa = a.p_value.unwrap_or(1.0);
                let pb = b.p_value.unwrap_or(1.0);
                pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let severity = self.classify_severity(best);
        let mut combined = (*best).clone();
        combined.violation_severity = Some(severity);
        combined.confidence = detected_results.iter().map(|r| r.confidence).fold(0.0_f64, f64::max);
        combined
    }

    fn classify_severity(&self, result: &DetectionResult) -> ViolationSeverity {
        match result.p_value {
            Some(p) if p < 0.001 => ViolationSeverity::Critical,
            Some(p) if p < 0.01 => ViolationSeverity::Error,
            Some(p) if p < 0.05 => ViolationSeverity::Warning,
            _ => ViolationSeverity::Info,
        }
    }
}

impl Detector for ViolationDetector {
    fn detect(&self, errors: &[f64]) -> DetectionResult {
        // errors are pre-computed, so use reference = 0
        if errors.len() < self.config.min_samples {
            return DetectionResult::no_violation();
        }
        let test = crate::statistical::TTest::new(0.0);
        let result = test.test(errors);
        if result.reject {
            DetectionResult::violation(
                1.0 - result.p_value,
                result.p_value,
                &format!("Mean deviation: t={:.4}, p={:.4e}", result.statistic, result.p_value),
            )
        } else {
            DetectionResult::no_violation()
        }
    }

    fn name(&self) -> &str {
        "ViolationDetector"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_good_data(n: usize) -> Vec<f64> {
        // Constant with tiny, zero-mean noise
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| 10.0 + (rng.gen::<f64>() - 0.5) * 1e-12)
            .collect()
    }

    fn make_bad_drift(n: usize) -> Vec<f64> {
        (0..n).map(|i| 10.0 + 0.01 * i as f64).collect()
    }

    fn make_bad_jump(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| if i < n / 2 { 10.0 } else { 10.5 })
            .collect()
    }

    #[test]
    fn test_no_violation_clean_data() {
        let det = ViolationDetector::with_defaults();
        let data = make_good_data(200);
        let result = det.detect_violation(&data);
        assert!(!result.detected, "Should not detect violation in clean data");
    }

    #[test]
    fn test_detect_drift() {
        let det = ViolationDetector::with_defaults();
        let data = make_bad_drift(200);
        let result = det.detect_violation(&data);
        assert!(result.detected, "Should detect linear drift");
    }

    #[test]
    fn test_detect_jump() {
        let det = ViolationDetector::with_defaults();
        let data = make_bad_jump(200);
        let result = det.detect_violation(&data);
        assert!(result.detected, "Should detect step jump");
    }

    #[test]
    fn test_multi_law_detection() {
        let det = ViolationDetector::with_defaults();
        let good = make_good_data(200);
        let bad = make_bad_drift(200);
        let series = vec![
            (ConservationKind::Energy, good),
            (ConservationKind::Momentum, bad),
        ];
        let result = det.detect_multi_law(&series);
        assert!(result.overall_detected);
        let violations = result.violations();
        assert!(!violations.is_empty());
    }

    #[test]
    fn test_insufficient_samples() {
        let det = ViolationDetector::with_defaults();
        let data = vec![1.0, 2.0, 3.0];
        let result = det.detect_violation(&data);
        assert!(!result.detected);
        assert!(result.details.contains("Insufficient"));
    }

    #[test]
    fn test_compute_errors() {
        let vals = vec![10.0, 10.1, 9.9, 10.2];
        let errs = compute_errors(&vals, 10.0);
        assert!((errs[0] - 0.0).abs() < 1e-12);
        assert!((errs[1] - 0.1).abs() < 1e-12);
        assert!((errs[2] - (-0.1)).abs() < 1e-12);
    }

    #[test]
    fn test_detrend() {
        let data: Vec<f64> = (0..100).map(|i| 5.0 + 0.1 * i as f64).collect();
        let detrended = detrend(&data);
        let mean = detrended.iter().sum::<f64>() / detrended.len() as f64;
        assert!(mean.abs() < 1e-10, "Detrended mean should be ~0");
    }

    #[test]
    fn test_autocorrelation_lag0() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let acf = autocorrelation(&data, 3);
        assert!((acf[0] - 1.0).abs() < 1e-12, "ACF at lag 0 should be 1");
    }

    #[test]
    fn test_moving_average() {
        let data = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        let smoothed = moving_average(&data, 3);
        assert!(smoothed[2] < 10.0, "Moving average should smooth the spike");
        assert!(smoothed[2] > 0.0);
    }

    #[test]
    fn test_hanning_window_endpoints() {
        let data = vec![1.0; 10];
        let windowed = hanning_window(&data);
        assert!(windowed[0].abs() < 0.01, "Hanning window should taper to ~0 at start");
        assert!(windowed[9].abs() < 0.01, "Hanning window should taper to ~0 at end");
        assert!(windowed[5] > 0.8, "Hanning window should be ~1 at center");
    }

    #[test]
    fn test_periodogram_dc() {
        let data = vec![5.0; 64];
        let (freqs, power) = periodogram(&data, 1.0);
        // DC component should be zero after mean subtraction
        assert!(power[0].abs() < 1e-10, "DC should be zero after mean subtraction");
    }

    #[test]
    fn test_severity_classification() {
        let det = ViolationDetector::with_defaults();
        let r1 = DetectionResult::violation(0.99, 0.0001, "test");
        assert_eq!(det.classify_severity(&r1), ViolationSeverity::Critical);

        let r2 = DetectionResult::violation(0.95, 0.005, "test");
        assert_eq!(det.classify_severity(&r2), ViolationSeverity::Error);

        let r3 = DetectionResult::violation(0.9, 0.03, "test");
        assert_eq!(det.classify_severity(&r3), ViolationSeverity::Warning);

        let r4 = DetectionResult::violation(0.5, 0.1, "test");
        assert_eq!(det.classify_severity(&r4), ViolationSeverity::Info);
    }

    #[test]
    fn test_detection_result_builder() {
        let r = DetectionResult::violation(0.95, 0.01, "test")
            .with_kind(ConservationKind::Energy)
            .with_severity(ViolationSeverity::Error)
            .with_change_point(42)
            .with_statistic(3.5);
        assert!(r.detected);
        assert_eq!(r.violation_kind, Some(ConservationKind::Energy));
        assert_eq!(r.change_point_index, Some(42));
    }

    #[test]
    fn test_multi_law_empty() {
        let det = ViolationDetector::with_defaults();
        let result = det.detect_multi_law(&[]);
        assert!(!result.overall_detected);
        assert!(result.violations().is_empty());
    }
}
