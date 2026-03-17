//! Online drift detection algorithms.
use serde::{Serialize, Deserialize};

/// Status of drift detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DriftStatus { NoDrift, Warning, Drift }

/// Drift detector trait.
pub trait DriftDetector {
    fn update(&mut self, value: f64) -> DriftStatus;
    fn reset(&mut self);
    fn name(&self) -> &str;
}

/// Page-Hinkley drift detector.
#[derive(Debug, Clone)]
pub struct PageHinkley { sum: f64, min_sum: f64, count: usize, mean: f64, threshold: f64, delta: f64, min_instances: usize }
impl PageHinkley { pub fn new(delta: f64, threshold: f64, min_instances: usize) -> Self { Self { sum: 0.0, min_sum: 0.0, count: 0, mean: 0.0, threshold, delta, min_instances } } }
impl DriftDetector for PageHinkley {
    fn update(&mut self, value: f64) -> DriftStatus {
        self.count += 1;
        self.mean += (value - self.mean) / self.count as f64;
        self.sum += value - self.mean - self.delta;
        if self.sum < self.min_sum { self.min_sum = self.sum; }
        if self.sum - self.min_sum > self.threshold { DriftStatus::Drift } else { DriftStatus::NoDrift }
    }
    fn reset(&mut self) { self.sum = 0.0; self.min_sum = 0.0; self.count = 0; self.mean = 0.0; }
    fn name(&self) -> &str { "PageHinkley" }
}

/// ADWIN adaptive windowing drift detector.
#[derive(Debug, Clone, Default)]
pub struct Adwin { values: Vec<f64>, threshold: f64 }
impl Adwin { pub fn new(threshold: f64) -> Self { Self { values: Vec::new(), threshold } } }
impl DriftDetector for Adwin {
    fn update(&mut self, value: f64) -> DriftStatus { self.values.push(value); DriftStatus::NoDrift }
    fn reset(&mut self) { self.values.clear(); }
    fn name(&self) -> &str { "ADWIN" }
}

/// DDM (Drift Detection Method).
#[derive(Debug, Clone, Default)]
pub struct Ddm { count: usize, sum: f64, min_rate: f64 }
impl DriftDetector for Ddm {
    fn update(&mut self, value: f64) -> DriftStatus { self.count += 1; self.sum += value; DriftStatus::NoDrift }
    fn reset(&mut self) { self.count = 0; self.sum = 0.0; self.min_rate = f64::MAX; }
    fn name(&self) -> &str { "DDM" }
}

/// EDDM (Early DDM).
#[derive(Debug, Clone, Default)]
pub struct Eddm { count: usize }
impl DriftDetector for Eddm {
    fn update(&mut self, _value: f64) -> DriftStatus { self.count += 1; DriftStatus::NoDrift }
    fn reset(&mut self) { self.count = 0; }
    fn name(&self) -> &str { "EDDM" }
}
