//! Statistical utility types used across the workspace.

use serde::{Deserialize, Serialize};

/// A confidence interval with a stated confidence level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64,
}

impl ConfidenceInterval {
    pub fn new(lower: f64, upper: f64, confidence_level: f64) -> Self {
        Self {
            lower,
            upper,
            confidence_level,
        }
    }

    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    pub fn midpoint(&self) -> f64 {
        (self.lower + self.upper) / 2.0
    }
}

/// Welford's online algorithm for computing running mean and variance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunningStats {
    pub count: u64,
    pub mean: f64,
    m2: f64,
}

impl RunningStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Summary statistics for a sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStats {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub count: usize,
    pub q1: f64,
    pub q3: f64,
}

/// A simple histogram with uniform bins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    pub bins: Vec<(f64, f64)>,
    pub counts: Vec<usize>,
    pub total: usize,
}
