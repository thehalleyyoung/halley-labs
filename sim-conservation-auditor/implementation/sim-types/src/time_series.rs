use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub times: Vec<f64>,
    pub values: Vec<f64>,
}

impl TimeSeries {
    pub fn new(times: Vec<f64>, values: Vec<f64>) -> Self {
        assert_eq!(times.len(), values.len(), "times and values must have the same length");
        Self { times, values }
    }

    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    pub fn variance(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let n = self.values.len() as f64;
        self.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn min(&self) -> f64 {
        self.values.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn max(&self) -> f64 {
        self.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn from_fn(start: f64, end: f64, n: usize, f: impl Fn(f64) -> f64) -> Self {
        let dt = (end - start) / (n - 1).max(1) as f64;
        let times: Vec<f64> = (0..n).map(|i| start + i as f64 * dt).collect();
        let values: Vec<f64> = times.iter().map(|&t| f(t)).collect();
        Self { times, values }
    }
}
