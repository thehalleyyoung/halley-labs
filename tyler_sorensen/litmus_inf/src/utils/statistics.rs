//! Basic statistics utilities for LITMUS∞.
//!
//! Provides Stats accumulator, Histogram, and Timer types.

use std::fmt;
use std::time::Instant;

/// Running statistics accumulator (mean, variance, min, max).
#[derive(Debug, Clone)]
pub struct Stats {
    pub count: usize,
    pub sum: f64,
    pub sum_sq: f64,
    pub min: f64,
    pub max: f64,
}

impl Stats {
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    pub fn add(&mut self, val: f64) {
        self.count += 1;
        self.sum += val;
        self.sum_sq += val * val;
        if val < self.min { self.min = val; }
        if val > self.max { self.max = val; }
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum / self.count as f64 }
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        let m = self.mean();
        self.sum_sq / self.count as f64 - m * m
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Default for Stats {
    fn default() -> Self { Self::new() }
}

/// Simple histogram with fixed bin count.
#[derive(Debug, Clone)]
pub struct Histogram {
    pub bins: Vec<usize>,
    pub min_val: f64,
    pub max_val: f64,
}

impl Histogram {
    pub fn new(min_val: f64, max_val: f64, num_bins: usize) -> Self {
        Self {
            bins: vec![0; num_bins],
            min_val,
            max_val,
        }
    }

    pub fn add(&mut self, val: f64) {
        let range = self.max_val - self.min_val;
        if range <= 0.0 { return; }
        let idx = ((val - self.min_val) / range * self.bins.len() as f64) as usize;
        let idx = idx.min(self.bins.len() - 1);
        self.bins[idx] += 1;
    }

    pub fn total(&self) -> usize {
        self.bins.iter().sum()
    }
}

/// A simple timer for measuring elapsed time.
#[derive(Debug, Clone)]
pub struct Timer {
    start: Instant,
    label: String,
}

impl Timer {
    pub fn new(label: &str) -> Self {
        Self {
            start: Instant::now(),
            label: label.to_string(),
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

impl fmt::Display for Timer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:.3}s", self.label, self.elapsed_secs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats() {
        let mut s = Stats::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            s.add(v);
        }
        assert_eq!(s.count, 5);
        assert!((s.mean() - 3.0).abs() < 1e-10);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);
    }

    #[test]
    fn test_histogram() {
        let mut h = Histogram::new(0.0, 10.0, 5);
        h.add(1.0);
        h.add(3.0);
        h.add(7.0);
        assert_eq!(h.total(), 3);
    }
}
