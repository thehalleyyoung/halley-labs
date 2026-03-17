use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearInterpolator {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
}

impl LinearInterpolator {
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> Self {
        Self { xs, ys }
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        if self.xs.is_empty() { return 0.0; }
        if x <= self.xs[0] { return self.ys[0]; }
        if x >= *self.xs.last().unwrap() { return *self.ys.last().unwrap(); }
        let i = self.xs.partition_point(|&xi| xi < x).saturating_sub(1);
        let t = (x - self.xs[i]) / (self.xs[i + 1] - self.xs[i]);
        self.ys[i] * (1.0 - t) + self.ys[i + 1] * t
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubicInterpolator {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
}

impl CubicInterpolator {
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> Self {
        Self { xs, ys }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HermiteInterpolator {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub dys: Vec<f64>,
}

impl HermiteInterpolator {
    pub fn new(xs: Vec<f64>, ys: Vec<f64>, dys: Vec<f64>) -> Self {
        Self { xs, ys, dys }
    }
}
