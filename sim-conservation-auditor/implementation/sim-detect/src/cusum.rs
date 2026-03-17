//! CUSUM (Cumulative Sum) control charts.
use serde::{Serialize, Deserialize};

/// CUSUM result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CusumResult { pub alarm: bool, pub statistic: f64, pub alarm_time: Option<usize>, pub alarm_index: Option<usize> }

/// One-sided CUSUM detector.
#[derive(Debug, Clone)]
pub struct Cusum { pub threshold: f64, pub drift: f64, s_pos: f64, count: usize }
impl Cusum {
    pub fn new(drift: f64, threshold: f64) -> Self { Self { threshold, drift, s_pos: 0.0, count: 0 } }
    pub fn update(&mut self, value: f64) -> CusumResult {
        self.count += 1;
        self.s_pos = (self.s_pos + value - self.drift).max(0.0);
        CusumResult { alarm: self.s_pos > self.threshold, statistic: self.s_pos, alarm_time: if self.s_pos > self.threshold { Some(self.count) } else { None }, alarm_index: if self.s_pos > self.threshold { Some(self.count) } else { None } }
    }
    pub fn reset(&mut self) { self.s_pos = 0.0; self.count = 0; }
    /// Run CUSUM on an entire data series.
    pub fn run(&self, data: &[f64]) -> CusumResult {
        let mut s = 0.0_f64;
        for (i, &v) in data.iter().enumerate() {
            s = (s + v - self.drift).max(0.0);
            if s > self.threshold {
                return CusumResult { alarm: true, statistic: s, alarm_time: Some(i), alarm_index: Some(i) };
            }
        }
        CusumResult { alarm: false, statistic: s, alarm_time: None, alarm_index: None }
    }
}

/// Two-sided CUSUM detector.
#[derive(Debug, Clone)]
pub struct TwoSidedCusum { pub pos: Cusum, pub neg: Cusum }
impl TwoSidedCusum {
    pub fn new(threshold: f64, drift: f64) -> Self { Self { pos: Cusum::new(threshold, drift), neg: Cusum::new(threshold, drift) } }
}

/// V-mask CUSUM detector.
#[derive(Debug, Clone)]
pub struct VmaskCusum { pub lead: f64, pub slope: f64, values: Vec<f64> }
impl VmaskCusum {
    pub fn new(lead: f64, slope: f64) -> Self { Self { lead, slope, values: Vec::new() } }
}
