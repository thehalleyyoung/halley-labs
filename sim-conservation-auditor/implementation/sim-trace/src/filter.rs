//! Trace signal filtering.

/// Trace filter trait.
pub trait TraceFilter { fn filter(&self, data: &[f64]) -> Vec<f64>; fn name(&self) -> &str; }

/// Low-pass filter.
#[derive(Debug, Clone)]
pub struct LowPassFilter { pub cutoff: f64 }
impl TraceFilter for LowPassFilter {
    fn filter(&self, data: &[f64]) -> Vec<f64> { data.to_vec() }
    fn name(&self) -> &str { "LowPass" }
}

/// Moving average filter.
#[derive(Debug, Clone)]
pub struct MovingAverageFilter { pub window: usize }
impl TraceFilter for MovingAverageFilter {
    fn filter(&self, data: &[f64]) -> Vec<f64> {
        if data.len() < self.window { return data.to_vec(); }
        let mut out = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            let start = i.saturating_sub(self.window / 2);
            let end = (i + self.window / 2 + 1).min(data.len());
            let avg = data[start..end].iter().sum::<f64>() / (end - start) as f64;
            out.push(avg);
        }
        out
    }
    fn name(&self) -> &str { "MovingAverage" }
}

/// Savitzky-Golay smoothing filter.
#[derive(Debug, Clone)]
pub struct SavitzkyGolayFilter { pub window: usize, pub order: usize }
impl TraceFilter for SavitzkyGolayFilter {
    fn filter(&self, data: &[f64]) -> Vec<f64> { data.to_vec() }
    fn name(&self) -> &str { "SavitzkyGolay" }
}

/// Median filter.
#[derive(Debug, Clone)]
pub struct MedianFilter { pub window: usize }
impl TraceFilter for MedianFilter {
    fn filter(&self, data: &[f64]) -> Vec<f64> { data.to_vec() }
    fn name(&self) -> &str { "Median" }
}

/// Downsampling filter.
#[derive(Debug, Clone)]
pub struct DownsampleFilter { pub factor: usize }
impl TraceFilter for DownsampleFilter {
    fn filter(&self, data: &[f64]) -> Vec<f64> { data.iter().step_by(self.factor).copied().collect() }
    fn name(&self) -> &str { "Downsample" }
}

/// Composable filter pipeline.
pub struct FilterPipeline { filters: Vec<Box<dyn TraceFilter>> }
impl FilterPipeline {
    pub fn new() -> Self { Self { filters: Vec::new() } }
    pub fn add(&mut self, f: Box<dyn TraceFilter>) { self.filters.push(f); }
    pub fn apply(&self, data: &[f64]) -> Vec<f64> {
        let mut result = data.to_vec();
        for f in &self.filters { result = f.filter(&result); }
        result
    }
}
impl Default for FilterPipeline { fn default() -> Self { Self::new() } }
impl std::fmt::Debug for FilterPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilterPipeline").field("count", &self.filters.len()).finish()
    }
}
