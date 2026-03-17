//! Noise models for conservation violation detection.

/// Noise model trait.
pub trait NoiseModel { fn estimate_variance(&self, data: &[f64]) -> f64; fn name(&self) -> &str; }

/// White noise (constant variance).
#[derive(Debug, Clone, Default)]
pub struct WhiteNoiseModel;
impl NoiseModel for WhiteNoiseModel {
    fn estimate_variance(&self, data: &[f64]) -> f64 {
        let n = data.len() as f64;
        if n < 2.0 { return 0.0; }
        let mean = data.iter().sum::<f64>() / n;
        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    }
    fn name(&self) -> &str { "WhiteNoise" }
}

/// Pink (1/f) noise.
#[derive(Debug, Clone, Default)]
pub struct PinkNoiseModel;
impl NoiseModel for PinkNoiseModel {
    fn estimate_variance(&self, data: &[f64]) -> f64 { WhiteNoiseModel.estimate_variance(data) }
    fn name(&self) -> &str { "PinkNoise" }
}

/// Brown (1/f²) noise.
#[derive(Debug, Clone, Default)]
pub struct BrownNoiseModel;
impl NoiseModel for BrownNoiseModel {
    fn estimate_variance(&self, data: &[f64]) -> f64 { WhiteNoiseModel.estimate_variance(data) }
    fn name(&self) -> &str { "BrownNoise" }
}

/// Noise estimation utilities.
#[derive(Debug, Clone, Default)]
pub struct NoiseEstimation;
impl NoiseEstimation {
    /// Estimate the noise level from differences of consecutive values.
    pub fn from_differences(data: &[f64]) -> f64 {
        if data.len() < 2 { return 0.0; }
        let diffs: Vec<f64> = data.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        let n = diffs.len() as f64;
        (diffs.iter().map(|d| d * d).sum::<f64>() / (2.0 * n)).sqrt()
    }
}
