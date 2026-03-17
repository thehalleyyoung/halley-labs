use shared_types::ConfidenceInterval;

#[derive(Debug, Clone)]
pub struct BootstrapCI { pub n_resamples: usize, pub confidence_level: f64 }
impl BootstrapCI {
    pub fn new(n: usize, cl: f64) -> Self { Self { n_resamples: n, confidence_level: cl } }
    pub fn compute(&self, data: &[f64]) -> ConfidenceInterval {
        let n = data.len();
        if n == 0 { return ConfidenceInterval::new(0.0, 0.0, self.confidence_level); }
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let se = if n > 1 {
            (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt() / (n as f64).sqrt()
        } else { 0.0 };
        let z = 1.96;
        ConfidenceInterval::new(mean - z * se, mean + z * se, self.confidence_level)
    }
}
impl Default for BootstrapCI { fn default() -> Self { Self::new(1000, 0.95) } }

#[derive(Debug, Clone)]
pub struct ClopperPearsonCI { pub confidence_level: f64 }
impl ClopperPearsonCI { pub fn new(cl: f64) -> Self { Self { confidence_level: cl } } }
impl Default for ClopperPearsonCI { fn default() -> Self { Self::new(0.95) } }

#[derive(Debug, Clone)]
pub struct WilsonCI { pub confidence_level: f64 }
impl WilsonCI { pub fn new(cl: f64) -> Self { Self { confidence_level: cl } } }
impl Default for WilsonCI { fn default() -> Self { Self::new(0.95) } }
