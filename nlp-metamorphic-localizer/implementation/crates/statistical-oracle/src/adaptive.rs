use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaDistribution { pub alpha: f64, pub beta: f64 }
impl BetaDistribution {
    pub fn new(alpha: f64, beta: f64) -> Self { Self { alpha, beta } }
    pub fn mean(&self) -> f64 { self.alpha / (self.alpha + self.beta) }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveResult { pub stage_name: String, pub score: f64, pub posterior: BetaDistribution }

pub struct AdaptiveLocator { pub prior_alpha: f64, pub prior_beta: f64 }
impl AdaptiveLocator {
    pub fn new(alpha: f64, beta: f64) -> Self { Self { prior_alpha: alpha, prior_beta: beta } }
}
impl Default for AdaptiveLocator { fn default() -> Self { Self::new(1.0, 1.0) } }
