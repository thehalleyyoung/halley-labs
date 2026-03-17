//! Objective functions for sonification optimization.
//!
//! Each objective wraps an evaluation of a mapping configuration and returns
//! a scalar value to maximize or minimize.

use std::collections::HashMap;

use crate::{
    MappingConfig, OptimizerError, OptimizerResult, StreamId,
};
use crate::mutual_information::{
    MutualInformationEstimator, PsychoacousticChannel, DiscriminabilityEstimator,
};

// ─────────────────────────────────────────────────────────────────────────────
// ObjectiveFn trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for objective functions.
pub trait ObjectiveFn: Send + Sync {
    /// Evaluate the objective for a given configuration.
    /// Returns a value where **higher is better** (even if we're minimizing,
    /// the implementation negates internally).
    fn evaluate(&self, config: &MappingConfig) -> OptimizerResult<f64>;

    /// Name of this objective.
    fn name(&self) -> &str;

    /// Whether this objective should be maximized (true) or minimized (false).
    fn is_maximizing(&self) -> bool {
        true
    }

    /// Compute finite-difference gradient w.r.t. a named parameter.
    fn gradient(
        &self,
        config: &MappingConfig,
        param_name: &str,
        epsilon: f64,
    ) -> OptimizerResult<f64> {
        let mut config_plus = config.clone();
        let mut config_minus = config.clone();

        let current = config.global_params.get(param_name).copied().unwrap_or(0.0);
        config_plus.global_params.insert(param_name.to_string(), current + epsilon);
        config_minus.global_params.insert(param_name.to_string(), current - epsilon);

        let f_plus = self.evaluate(&config_plus)?;
        let f_minus = self.evaluate(&config_minus)?;

        Ok((f_plus - f_minus) / (2.0 * epsilon))
    }

    /// Compute full gradient vector for all global parameters.
    fn gradient_vector(
        &self,
        config: &MappingConfig,
        epsilon: f64,
    ) -> OptimizerResult<HashMap<String, f64>> {
        let params: Vec<String> = config.global_params.keys().cloned().collect();
        let mut gradients = HashMap::new();

        for param in &params {
            let g = self.gradient(config, param, epsilon)?;
            gradients.insert(param.clone(), g);
        }

        Ok(gradients)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MutualInformationObjective
// ─────────────────────────────────────────────────────────────────────────────

/// Maximize I_ψ(D; A) — psychoacoustically-constrained mutual information.
#[derive(Clone)]
pub struct MutualInformationObjective {
    pub estimator: MutualInformationEstimator,
    pub channel: PsychoacousticChannel,
    pub reference_joint: Vec<Vec<f64>>,
}

impl MutualInformationObjective {
    pub fn new(
        estimator: MutualInformationEstimator,
        channel: PsychoacousticChannel,
        reference_joint: Vec<Vec<f64>>,
    ) -> Self {
        MutualInformationObjective {
            estimator,
            channel,
            reference_joint,
        }
    }

    /// Build a simple reference joint distribution from a config.
    pub fn from_config(config: &MappingConfig, bins: usize) -> Self {
        let n = config.stream_count().max(2);
        let mut joint = vec![vec![0.0; bins]; bins];
        // Diagonal-ish distribution: higher MI when streams are well-separated
        for i in 0..bins {
            for j in 0..bins {
                let dist = ((i as f64 - j as f64) / bins as f64).abs();
                joint[i][j] = (-dist * n as f64).exp();
            }
        }
        MutualInformationObjective {
            estimator: MutualInformationEstimator::new(bins, 2.0),
            channel: PsychoacousticChannel::default(),
            reference_joint: joint,
        }
    }
}

impl ObjectiveFn for MutualInformationObjective {
    fn evaluate(&self, config: &MappingConfig) -> OptimizerResult<f64> {
        // Modulate the reference joint by the config's stream separation
        let mut modulated = self.reference_joint.clone();
        let streams: Vec<_> = config.stream_params.values().collect();

        if streams.len() >= 2 {
            // Better separation => more diagonal joint => higher MI
            let mut avg_separation = 0.0;
            let mut count = 0;
            for i in 0..streams.len() {
                for j in (i + 1)..streams.len() {
                    let freq_sep = (streams[i].frequency_hz - streams[j].frequency_hz).abs();
                    avg_separation += freq_sep;
                    count += 1;
                }
            }
            if count > 0 {
                avg_separation /= count as f64;
            }

            let sharpness = (avg_separation / 1000.0).min(5.0);
            let bins = modulated.len();
            for i in 0..bins {
                for j in 0..bins {
                    let dist = ((i as f64 - j as f64) / bins as f64).abs();
                    modulated[i][j] = (-dist * sharpness).exp();
                }
            }
        }

        self.estimator
            .psychoacoustic_mutual_information(&modulated, &self.channel)
    }

    fn name(&self) -> &str {
        "mutual_information"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DiscriminabilityObjective
// ─────────────────────────────────────────────────────────────────────────────

/// Maximize d'_model — average pairwise discriminability.
#[derive(Debug, Clone)]
pub struct DiscriminabilityObjective {
    pub estimator: DiscriminabilityEstimator,
}

impl Default for DiscriminabilityObjective {
    fn default() -> Self {
        DiscriminabilityObjective {
            estimator: DiscriminabilityEstimator::default(),
        }
    }
}

impl DiscriminabilityObjective {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ObjectiveFn for DiscriminabilityObjective {
    fn evaluate(&self, config: &MappingConfig) -> OptimizerResult<f64> {
        Ok(self.estimator.d_prime_model(config))
    }

    fn name(&self) -> &str {
        "discriminability"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LatencyObjective
// ─────────────────────────────────────────────────────────────────────────────

/// Minimize rendering latency (returned as negative for maximization framework).
#[derive(Debug, Clone)]
pub struct LatencyObjective {
    /// Base processing latency in ms.
    pub base_latency_ms: f64,
    /// Per-stream latency contribution in ms.
    pub per_stream_ms: f64,
    /// Maximum acceptable latency for normalization.
    pub max_latency_ms: f64,
}

impl Default for LatencyObjective {
    fn default() -> Self {
        LatencyObjective {
            base_latency_ms: 5.0,
            per_stream_ms: 2.0,
            max_latency_ms: 100.0,
        }
    }
}

impl LatencyObjective {
    pub fn new(base: f64, per_stream: f64, max: f64) -> Self {
        LatencyObjective {
            base_latency_ms: base,
            per_stream_ms: per_stream,
            max_latency_ms: max,
        }
    }

    pub fn estimate_latency(&self, config: &MappingConfig) -> f64 {
        self.base_latency_ms + self.per_stream_ms * config.stream_count() as f64
    }
}

impl ObjectiveFn for LatencyObjective {
    fn evaluate(&self, config: &MappingConfig) -> OptimizerResult<f64> {
        let latency = self.estimate_latency(config);
        // Return as 1 - (latency / max) so higher is better
        Ok((1.0 - latency / self.max_latency_ms).max(0.0))
    }

    fn name(&self) -> &str {
        "latency"
    }

    fn is_maximizing(&self) -> bool {
        true // We've already negated internally
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CognitiveLoadObjective
// ─────────────────────────────────────────────────────────────────────────────

/// Minimize cognitive load (lower is better → returned as negative utility).
#[derive(Debug, Clone)]
pub struct CognitiveLoadObjective {
    /// Maximum cognitive load for normalization.
    pub max_load: f64,
    /// Base load.
    pub base_load: f64,
    /// Per-stream load increment.
    pub per_stream_load: f64,
    /// Interaction factor: load from stream pairs.
    pub interaction_factor: f64,
}

impl Default for CognitiveLoadObjective {
    fn default() -> Self {
        CognitiveLoadObjective {
            max_load: 30.0,
            base_load: 1.0,
            per_stream_load: 1.5,
            interaction_factor: 0.2,
        }
    }
}

impl CognitiveLoadObjective {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn estimate_load(&self, config: &MappingConfig) -> f64 {
        let n = config.stream_count() as f64;
        self.base_load + self.per_stream_load * n + self.interaction_factor * n * (n - 1.0) / 2.0
    }
}

impl ObjectiveFn for CognitiveLoadObjective {
    fn evaluate(&self, config: &MappingConfig) -> OptimizerResult<f64> {
        let load = self.estimate_load(config);
        Ok((1.0 - load / self.max_load).max(0.0))
    }

    fn name(&self) -> &str {
        "cognitive_load"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpectralClarityObjective
// ─────────────────────────────────────────────────────────────────────────────

/// Maximize masking margins across all stream pairs.
#[derive(Debug, Clone)]
pub struct SpectralClarityObjective {
    /// Target masking margin in dB.
    pub target_margin_db: f64,
}

impl Default for SpectralClarityObjective {
    fn default() -> Self {
        SpectralClarityObjective {
            target_margin_db: 15.0,
        }
    }
}

impl SpectralClarityObjective {
    pub fn new(target: f64) -> Self {
        SpectralClarityObjective {
            target_margin_db: target,
        }
    }
}

impl ObjectiveFn for SpectralClarityObjective {
    fn evaluate(&self, config: &MappingConfig) -> OptimizerResult<f64> {
        let streams: Vec<_> = config.stream_params.values().collect();
        if streams.len() < 2 {
            return Ok(1.0);
        }

        let mut min_margin = f64::INFINITY;
        for i in 0..streams.len() {
            for j in (i + 1)..streams.len() {
                let freq_diff = (streams[i].frequency_hz - streams[j].frequency_hz).abs();
                let amp_diff = (streams[i].amplitude_db - streams[j].amplitude_db).abs();

                // Simplified masking model: margin depends on frequency and amplitude separation
                let freq_factor = (freq_diff / 100.0).min(1.0);
                let margin = amp_diff * freq_factor;
                min_margin = min_margin.min(margin);
            }
        }

        if min_margin.is_infinite() {
            Ok(1.0)
        } else {
            Ok((min_margin / self.target_margin_db).min(1.0))
        }
    }

    fn name(&self) -> &str {
        "spectral_clarity"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompositeObjective
// ─────────────────────────────────────────────────────────────────────────────

/// Weighted combination of multiple objectives.
pub struct CompositeObjective {
    pub objectives: Vec<(f64, Box<dyn ObjectiveFn>)>,
    pub name: String,
}

impl CompositeObjective {
    pub fn new() -> Self {
        CompositeObjective {
            objectives: Vec::new(),
            name: "composite".to_string(),
        }
    }

    pub fn add(&mut self, weight: f64, objective: Box<dyn ObjectiveFn>) {
        self.objectives.push((weight, objective));
    }

    pub fn with_objective(mut self, weight: f64, objective: Box<dyn ObjectiveFn>) -> Self {
        self.objectives.push((weight, objective));
        self
    }

    /// Evaluate each sub-objective separately.
    pub fn evaluate_components(
        &self,
        config: &MappingConfig,
    ) -> OptimizerResult<Vec<(String, f64, f64)>> {
        let mut components = Vec::new();
        for (weight, obj) in &self.objectives {
            let value = obj.evaluate(config)?;
            components.push((obj.name().to_string(), *weight, value));
        }
        Ok(components)
    }

    /// Normalize weights to sum to 1.
    pub fn normalize_weights(&mut self) {
        let total: f64 = self.objectives.iter().map(|(w, _)| w).sum();
        if total > 0.0 {
            for (w, _) in &mut self.objectives {
                *w /= total;
            }
        }
    }
}

impl Default for CompositeObjective {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectiveFn for CompositeObjective {
    fn evaluate(&self, config: &MappingConfig) -> OptimizerResult<f64> {
        let mut total = 0.0;
        for (weight, objective) in &self.objectives {
            let value = objective.evaluate(config)?;
            total += weight * value;
        }
        Ok(total)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Build a standard composite objective with default weights.
pub fn default_composite_objective() -> CompositeObjective {
    let mi = MutualInformationObjective::from_config(&MappingConfig::new(), 16);
    let disc = DiscriminabilityObjective::new();
    let lat = LatencyObjective::default();
    let cog = CognitiveLoadObjective::new();
    let spec = SpectralClarityObjective::default();

    CompositeObjective::new()
        .with_objective(0.4, Box::new(mi))
        .with_objective(0.2, Box::new(disc))
        .with_objective(0.15, Box::new(lat))
        .with_objective(0.1, Box::new(cog))
        .with_objective(0.15, Box::new(spec))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StreamMapping, StreamId};

    fn make_config(streams: Vec<(u32, f64, f64)>) -> MappingConfig {
        let mut config = MappingConfig::new();
        for (id, freq, amp) in streams {
            config.stream_params.insert(
                StreamId(id),
                StreamMapping::new(StreamId(id), freq, amp),
            );
        }
        config
    }

    #[test]
    fn test_discriminability_objective() {
        let obj = DiscriminabilityObjective::new();
        let config = make_config(vec![(0, 440.0, 60.0), (1, 880.0, 55.0)]);
        let val = obj.evaluate(&config).unwrap();
        assert!(val > 0.0, "d' should be positive for separated streams");
    }

    #[test]
    fn test_latency_objective() {
        let obj = LatencyObjective::default();
        let config = make_config(vec![(0, 440.0, 60.0)]);
        let val = obj.evaluate(&config).unwrap();
        assert!(val > 0.0 && val <= 1.0, "Latency objective in [0,1], got {}", val);
    }

    #[test]
    fn test_latency_increases_with_streams() {
        let obj = LatencyObjective::default();
        let c1 = make_config(vec![(0, 440.0, 60.0)]);
        let c2 = make_config(vec![(0, 440.0, 60.0), (1, 880.0, 55.0), (2, 1320.0, 50.0)]);
        let v1 = obj.evaluate(&c1).unwrap();
        let v2 = obj.evaluate(&c2).unwrap();
        assert!(v1 > v2, "More streams => worse latency: {} vs {}", v1, v2);
    }

    #[test]
    fn test_cognitive_load_objective() {
        let obj = CognitiveLoadObjective::new();
        let config = make_config(vec![(0, 440.0, 60.0), (1, 880.0, 55.0)]);
        let val = obj.evaluate(&config).unwrap();
        assert!(val > 0.0 && val <= 1.0, "Cognitive load in [0,1], got {}", val);
    }

    #[test]
    fn test_cognitive_load_increases() {
        let obj = CognitiveLoadObjective::new();
        let c1 = make_config(vec![(0, 440.0, 60.0)]);
        let many: Vec<_> = (0..8).map(|i| (i, 200.0 + i as f64 * 100.0, 60.0)).collect();
        let c2 = make_config(many);
        let v1 = obj.evaluate(&c1).unwrap();
        let v2 = obj.evaluate(&c2).unwrap();
        assert!(v1 > v2, "More streams => higher load => lower value");
    }

    #[test]
    fn test_spectral_clarity_well_separated() {
        let obj = SpectralClarityObjective::new(15.0);
        let config = make_config(vec![(0, 440.0, 60.0), (1, 2000.0, 70.0)]);
        let val = obj.evaluate(&config).unwrap();
        assert!(val > 0.0);
    }

    #[test]
    fn test_spectral_clarity_single_stream() {
        let obj = SpectralClarityObjective::new(15.0);
        let config = make_config(vec![(0, 440.0, 60.0)]);
        let val = obj.evaluate(&config).unwrap();
        assert!((val - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_composite_objective() {
        let mut comp = CompositeObjective::new();
        comp.add(0.5, Box::new(DiscriminabilityObjective::new()));
        comp.add(0.5, Box::new(LatencyObjective::default()));

        let config = make_config(vec![(0, 440.0, 60.0), (1, 880.0, 55.0)]);
        let val = comp.evaluate(&config).unwrap();
        assert!(val > 0.0);
    }

    #[test]
    fn test_composite_evaluate_components() {
        let mut comp = CompositeObjective::new();
        comp.add(0.5, Box::new(DiscriminabilityObjective::new()));
        comp.add(0.5, Box::new(LatencyObjective::default()));

        let config = make_config(vec![(0, 440.0, 60.0)]);
        let components = comp.evaluate_components(&config).unwrap();
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_normalize_weights() {
        let mut comp = CompositeObjective::new();
        comp.add(2.0, Box::new(DiscriminabilityObjective::new()));
        comp.add(3.0, Box::new(LatencyObjective::default()));
        comp.normalize_weights();

        let total: f64 = comp.objectives.iter().map(|(w, _)| w).sum();
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_gradient_estimation() {
        let obj = LatencyObjective::default();
        let mut config = MappingConfig::new();
        config.global_params.insert("dummy".into(), 5.0);
        // Gradient should be computable (even if ~0 since latency doesn't depend on global params)
        let g = obj.gradient(&config, "dummy", 0.001).unwrap();
        assert!(g.is_finite());
    }

    #[test]
    fn test_mi_objective_from_config() {
        let config = make_config(vec![(0, 440.0, 60.0), (1, 880.0, 55.0)]);
        let obj = MutualInformationObjective::from_config(&config, 8);
        let val = obj.evaluate(&config).unwrap();
        assert!(val >= 0.0, "MI should be non-negative, got {}", val);
    }

    #[test]
    fn test_default_composite() {
        let comp = default_composite_objective();
        assert_eq!(comp.objectives.len(), 5);
        let config = make_config(vec![(0, 440.0, 60.0)]);
        let val = comp.evaluate(&config).unwrap();
        assert!(val.is_finite());
    }
}
