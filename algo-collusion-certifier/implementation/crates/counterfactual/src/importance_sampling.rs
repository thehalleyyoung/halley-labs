//! Importance sampling for efficient counterfactual estimation.
//!
//! Concentrates sampling effort around likely punishment responses and
//! profitable deviations. Provides self-normalized estimators and
//! effective sample size monitoring.

use shared_types::{
    CollusionError, CollusionResult, ConfidenceInterval, Price, Profit,
};
use serde::{Deserialize, Serialize};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::f64::consts::PI;

// ── ProposalDistribution ────────────────────────────────────────────────────

/// Importance sampling proposal distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProposalDistribution {
    /// Uniform over [lo, hi].
    Uniform { lo: f64, hi: f64 },
    /// Gaussian centered on mean with given std.
    Gaussian { mean: f64, std: f64 },
    /// Mixture of Gaussians for multi-modal targets.
    GaussianMixture {
        means: Vec<f64>,
        stds: Vec<f64>,
        weights: Vec<f64>,
    },
    /// Concentrated around a focal price (e.g., Nash deviation).
    FocalPoint {
        center: f64,
        scale: f64,
        tail_weight: f64,
    },
}

impl ProposalDistribution {
    /// Sample from the proposal distribution.
    pub fn sample(&self, rng: &mut StdRng) -> f64 {
        match self {
            ProposalDistribution::Uniform { lo, hi } => {
                rng.gen_range(*lo..*hi)
            }
            ProposalDistribution::Gaussian { mean, std } => {
                let u1: f64 = rng.gen::<f64>().max(1e-15);
                let u2: f64 = rng.gen::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                mean + std * z
            }
            ProposalDistribution::GaussianMixture { means, stds, weights } => {
                let u: f64 = rng.gen();
                let mut cum = 0.0;
                let mut component = 0;
                for (i, w) in weights.iter().enumerate() {
                    cum += w;
                    if u < cum {
                        component = i;
                        break;
                    }
                    component = i;
                }
                let u1: f64 = rng.gen::<f64>().max(1e-15);
                let u2: f64 = rng.gen::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                means[component] + stds[component] * z
            }
            ProposalDistribution::FocalPoint { center, scale, tail_weight } => {
                let u: f64 = rng.gen();
                if u < *tail_weight {
                    // Heavy tail: uniform over wider range
                    let range = scale * 5.0;
                    rng.gen_range((center - range)..(center + range))
                } else {
                    // Concentrated: Gaussian near center
                    let u1: f64 = rng.gen::<f64>().max(1e-15);
                    let u2: f64 = rng.gen::<f64>();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                    center + scale * z
                }
            }
        }
    }

    /// Evaluate the proposal density at a point.
    pub fn density(&self, x: f64) -> f64 {
        match self {
            ProposalDistribution::Uniform { lo, hi } => {
                if x >= *lo && x <= *hi {
                    1.0 / (hi - lo)
                } else {
                    0.0
                }
            }
            ProposalDistribution::Gaussian { mean, std } => {
                gaussian_pdf(x, *mean, *std)
            }
            ProposalDistribution::GaussianMixture { means, stds, weights } => {
                means.iter().zip(stds.iter()).zip(weights.iter())
                    .map(|((m, s), w)| w * gaussian_pdf(x, *m, *s))
                    .sum()
            }
            ProposalDistribution::FocalPoint { center, scale, tail_weight } => {
                let range = scale * 5.0;
                let uniform_part = if x >= center - range && x <= center + range {
                    tail_weight / (2.0 * range)
                } else {
                    0.0
                };
                let gaussian_part = (1.0 - tail_weight) * gaussian_pdf(x, *center, *scale);
                uniform_part + gaussian_part
            }
        }
    }

    /// Create a proposal centered around Nash/punishment response prices.
    pub fn for_punishment(nash_price: f64, observed_price: f64, price_range: f64) -> Self {
        let center = (nash_price + observed_price) / 2.0;
        ProposalDistribution::GaussianMixture {
            means: vec![nash_price, observed_price, center],
            stds: vec![price_range * 0.1, price_range * 0.1, price_range * 0.2],
            weights: vec![0.4, 0.3, 0.3],
        }
    }
}

fn gaussian_pdf(x: f64, mean: f64, std: f64) -> f64 {
    if std <= 0.0 { return 0.0; }
    let z = (x - mean) / std;
    (-0.5 * z * z).exp() / (std * (2.0 * PI).sqrt())
}

// ── ImportanceWeight ────────────────────────────────────────────────────────

/// A single importance weight w_i = p(x_i) / q(x_i).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ImportanceWeight {
    pub weight: f64,
    pub log_weight: f64,
    pub sample_value: f64,
    pub target_density: f64,
    pub proposal_density: f64,
}

impl ImportanceWeight {
    pub fn compute(sample_value: f64, target_density: f64, proposal_density: f64) -> Self {
        let w = if proposal_density > 1e-300 {
            target_density / proposal_density
        } else {
            0.0
        };
        Self {
            weight: w,
            log_weight: w.max(1e-300).ln(),
            sample_value,
            target_density,
            proposal_density,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.weight.is_finite() && self.weight >= 0.0
    }
}

// ── EffectiveSampleSize ─────────────────────────────────────────────────────

/// Monitor importance sampling degeneracy via effective sample size.
///
/// ESS = (Σ w_i)² / Σ w_i²
/// When ESS << n, the IS estimate is dominated by a few large weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveSampleSize {
    pub ess: f64,
    pub nominal_size: usize,
    pub ess_ratio: f64,
    pub max_weight: f64,
    pub is_degenerate: bool,
}

impl EffectiveSampleSize {
    /// Compute ESS from a set of importance weights.
    pub fn compute(weights: &[f64]) -> Self {
        let n = weights.len();
        if n == 0 {
            return Self {
                ess: 0.0,
                nominal_size: 0,
                ess_ratio: 0.0,
                max_weight: 0.0,
                is_degenerate: true,
            };
        }

        let sum_w: f64 = weights.iter().sum();
        let sum_w2: f64 = weights.iter().map(|w| w * w).sum();
        let max_w = weights.iter().copied().fold(0.0f64, f64::max);

        let ess = if sum_w2 > 1e-300 {
            (sum_w * sum_w) / sum_w2
        } else {
            0.0
        };

        let ratio = ess / n as f64;

        Self {
            ess,
            nominal_size: n,
            ess_ratio: ratio,
            max_weight: max_w,
            is_degenerate: ratio < 0.1,
        }
    }

    /// Compute from log weights for numerical stability.
    pub fn from_log_weights(log_weights: &[f64]) -> Self {
        if log_weights.is_empty() {
            return Self::compute(&[]);
        }

        let max_log = log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let shifted: Vec<f64> = log_weights.iter().map(|lw| (lw - max_log).exp()).collect();
        Self::compute(&shifted)
    }
}

// ── SelfNormalizedEstimator ─────────────────────────────────────────────────

/// Self-normalized importance sampling estimator.
///
/// μ̂_SN = Σ w̃_i f(x_i), where w̃_i = w_i / Σ w_j
///
/// More stable than the unnormalized estimator when the normalizing
/// constant of the target is unknown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfNormalizedEstimator {
    pub estimate: f64,
    pub ess: EffectiveSampleSize,
    pub num_samples: usize,
    pub normalized_weights: Vec<f64>,
}

impl SelfNormalizedEstimator {
    /// Compute self-normalized IS estimate.
    pub fn estimate(weights: &[ImportanceWeight], function_values: &[f64]) -> Self {
        let n = weights.len().min(function_values.len());
        if n == 0 {
            return Self {
                estimate: 0.0,
                ess: EffectiveSampleSize::compute(&[]),
                num_samples: 0,
                normalized_weights: Vec::new(),
            };
        }

        let raw_weights: Vec<f64> = weights[..n].iter().map(|w| w.weight).collect();
        let sum_w: f64 = raw_weights.iter().sum();

        let normalized: Vec<f64> = if sum_w > 1e-300 {
            raw_weights.iter().map(|w| w / sum_w).collect()
        } else {
            vec![1.0 / n as f64; n]
        };

        let estimate: f64 = normalized.iter()
            .zip(function_values[..n].iter())
            .map(|(w, f)| w * f)
            .sum();

        let ess = EffectiveSampleSize::compute(&raw_weights);

        Self {
            estimate,
            ess,
            num_samples: n,
            normalized_weights: normalized,
        }
    }

    /// Variance estimate for the self-normalized IS estimator.
    pub fn variance_estimate(&self, function_values: &[f64]) -> f64 {
        let n = self.normalized_weights.len().min(function_values.len());
        if n < 2 {
            return 0.0;
        }

        let mu = self.estimate;
        let var: f64 = self.normalized_weights[..n].iter()
            .zip(function_values[..n].iter())
            .map(|(w, f)| w * (f - mu).powi(2))
            .sum();

        var / self.ess.ess.max(1.0)
    }
}

// ── ImportanceSampler ───────────────────────────────────────────────────────

/// Main importance sampler struct.
pub struct ImportanceSampler {
    proposal: ProposalDistribution,
    target_density: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    function: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    num_samples: usize,
    seed: u64,
}

impl ImportanceSampler {
    pub fn new(
        proposal: ProposalDistribution,
        target_density: impl Fn(f64) -> f64 + Send + Sync + 'static,
        function: impl Fn(f64) -> f64 + Send + Sync + 'static,
        num_samples: usize,
        seed: u64,
    ) -> Self {
        Self {
            proposal,
            target_density: Box::new(target_density),
            function: Box::new(function),
            num_samples,
            seed,
        }
    }

    /// Run importance sampling and return the IS estimate.
    pub fn run(&self) -> ImportanceSamplingResult {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut weights = Vec::with_capacity(self.num_samples);
        let mut function_values = Vec::with_capacity(self.num_samples);
        let mut samples = Vec::with_capacity(self.num_samples);

        for _ in 0..self.num_samples {
            let x = self.proposal.sample(&mut rng);
            let target_d = (self.target_density)(x);
            let proposal_d = self.proposal.density(x);

            let iw = ImportanceWeight::compute(x, target_d, proposal_d);
            let fval = (self.function)(x);

            weights.push(iw);
            function_values.push(fval);
            samples.push(x);
        }

        let estimator = SelfNormalizedEstimator::estimate(&weights, &function_values);
        let variance = estimator.variance_estimate(&function_values);
        let se = (variance / self.num_samples as f64).sqrt();

        let ci = ConfidenceInterval::new(
            estimator.estimate - 1.96 * se,
            estimator.estimate + 1.96 * se,
            0.95,
            estimator.estimate,
        );

        ImportanceSamplingResult {
            estimate: estimator.estimate,
            std_error: se,
            confidence_interval: ci,
            ess: estimator.ess,
            weights,
            function_values,
            samples,
            num_samples: self.num_samples,
        }
    }
}

/// Result of importance sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceSamplingResult {
    pub estimate: f64,
    pub std_error: f64,
    pub confidence_interval: ConfidenceInterval,
    pub ess: EffectiveSampleSize,
    pub weights: Vec<ImportanceWeight>,
    pub function_values: Vec<f64>,
    pub samples: Vec<f64>,
    pub num_samples: usize,
}

// ── StratifiedImportanceSampling ────────────────────────────────────────────

/// Stratified importance sampling: divide sample space into strata
/// and sample from each stratum proportionally.
pub struct StratifiedImportanceSampling {
    strata: Vec<Stratum>,
    target_density: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    function: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    seed: u64,
}

#[derive(Debug, Clone)]
pub struct Stratum {
    pub lo: f64,
    pub hi: f64,
    pub num_samples: usize,
}

impl StratifiedImportanceSampling {
    pub fn new(
        strata: Vec<Stratum>,
        target_density: impl Fn(f64) -> f64 + Send + Sync + 'static,
        function: impl Fn(f64) -> f64 + Send + Sync + 'static,
        seed: u64,
    ) -> Self {
        Self {
            strata,
            target_density: Box::new(target_density),
            function: Box::new(function),
            seed,
        }
    }

    /// Create uniform strata over [lo, hi].
    pub fn uniform_strata(
        lo: f64,
        hi: f64,
        num_strata: usize,
        samples_per_stratum: usize,
        target_density: impl Fn(f64) -> f64 + Send + Sync + 'static,
        function: impl Fn(f64) -> f64 + Send + Sync + 'static,
        seed: u64,
    ) -> Self {
        let step = (hi - lo) / num_strata as f64;
        let strata = (0..num_strata)
            .map(|i| Stratum {
                lo: lo + i as f64 * step,
                hi: lo + (i + 1) as f64 * step,
                num_samples: samples_per_stratum,
            })
            .collect();
        Self::new(strata, target_density, function, seed)
    }

    /// Run stratified IS.
    pub fn run(&self) -> ImportanceSamplingResult {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut all_weights = Vec::new();
        let mut all_fvals = Vec::new();
        let mut all_samples = Vec::new();

        for stratum in &self.strata {
            let proposal = ProposalDistribution::Uniform { lo: stratum.lo, hi: stratum.hi };
            for _ in 0..stratum.num_samples {
                let x = proposal.sample(&mut rng);
                let td = (self.target_density)(x);
                let pd = proposal.density(x);
                let iw = ImportanceWeight::compute(x, td, pd);
                let fval = (self.function)(x);

                all_weights.push(iw);
                all_fvals.push(fval);
                all_samples.push(x);
            }
        }

        let estimator = SelfNormalizedEstimator::estimate(&all_weights, &all_fvals);
        let variance = estimator.variance_estimate(&all_fvals);
        let n = all_weights.len();
        let se = (variance / n as f64).sqrt();

        let ci = ConfidenceInterval::new(
            estimator.estimate - 1.96 * se,
            estimator.estimate + 1.96 * se,
            0.95,
            estimator.estimate,
        );

        ImportanceSamplingResult {
            estimate: estimator.estimate,
            std_error: se,
            confidence_interval: ci,
            ess: estimator.ess,
            weights: all_weights,
            function_values: all_fvals,
            samples: all_samples,
            num_samples: n,
        }
    }
}

// ── ImportanceSamplingCI ────────────────────────────────────────────────────

/// Confidence interval derived from importance sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceSamplingCI {
    pub estimate: f64,
    pub ci: ConfidenceInterval,
    pub ess: f64,
    pub method: String,
}

impl ImportanceSamplingCI {
    /// Normal approximation CI from IS estimate and variance.
    pub fn normal_approximation(
        estimate: f64,
        variance: f64,
        n: usize,
        confidence_level: f64,
    ) -> Self {
        let se = (variance / n as f64).sqrt();
        let z = if confidence_level >= 0.99 { 2.576 } else { 1.96 };
        Self {
            estimate,
            ci: ConfidenceInterval::new(
                estimate - z * se,
                estimate + z * se,
                confidence_level,
                estimate,
            ),
            ess: n as f64,
            method: "normal_approximation".to_string(),
        }
    }

    /// Check if zero is excluded (significant effect).
    pub fn is_significant(&self) -> bool {
        self.ci.lower > 0.0 || self.ci.upper < 0.0
    }
}

// ── OptimalProposal ─────────────────────────────────────────────────────────

/// Adapt proposal distribution to minimize variance.
pub fn adapt_proposal(
    samples: &[f64],
    weights: &[f64],
    function_values: &[f64],
) -> ProposalDistribution {
    if samples.is_empty() || weights.is_empty() {
        return ProposalDistribution::Uniform { lo: 0.0, hi: 1.0 };
    }

    // Compute weighted mean and std for adapted Gaussian proposal
    let sum_w: f64 = weights.iter().sum();
    if sum_w < 1e-300 {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        return ProposalDistribution::Gaussian { mean, std: 1.0 };
    }

    let w_norm: Vec<f64> = weights.iter().map(|w| w / sum_w).collect();

    let weighted_mean: f64 = samples.iter()
        .zip(w_norm.iter())
        .map(|(x, w)| x * w)
        .sum();

    let weighted_var: f64 = samples.iter()
        .zip(w_norm.iter())
        .map(|(x, w)| w * (x - weighted_mean).powi(2))
        .sum();

    let std = weighted_var.sqrt().max(0.01);

    // Create mixture: half near weighted mean, half wider
    ProposalDistribution::GaussianMixture {
        means: vec![weighted_mean, weighted_mean],
        stds: vec![std, std * 3.0],
        weights: vec![0.7, 0.3],
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_proposal() {
        let prop = ProposalDistribution::Uniform { lo: 0.0, hi: 1.0 };
        let mut rng = StdRng::seed_from_u64(42);
        let sample = prop.sample(&mut rng);
        assert!(sample >= 0.0 && sample <= 1.0);
        assert!((prop.density(0.5) - 1.0).abs() < 1e-10);
        assert!((prop.density(1.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_proposal() {
        let prop = ProposalDistribution::Gaussian { mean: 5.0, std: 1.0 };
        let mut rng = StdRng::seed_from_u64(42);
        let samples: Vec<f64> = (0..1000).map(|_| prop.sample(&mut rng)).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_gaussian_mixture_proposal() {
        let prop = ProposalDistribution::GaussianMixture {
            means: vec![0.0, 5.0],
            stds: vec![1.0, 1.0],
            weights: vec![0.5, 0.5],
        };
        let d = prop.density(0.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_focal_point_proposal() {
        let prop = ProposalDistribution::FocalPoint {
            center: 3.0,
            scale: 1.0,
            tail_weight: 0.2,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let sample = prop.sample(&mut rng);
        assert!(sample.is_finite());
        assert!(prop.density(3.0) > 0.0);
    }

    #[test]
    fn test_importance_weight() {
        let iw = ImportanceWeight::compute(1.0, 0.5, 0.25);
        assert!((iw.weight - 2.0).abs() < 1e-10);
        assert!(iw.is_valid());
    }

    #[test]
    fn test_importance_weight_zero_proposal() {
        let iw = ImportanceWeight::compute(1.0, 0.5, 0.0);
        assert!((iw.weight - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_effective_sample_size_uniform() {
        let weights = vec![1.0; 100];
        let ess = EffectiveSampleSize::compute(&weights);
        assert!((ess.ess - 100.0).abs() < 1e-10);
        assert!(!ess.is_degenerate);
    }

    #[test]
    fn test_effective_sample_size_degenerate() {
        let mut weights = vec![0.0; 100];
        weights[0] = 1.0;
        let ess = EffectiveSampleSize::compute(&weights);
        assert!((ess.ess - 1.0).abs() < 1e-10);
        assert!(ess.is_degenerate);
    }

    #[test]
    fn test_self_normalized_estimator() {
        let weights: Vec<ImportanceWeight> = (0..100)
            .map(|_| ImportanceWeight::compute(1.0, 1.0, 1.0))
            .collect();
        let values: Vec<f64> = vec![5.0; 100];
        let est = SelfNormalizedEstimator::estimate(&weights, &values);
        assert!((est.estimate - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_importance_sampler() {
        let proposal = ProposalDistribution::Uniform { lo: 0.0, hi: 1.0 };
        let target = |x: f64| -> f64 { if x >= 0.0 && x <= 1.0 { 1.0 } else { 0.0 } };
        let function = |x: f64| -> f64 { x * x };
        let sampler = ImportanceSampler::new(proposal, target, function, 10000, 42);
        let result = sampler.run();
        // E[X²] for Uniform(0,1) = 1/3
        assert!((result.estimate - 1.0 / 3.0).abs() < 0.05);
    }

    #[test]
    fn test_stratified_is() {
        let sis = StratifiedImportanceSampling::uniform_strata(
            0.0, 1.0, 10, 100,
            |x| if x >= 0.0 && x <= 1.0 { 1.0 } else { 0.0 },
            |x| x,
            42,
        );
        let result = sis.run();
        // E[X] for Uniform(0,1) = 0.5
        assert!((result.estimate - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_is_ci_normal() {
        let ci = ImportanceSamplingCI::normal_approximation(5.0, 1.0, 100, 0.95);
        assert!(ci.ci.contains(5.0));
        assert!(ci.ci.width() > 0.0);
    }

    #[test]
    fn test_is_ci_significance() {
        let ci = ImportanceSamplingCI::normal_approximation(5.0, 0.1, 1000, 0.95);
        assert!(ci.is_significant()); // both bounds > 0
    }

    #[test]
    fn test_adapt_proposal() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![0.1, 0.2, 0.4, 0.2, 0.1];
        let fvals = vec![1.0, 4.0, 9.0, 16.0, 25.0];
        let adapted = adapt_proposal(&samples, &weights, &fvals);
        match adapted {
            ProposalDistribution::GaussianMixture { means, .. } => {
                // Weighted mean should be around 3.0
                assert!((means[0] - 3.0).abs() < 0.5);
            }
            _ => panic!("Expected GaussianMixture"),
        }
    }

    #[test]
    fn test_proposal_for_punishment() {
        let prop = ProposalDistribution::for_punishment(3.0, 5.0, 10.0);
        let d = prop.density(3.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_ess_from_log_weights() {
        let log_weights = vec![0.0; 50]; // all equal weights
        let ess = EffectiveSampleSize::from_log_weights(&log_weights);
        assert!((ess.ess - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_variance_estimate() {
        let weights: Vec<ImportanceWeight> = (0..100)
            .map(|_| ImportanceWeight::compute(1.0, 1.0, 1.0))
            .collect();
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let est = SelfNormalizedEstimator::estimate(&weights, &values);
        let var = est.variance_estimate(&values);
        assert!(var > 0.0);
    }
}
