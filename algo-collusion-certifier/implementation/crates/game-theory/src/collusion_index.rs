//! Collusion measurement: collusion premium, collusion index, severity classification.
//!
//! Provides multiple metrics for quantifying the degree of collusion relative to
//! competitive and monopoly benchmarks, with confidence intervals and interval arithmetic.

use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use shared_types::*;

// ── Collusion Premium ───────────────────────────────────────────────────────

/// Collusion premium: CP = (π_obs - π_NE) / π_NE
///
/// Measures excess profit relative to the competitive (Nash) benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollusionPremium {
    pub value: f64,
    pub observed_profit: f64,
    pub nash_profit: f64,
}

impl CollusionPremium {
    /// Compute the collusion premium.
    /// Returns 0 if Nash profit is zero or negative (use AbsoluteMargin instead).
    pub fn compute(observed_profit: f64, nash_profit: f64) -> Self {
        let value = if nash_profit.abs() < 1e-12 {
            // Zero-profit NE (e.g., homogeneous Bertrand): CP is undefined.
            // Use observed_profit as a direct measure.
            if observed_profit > 1e-12 { f64::INFINITY } else { 0.0 }
        } else {
            (observed_profit - nash_profit) / nash_profit
        };
        Self { value, observed_profit, nash_profit }
    }

    /// Is there positive collusion premium?
    pub fn is_collusive(&self) -> bool {
        self.value > 1e-6
    }
}

// ── Absolute Margin ─────────────────────────────────────────────────────────

/// For homogeneous Bertrand where NE profit is zero, measure the absolute price
/// margin above marginal cost: Δp = p_obs - mc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbsoluteMargin {
    pub margin: f64,
    pub observed_price: f64,
    pub marginal_cost: f64,
    pub relative_margin: f64,
}

impl AbsoluteMargin {
    pub fn compute(observed_price: f64, marginal_cost: f64) -> Self {
        let margin = observed_price - marginal_cost;
        let relative = if marginal_cost.abs() > 1e-12 {
            margin / marginal_cost
        } else {
            if margin > 1e-12 { f64::INFINITY } else { 0.0 }
        };
        Self {
            margin,
            observed_price,
            marginal_cost,
            relative_margin: relative,
        }
    }

    pub fn is_above_competitive(&self) -> bool {
        self.margin > 1e-6
    }
}

// ── Collusion Index ─────────────────────────────────────────────────────────

/// Normalized collusion index: CI = CP / (1 + CP), bounded in [0, 1).
///
/// CI = 0 corresponds to competitive behavior.
/// CI approaching 1 corresponds to extreme collusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollusionIndex {
    pub value: f64,
    pub collusion_premium: f64,
}

impl CollusionIndex {
    pub fn from_premium(cp: f64) -> Self {
        let value = if cp.is_infinite() {
            1.0
        } else if cp <= 0.0 {
            0.0
        } else {
            cp / (1.0 + cp)
        };
        Self { value, collusion_premium: cp }
    }

    pub fn compute(observed_profit: f64, nash_profit: f64) -> Self {
        let cp = CollusionPremium::compute(observed_profit, nash_profit);
        Self::from_premium(cp.value)
    }

    /// Alternative: normalized between Nash and monopoly profits.
    pub fn normalized(observed_profit: f64, nash_profit: f64, monopoly_profit: f64) -> Self {
        let range = monopoly_profit - nash_profit;
        let value = if range.abs() < 1e-12 {
            0.0
        } else {
            ((observed_profit - nash_profit) / range).clamp(0.0, 1.0)
        };
        let cp = if nash_profit.abs() > 1e-12 {
            (observed_profit - nash_profit) / nash_profit
        } else {
            0.0
        };
        Self { value, collusion_premium: cp }
    }
}

// ── Collusion Premium Confidence Intervals ──────────────────────────────────

/// Bootstrap confidence intervals for the collusion premium.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollusionPremiumCI {
    pub point_estimate: f64,
    pub ci: ConfidenceInterval,
    pub num_bootstrap_samples: usize,
    pub bootstrap_distribution: Vec<f64>,
}

impl CollusionPremiumCI {
    /// Compute bootstrap CI for the collusion premium from observed profit samples.
    pub fn bootstrap(
        profit_samples: &[f64],
        nash_profit: f64,
        confidence_level: f64,
        num_bootstrap: usize,
        seed: u64,
    ) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n = profit_samples.len();
        if n == 0 {
            return Self {
                point_estimate: 0.0,
                ci: ConfidenceInterval::new(0.0, 0.0, confidence_level, 0.0),
                num_bootstrap_samples: 0,
                bootstrap_distribution: vec![],
            };
        }

        let observed_mean: f64 = profit_samples.iter().sum::<f64>() / n as f64;
        let point_cp = CollusionPremium::compute(observed_mean, nash_profit).value;

        let mut bootstrap_cps = Vec::with_capacity(num_bootstrap);
        for _ in 0..num_bootstrap {
            let mut resample_sum = 0.0;
            for _ in 0..n {
                let idx = rng.gen_range(0..n);
                resample_sum += profit_samples[idx];
            }
            let resample_mean = resample_sum / n as f64;
            let cp = CollusionPremium::compute(resample_mean, nash_profit).value;
            if cp.is_finite() {
                bootstrap_cps.push(cp);
            }
        }

        bootstrap_cps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = 1.0 - confidence_level;
        let lower_idx = ((alpha / 2.0) * bootstrap_cps.len() as f64).floor() as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * bootstrap_cps.len() as f64).ceil() as usize;
        let lower = bootstrap_cps.get(lower_idx).copied().unwrap_or(point_cp);
        let upper = bootstrap_cps.get(upper_idx.min(bootstrap_cps.len().saturating_sub(1))).copied().unwrap_or(point_cp);

        Self {
            point_estimate: point_cp,
            ci: ConfidenceInterval::new(lower, upper, confidence_level, point_cp),
            num_bootstrap_samples: num_bootstrap,
            bootstrap_distribution: bootstrap_cps,
        }
    }

    /// Is the collusion premium significantly positive?
    pub fn is_significant(&self) -> bool {
        self.ci.lower > 0.0
    }
}

// ── Collusion Severity ──────────────────────────────────────────────────────

/// Classification of collusion severity based on the collusion index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollusionSeverity {
    None,
    Mild,
    Moderate,
    Severe,
    Extreme,
}

impl CollusionSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Mild => "Mild",
            Self::Moderate => "Moderate",
            Self::Severe => "Severe",
            Self::Extreme => "Extreme",
        }
    }
}

impl std::fmt::Display for CollusionSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Classify collusion severity from a collusion index value in [0, 1].
pub fn classify_collusion(ci: f64) -> CollusionSeverity {
    if ci < 0.05 {
        CollusionSeverity::None
    } else if ci < 0.20 {
        CollusionSeverity::Mild
    } else if ci < 0.50 {
        CollusionSeverity::Moderate
    } else if ci < 0.80 {
        CollusionSeverity::Severe
    } else {
        CollusionSeverity::Extreme
    }
}

// ── Interval Collusion Premium ──────────────────────────────────────────────

/// CP computation with interval arithmetic for rigorous bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalCollusionPremium {
    pub lower: f64,
    pub upper: f64,
    pub observed_interval: IntervalF64,
    pub nash_interval: IntervalF64,
}

impl IntervalCollusionPremium {
    /// Compute CP with interval-valued profits.
    pub fn compute(observed: IntervalF64, nash: IntervalF64) -> Self {
        // CP = (obs - nash) / nash
        // Lower bound: (obs.lo - nash.hi) / nash.hi  (most conservative)
        // Upper bound: (obs.hi - nash.lo) / nash.lo  (most optimistic)
        let lower = if nash.hi.abs() > 1e-12 {
            (observed.lo - nash.hi) / nash.hi
        } else {
            0.0
        };
        let upper = if nash.lo.abs() > 1e-12 {
            (observed.hi - nash.lo) / nash.lo
        } else {
            f64::INFINITY
        };

        Self {
            lower: lower.max(0.0),
            upper,
            observed_interval: observed,
            nash_interval: nash,
        }
    }

    /// Is the collusion premium certifiably positive?
    pub fn is_certifiably_collusive(&self) -> bool {
        self.lower > 0.0
    }

    pub fn midpoint(&self) -> f64 {
        if self.upper.is_infinite() {
            self.lower
        } else {
            (self.lower + self.upper) / 2.0
        }
    }
}

// ── Monopoly Benchmark ──────────────────────────────────────────────────────

/// Compare observed profits to monopoly profits as an upper bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonopolyBenchmark {
    /// Fraction of monopoly profit achieved: (obs - nash) / (mono - nash).
    pub fraction_of_monopoly: f64,
    pub observed_profit: f64,
    pub nash_profit: f64,
    pub monopoly_profit: f64,
}

impl MonopolyBenchmark {
    pub fn compute(observed: f64, nash: f64, monopoly: f64) -> Self {
        let range = monopoly - nash;
        let fraction = if range.abs() < 1e-12 {
            0.0
        } else {
            ((observed - nash) / range).clamp(0.0, 1.5) // allow slight overshoot
        };
        Self {
            fraction_of_monopoly: fraction,
            observed_profit: observed,
            nash_profit: nash,
            monopoly_profit: monopoly,
        }
    }

    /// Is observed profit above the monopoly benchmark?
    pub fn exceeds_monopoly(&self) -> bool {
        self.observed_profit > self.monopoly_profit + 1e-6
    }
}

// ── Smooth CP Transition ────────────────────────────────────────────────────

/// Smooth handling of near-zero competitive profits to avoid division instability.
///
/// Uses a smoothed denominator: max(π_NE, ε) or a logistic transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothCPTransition {
    pub smoothed_cp: f64,
    pub raw_cp: f64,
    pub smoothing_parameter: f64,
}

impl SmoothCPTransition {
    /// Compute smoothed CP using max(nash, epsilon) as denominator.
    pub fn with_floor(observed: f64, nash: f64, epsilon: f64) -> Self {
        let eps = epsilon.max(1e-6);
        let denom = nash.max(eps);
        let raw = if nash.abs() > 1e-12 {
            (observed - nash) / nash
        } else {
            if observed > 1e-12 { f64::INFINITY } else { 0.0 }
        };
        let smoothed = (observed - nash) / denom;
        Self {
            smoothed_cp: smoothed.max(0.0),
            raw_cp: raw,
            smoothing_parameter: eps,
        }
    }

    /// Compute smoothed CP using logistic blending between absolute and relative measures.
    pub fn logistic_blend(observed: f64, nash: f64, transition_width: f64) -> Self {
        let tw = transition_width.max(1e-6);
        // Logistic weight: w = 1 / (1 + exp(-nash/tw))
        // For large nash: w ≈ 1 (use relative CP)
        // For small nash: w ≈ 0 (use absolute margin)
        let w = 1.0 / (1.0 + (-nash / tw).exp());
        let relative_cp = if nash.abs() > 1e-12 {
            (observed - nash) / nash
        } else {
            0.0
        };
        let absolute_cp = observed - nash;
        let smoothed = w * relative_cp + (1.0 - w) * absolute_cp;

        let raw = if nash.abs() > 1e-12 {
            (observed - nash) / nash
        } else {
            if observed > 1e-12 { f64::INFINITY } else { 0.0 }
        };

        Self {
            smoothed_cp: smoothed.max(0.0),
            raw_cp: raw,
            smoothing_parameter: tw,
        }
    }
}

// ── Demand Robustness ───────────────────────────────────────────────────────

/// CP computation under different demand specifications for robustness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandRobustnessResult {
    pub base_cp: f64,
    pub cp_under_variants: Vec<(String, f64)>,
    pub min_cp: f64,
    pub max_cp: f64,
    pub is_robust: bool,
}

/// Compute CP under different demand perturbations.
pub fn demand_robust_cp(
    observed_profit: f64,
    base_nash_profit: f64,
    demand_perturbations: &[(String, f64)], // (label, perturbed_nash_profit)
) -> DemandRobustnessResult {
    let base_cp = CollusionPremium::compute(observed_profit, base_nash_profit).value;

    let variants: Vec<(String, f64)> = demand_perturbations.iter().map(|(label, nash)| {
        let cp = CollusionPremium::compute(observed_profit, *nash).value;
        (label.clone(), if cp.is_finite() { cp } else { base_cp })
    }).collect();

    let mut all_cps: Vec<f64> = variants.iter().map(|(_, cp)| *cp).collect();
    all_cps.push(if base_cp.is_finite() { base_cp } else { 0.0 });

    let min_cp = all_cps.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_cp = all_cps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Robust if sign doesn't change and range is bounded
    let is_robust = (min_cp > -0.1 && max_cp < min_cp * 3.0 + 0.1)
        || (all_cps.iter().all(|&cp| cp > 0.0) || all_cps.iter().all(|&cp| cp <= 0.0));

    DemandRobustnessResult {
        base_cp: if base_cp.is_finite() { base_cp } else { 0.0 },
        cp_under_variants: variants,
        min_cp,
        max_cp,
        is_robust,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collusion_premium_positive() {
        let cp = CollusionPremium::compute(3.0, 2.0);
        assert!((cp.value - 0.5).abs() < 1e-10);
        assert!(cp.is_collusive());
    }

    #[test]
    fn test_collusion_premium_zero() {
        let cp = CollusionPremium::compute(2.0, 2.0);
        assert!((cp.value - 0.0).abs() < 1e-10);
        assert!(!cp.is_collusive());
    }

    #[test]
    fn test_collusion_premium_zero_nash() {
        let cp = CollusionPremium::compute(1.0, 0.0);
        assert!(cp.value.is_infinite());
    }

    #[test]
    fn test_absolute_margin() {
        let am = AbsoluteMargin::compute(5.0, 3.0);
        assert!((am.margin - 2.0).abs() < 1e-10);
        assert!(am.is_above_competitive());
    }

    #[test]
    fn test_absolute_margin_at_mc() {
        let am = AbsoluteMargin::compute(3.0, 3.0);
        assert!((am.margin - 0.0).abs() < 1e-10);
        assert!(!am.is_above_competitive());
    }

    #[test]
    fn test_collusion_index_from_premium() {
        let ci = CollusionIndex::from_premium(1.0);
        assert!((ci.value - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_collusion_index_zero() {
        let ci = CollusionIndex::from_premium(0.0);
        assert!((ci.value - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_collusion_index_infinite() {
        let ci = CollusionIndex::from_premium(f64::INFINITY);
        assert!((ci.value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_collusion_index_normalized() {
        let ci = CollusionIndex::normalized(2.0, 1.0, 3.0);
        assert!((ci.value - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_classify_collusion_none() {
        assert_eq!(classify_collusion(0.02), CollusionSeverity::None);
    }

    #[test]
    fn test_classify_collusion_mild() {
        assert_eq!(classify_collusion(0.10), CollusionSeverity::Mild);
    }

    #[test]
    fn test_classify_collusion_moderate() {
        assert_eq!(classify_collusion(0.35), CollusionSeverity::Moderate);
    }

    #[test]
    fn test_classify_collusion_severe() {
        assert_eq!(classify_collusion(0.65), CollusionSeverity::Severe);
    }

    #[test]
    fn test_classify_collusion_extreme() {
        assert_eq!(classify_collusion(0.90), CollusionSeverity::Extreme);
    }

    #[test]
    fn test_bootstrap_ci() {
        let samples: Vec<f64> = vec![3.0, 3.1, 2.9, 3.2, 2.8, 3.0, 3.1, 2.9, 3.0, 3.0];
        let ci = CollusionPremiumCI::bootstrap(&samples, 2.0, 0.95, 1000, 42);
        assert!(ci.point_estimate > 0.0);
        assert!(ci.ci.lower <= ci.ci.upper);
    }

    #[test]
    fn test_bootstrap_ci_significance() {
        // Strong collusion signal
        let samples: Vec<f64> = vec![5.0; 20];
        let ci = CollusionPremiumCI::bootstrap(&samples, 2.0, 0.95, 1000, 42);
        assert!(ci.is_significant());
    }

    #[test]
    fn test_bootstrap_ci_not_significant() {
        // No collusion
        let samples: Vec<f64> = vec![2.0; 20];
        let ci = CollusionPremiumCI::bootstrap(&samples, 2.0, 0.95, 1000, 42);
        assert!(!ci.is_significant());
    }

    #[test]
    fn test_interval_cp() {
        let observed = Interval::new(2.8, 3.2);
        let nash = Interval::new(1.9, 2.1);
        let icp = IntervalCollusionPremium::compute(observed, nash);
        assert!(icp.lower > 0.0);
        assert!(icp.upper > icp.lower);
    }

    #[test]
    fn test_interval_cp_certifiable() {
        let observed = Interval::new(3.0, 4.0);
        let nash = Interval::new(1.0, 1.5);
        let icp = IntervalCollusionPremium::compute(observed, nash);
        assert!(icp.is_certifiably_collusive());
    }

    #[test]
    fn test_monopoly_benchmark() {
        let mb = MonopolyBenchmark::compute(2.0, 1.0, 3.0);
        assert!((mb.fraction_of_monopoly - 0.5).abs() < 1e-10);
        assert!(!mb.exceeds_monopoly());
    }

    #[test]
    fn test_monopoly_benchmark_exceeds() {
        let mb = MonopolyBenchmark::compute(4.0, 1.0, 3.0);
        assert!(mb.exceeds_monopoly());
    }

    #[test]
    fn test_smooth_cp_floor() {
        let scp = SmoothCPTransition::with_floor(3.0, 0.001, 0.1);
        assert!(scp.smoothed_cp > 0.0);
        assert!(scp.smoothed_cp.is_finite());
    }

    #[test]
    fn test_smooth_cp_logistic() {
        let scp = SmoothCPTransition::logistic_blend(3.0, 2.0, 1.0);
        assert!(scp.smoothed_cp > 0.0);
    }

    #[test]
    fn test_demand_robustness() {
        let perturbations = vec![
            ("high_elasticity".to_string(), 1.5),
            ("low_elasticity".to_string(), 2.5),
        ];
        let result = demand_robust_cp(3.0, 2.0, &perturbations);
        assert!(result.base_cp > 0.0);
        assert_eq!(result.cp_under_variants.len(), 2);
    }

    #[test]
    fn test_demand_robustness_consistent_sign() {
        let perturbations = vec![
            ("v1".to_string(), 1.8),
            ("v2".to_string(), 2.2),
        ];
        let result = demand_robust_cp(3.0, 2.0, &perturbations);
        assert!(result.is_robust);
    }
}
