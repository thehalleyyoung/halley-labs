//! Statistical bound computation using Hoeffding's inequality.
//!
//! Implements error bounds for stratified sampling over the body-parameter
//! space. Provides the Hoeffding inequality bounds with union bound
//! corrections for multiple elements and devices, and the volume-subtraction
//! model where Tier 1 green regions reduce the effective unverified fraction.

use serde::{Deserialize, Serialize};

// ─────────────────────── Hoeffding Bound ───────────────────────────────────

/// Hoeffding inequality-based statistical bound computation.
///
/// Given `n` i.i.d. samples from a stratum with binary outcomes (pass/fail),
/// Hoeffding's inequality gives:
///
///   P(|p̂ - p| ≥ ε) ≤ 2·exp(-2nε²)
///
/// For stratified sampling with K strata, m elements, and p devices,
/// we apply the union bound:
///
///   ε = √( ln(2·K·m·p / δ) / (2·n_min) )
///
/// where n_min is the minimum sample count per stratum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoeffdingBound {
    /// Confidence parameter δ (probability of error exceeding bound).
    pub delta: f64,
    /// Number of strata K.
    pub num_strata: usize,
    /// Number of interactable elements m.
    pub num_elements: usize,
    /// Number of device configurations p.
    pub num_devices: usize,
}

impl HoeffdingBound {
    /// Create a new Hoeffding bound configuration.
    pub fn new(
        delta: f64,
        num_strata: usize,
        num_elements: usize,
        num_devices: usize,
    ) -> Self {
        assert!(delta > 0.0 && delta < 1.0, "δ must be in (0, 1)");
        assert!(num_strata > 0, "Must have at least 1 stratum");
        assert!(num_elements > 0, "Must have at least 1 element");
        assert!(num_devices > 0, "Must have at least 1 device");
        Self {
            delta,
            num_strata,
            num_elements,
            num_devices,
        }
    }

    /// Number of union-bound terms: K × m × p.
    pub fn num_hypotheses(&self) -> usize {
        self.num_strata * self.num_elements * self.num_devices
    }

    /// Per-hypothesis confidence parameter after Bonferroni correction.
    ///
    /// δ_per = δ / (K · m · p)
    pub fn per_hypothesis_delta(&self) -> f64 {
        self.delta / self.num_hypotheses() as f64
    }

    /// Compute the ε (error bound) given `n_samples` total samples,
    /// distributed uniformly across strata.
    ///
    /// ε = √( ln(2·K·m·p / δ) / (2·n_per_stratum) )
    pub fn compute_epsilon(
        &self,
        n_samples: usize,
        delta: f64,
        num_strata: usize,
        num_elements: usize,
        num_devices: usize,
    ) -> f64 {
        if n_samples == 0 || num_strata == 0 {
            return 1.0;
        }
        let n_per_stratum = n_samples / num_strata.max(1);
        if n_per_stratum == 0 {
            return 1.0;
        }
        let h = num_strata * num_elements * num_devices;
        let log_term = (2.0 * h as f64 / delta).ln();
        (log_term / (2.0 * n_per_stratum as f64)).sqrt().min(1.0)
    }

    /// Compute ε using instance parameters.
    pub fn epsilon(&self, n_samples: usize) -> f64 {
        self.compute_epsilon(
            n_samples,
            self.delta,
            self.num_strata,
            self.num_elements,
            self.num_devices,
        )
    }

    /// Compute per-stratum error bound given stratum sample count.
    ///
    /// ε_s = √( ln(2·m·p / δ_s) / (2·n_s) )
    ///
    /// where δ_s = δ / K is the per-stratum confidence budget.
    pub fn per_stratum_bound(&self, stratum_sample_count: usize) -> f64 {
        if stratum_sample_count == 0 {
            return 1.0;
        }
        let delta_s = self.delta / self.num_strata as f64;
        let h = self.num_elements * self.num_devices;
        let log_term = (2.0 * h as f64 / delta_s).ln();
        (log_term / (2.0 * stratum_sample_count as f64))
            .sqrt()
            .min(1.0)
    }

    /// Compute the minimum number of samples needed to achieve target ε.
    ///
    /// n ≥ K · ⌈ ln(2·K·m·p / δ) / (2·ε²) ⌉
    pub fn minimum_samples(
        &self,
        target_epsilon: f64,
        delta: f64,
        num_strata: usize,
    ) -> usize {
        if target_epsilon <= 0.0 {
            return usize::MAX;
        }
        let h = num_strata * self.num_elements * self.num_devices;
        let log_term = (2.0 * h as f64 / delta).ln();
        let per_stratum = (log_term / (2.0 * target_epsilon * target_epsilon)).ceil() as usize;
        num_strata * per_stratum.max(1)
    }

    /// Compute minimum samples using instance parameters.
    pub fn min_samples(&self, target_epsilon: f64) -> usize {
        self.minimum_samples(target_epsilon, self.delta, self.num_strata)
    }

    /// Compute the confidence level (1 - δ) achieved with given n and ε.
    ///
    /// δ_achieved = 2·K·m·p · exp(-2·n_per_stratum·ε²)
    pub fn achieved_delta(&self, n_samples: usize, epsilon: f64) -> f64 {
        if n_samples == 0 {
            return 1.0;
        }
        let n_per = n_samples / self.num_strata.max(1);
        let h = self.num_hypotheses();
        let exp_term = (-2.0 * n_per as f64 * epsilon * epsilon).exp();
        (2.0 * h as f64 * exp_term).min(1.0)
    }

    /// Confidence level (1 - δ_achieved).
    pub fn achieved_confidence(&self, n_samples: usize, epsilon: f64) -> f64 {
        1.0 - self.achieved_delta(n_samples, epsilon)
    }
}

// ────────────────────── Stratified Bound ───────────────────────────────────

/// Stratified Hoeffding bound with per-stratum sample counts.
///
/// When strata have different sample counts (e.g., from adaptive sampling),
/// the overall bound is computed as the volume-weighted combination of
/// per-stratum bounds:
///
///   ε_stratified = Σ_s (w_s · ε_s)
///
/// where w_s = V_s / V_total and ε_s is the per-stratum Hoeffding bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratifiedBound {
    /// Per-stratum data: (volume_weight, sample_count, pass_count).
    pub strata_data: Vec<StratumBoundData>,
    /// Confidence parameter δ.
    pub delta: f64,
    /// Number of elements m.
    pub num_elements: usize,
    /// Number of devices p.
    pub num_devices: usize,
}

/// Data for a single stratum's bound computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumBoundData {
    pub volume_weight: f64,
    pub sample_count: usize,
    pub pass_count: usize,
    pub fail_count: usize,
}

impl StratumBoundData {
    /// Observed pass rate in this stratum.
    pub fn pass_rate(&self) -> f64 {
        if self.sample_count == 0 {
            0.0
        } else {
            self.pass_count as f64 / self.sample_count as f64
        }
    }
}

impl StratifiedBound {
    /// Create a new stratified bound.
    pub fn new(delta: f64, num_elements: usize, num_devices: usize) -> Self {
        Self {
            strata_data: Vec::new(),
            delta,
            num_elements,
            num_devices,
        }
    }

    /// Add stratum data.
    pub fn add_stratum(
        &mut self,
        volume_weight: f64,
        sample_count: usize,
        pass_count: usize,
        fail_count: usize,
    ) {
        self.strata_data.push(StratumBoundData {
            volume_weight,
            sample_count,
            pass_count,
            fail_count,
        });
    }

    /// Number of strata.
    pub fn num_strata(&self) -> usize {
        self.strata_data.len()
    }

    /// Per-stratum δ budget (Bonferroni).
    pub fn per_stratum_delta(&self) -> f64 {
        let k = self.strata_data.len().max(1);
        self.delta / k as f64
    }

    /// Per-stratum ε for a stratum with given sample count.
    pub fn stratum_epsilon(&self, sample_count: usize) -> f64 {
        if sample_count == 0 {
            return 1.0;
        }
        let delta_s = self.per_stratum_delta();
        let h = self.num_elements * self.num_devices;
        let log_term = (2.0 * h as f64 / delta_s).ln();
        (log_term / (2.0 * sample_count as f64)).sqrt().min(1.0)
    }

    /// Compute the overall stratified error bound.
    ///
    /// ε = Σ_s w_s · ε_s
    pub fn overall_epsilon(&self) -> f64 {
        self.strata_data
            .iter()
            .map(|s| {
                let eps_s = self.stratum_epsilon(s.sample_count);
                s.volume_weight * eps_s
            })
            .sum::<f64>()
            .min(1.0)
    }

    /// Compute the overall estimated coverage (volume-weighted pass rate).
    pub fn estimated_coverage(&self) -> f64 {
        self.strata_data
            .iter()
            .map(|s| s.volume_weight * s.pass_rate())
            .sum()
    }

    /// Compute the lower confidence bound on coverage.
    ///
    /// κ_lower = Σ_s w_s · max(0, p̂_s - ε_s)
    pub fn coverage_lower_bound(&self) -> f64 {
        self.strata_data
            .iter()
            .map(|s| {
                let eps_s = self.stratum_epsilon(s.sample_count);
                s.volume_weight * (s.pass_rate() - eps_s).max(0.0)
            })
            .sum()
    }

    /// Compute the upper confidence bound on coverage.
    ///
    /// κ_upper = Σ_s w_s · min(1, p̂_s + ε_s)
    pub fn coverage_upper_bound(&self) -> f64 {
        self.strata_data
            .iter()
            .map(|s| {
                let eps_s = self.stratum_epsilon(s.sample_count);
                s.volume_weight * (s.pass_rate() + eps_s).min(1.0)
            })
            .sum()
    }

    /// Total number of samples across all strata.
    pub fn total_samples(&self) -> usize {
        self.strata_data.iter().map(|s| s.sample_count).sum()
    }

    /// Minimum per-stratum sample count.
    pub fn min_stratum_samples(&self) -> usize {
        self.strata_data
            .iter()
            .map(|s| s.sample_count)
            .min()
            .unwrap_or(0)
    }
}

// ────────────────── Volume Subtraction Model ──────────────────────────────

/// Volume-subtraction model for combining Tier 1 green regions with sampling.
///
/// Tier 1 verification proves that certain sub-regions of the parameter space
/// are definitely accessible (green) or inaccessible (red). The green volume
/// reduces the effective fraction that must be covered by sampling:
///
///   ρ_s = 1 - V_green / V_total
///
/// The effective ε for the combined certificate accounts for this:
///
///   ε_effective = ρ_s · ε_sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeSubtractionModel {
    /// Total parameter-space volume.
    pub total_volume: f64,
    /// Volume provably verified (green) by Tier 1.
    pub green_volume: f64,
    /// Volume provably unreachable (red) by Tier 1.
    pub red_volume: f64,
    /// Volume verified by SMT proofs.
    pub smt_verified_volume: f64,
}

impl VolumeSubtractionModel {
    /// Create a new volume-subtraction model.
    pub fn new(total_volume: f64) -> Self {
        Self {
            total_volume: total_volume.max(0.0),
            green_volume: 0.0,
            red_volume: 0.0,
            smt_verified_volume: 0.0,
        }
    }

    /// Add Tier 1 green (verified accessible) volume.
    pub fn add_green_volume(&mut self, volume: f64) {
        self.green_volume += volume.max(0.0);
    }

    /// Add Tier 1 red (provably inaccessible) volume.
    pub fn add_red_volume(&mut self, volume: f64) {
        self.red_volume += volume.max(0.0);
    }

    /// Add SMT-verified volume.
    pub fn add_smt_volume(&mut self, volume: f64) {
        self.smt_verified_volume += volume.max(0.0);
    }

    /// Verified fraction: (V_green + V_smt) / V_total.
    pub fn verified_fraction(&self) -> f64 {
        if self.total_volume <= 0.0 {
            return 0.0;
        }
        ((self.green_volume + self.smt_verified_volume) / self.total_volume).min(1.0)
    }

    /// Yellow (uncertain) fraction requiring sampling.
    pub fn yellow_fraction(&self) -> f64 {
        if self.total_volume <= 0.0 {
            return 1.0;
        }
        let known = self.green_volume + self.red_volume + self.smt_verified_volume;
        (1.0 - known / self.total_volume).max(0.0)
    }

    /// Effective unverified fraction ρ_s (excluding red since red is
    /// provably inaccessible and not a coverage concern).
    ///
    /// ρ_s = 1 - (V_green + V_smt) / (V_total - V_red)
    pub fn effective_unverified_fraction(&self) -> f64 {
        let effective_total = (self.total_volume - self.red_volume).max(1e-15);
        let verified = self.green_volume + self.smt_verified_volume;
        (1.0 - verified / effective_total).max(0.0).min(1.0)
    }

    /// Compute the effective ε for the combined certificate.
    ///
    /// ε_effective = ρ_s · ε_sampling
    ///
    /// Because sampling only needs to cover the unverified fraction,
    /// the sampling error is scaled down.
    pub fn effective_epsilon(&self, sampling_epsilon: f64) -> f64 {
        self.effective_unverified_fraction() * sampling_epsilon
    }

    /// Compute effective coverage κ combining verified + sampled.
    ///
    /// κ = V_green/V_accessible + ρ_s · κ_sampling
    ///
    /// where V_accessible = V_total - V_red.
    pub fn effective_kappa(&self, sampling_kappa: f64) -> f64 {
        let v_accessible = (self.total_volume - self.red_volume).max(1e-15);
        let verified_contribution = (self.green_volume + self.smt_verified_volume) / v_accessible;
        let sampling_contribution =
            self.effective_unverified_fraction() * sampling_kappa;
        (verified_contribution + sampling_contribution).min(1.0)
    }

    /// Summary of the volume model.
    pub fn summary(&self) -> VolumeModelSummary {
        VolumeModelSummary {
            total_volume: self.total_volume,
            green_volume: self.green_volume,
            red_volume: self.red_volume,
            smt_volume: self.smt_verified_volume,
            verified_fraction: self.verified_fraction(),
            yellow_fraction: self.yellow_fraction(),
            effective_unverified: self.effective_unverified_fraction(),
        }
    }
}

/// Summary of the volume-subtraction model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeModelSummary {
    pub total_volume: f64,
    pub green_volume: f64,
    pub red_volume: f64,
    pub smt_volume: f64,
    pub verified_fraction: f64,
    pub yellow_fraction: f64,
    pub effective_unverified: f64,
}

impl std::fmt::Display for VolumeModelSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Volume: total={:.6} green={:.6} red={:.6} smt={:.6} | \
             verified={:.4} yellow={:.4} ρ_s={:.4}",
            self.total_volume,
            self.green_volume,
            self.red_volume,
            self.smt_volume,
            self.verified_fraction,
            self.yellow_fraction,
            self.effective_unverified,
        )
    }
}

// ────────────────── Minimum Sample Computation ────────────────────────────

/// Compute the minimum number of samples needed across all strata
/// to achieve a target epsilon with confidence 1 - delta.
pub fn compute_minimum_samples(
    target_epsilon: f64,
    delta: f64,
    num_strata: usize,
    num_elements: usize,
    num_devices: usize,
) -> usize {
    if target_epsilon <= 0.0 || delta <= 0.0 {
        return usize::MAX;
    }
    let h = num_strata * num_elements * num_devices;
    let log_term = (2.0 * h as f64 / delta).ln();
    let per_stratum = (log_term / (2.0 * target_epsilon * target_epsilon)).ceil() as usize;
    num_strata * per_stratum.max(1)
}

/// Compute epsilon from total samples and parameters.
pub fn compute_epsilon(
    n_samples: usize,
    delta: f64,
    num_strata: usize,
    num_elements: usize,
    num_devices: usize,
) -> f64 {
    if n_samples == 0 || num_strata == 0 {
        return 1.0;
    }
    let n_per = n_samples / num_strata.max(1);
    if n_per == 0 {
        return 1.0;
    }
    let h = num_strata * num_elements * num_devices;
    let log_term = (2.0 * h as f64 / delta).ln();
    (log_term / (2.0 * n_per as f64)).sqrt().min(1.0)
}

/// Compute the Clopper-Pearson exact confidence interval for a proportion.
///
/// Given k successes in n trials, returns (lower, upper) bounds
/// at confidence level 1 - alpha using the normal approximation.
pub fn clopper_pearson_approx(k: usize, n: usize, alpha: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let p_hat = k as f64 / n as f64;
    let z = normal_quantile(1.0 - alpha / 2.0);
    let se = (p_hat * (1.0 - p_hat) / n as f64).sqrt();
    let lower = (p_hat - z * se).max(0.0);
    let upper = (p_hat + z * se).min(1.0);
    (lower, upper)
}

/// Approximate normal quantile (inverse CDF) using Abramowitz & Stegun.
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    // Rational approximation (Abramowitz & Stegun 26.2.23)
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let num = c0 + c1 * t + c2 * t * t;
    let den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    let x = t - num / den;

    if p < 0.5 { -x } else { x }
}

/// Wilson score interval for a proportion (better than Wald for small n).
pub fn wilson_interval(k: usize, n: usize, alpha: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let p_hat = k as f64 / n as f64;
    let z = normal_quantile(1.0 - alpha / 2.0);
    let z2 = z * z;
    let n_f = n as f64;

    let denom = 1.0 + z2 / n_f;
    let center = (p_hat + z2 / (2.0 * n_f)) / denom;
    let margin = z * (p_hat * (1.0 - p_hat) / n_f + z2 / (4.0 * n_f * n_f)).sqrt() / denom;

    ((center - margin).max(0.0), (center + margin).min(1.0))
}

// ────────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hoeffding_basic() {
        let bound = HoeffdingBound::new(0.05, 10, 5, 3);
        assert_eq!(bound.num_hypotheses(), 150);
        let eps = bound.epsilon(1000);
        assert!(eps > 0.0 && eps < 1.0);
    }

    #[test]
    fn test_epsilon_decreases_with_samples() {
        let bound = HoeffdingBound::new(0.05, 10, 5, 3);
        let eps_100 = bound.epsilon(100);
        let eps_1000 = bound.epsilon(1000);
        let eps_10000 = bound.epsilon(10000);
        assert!(eps_100 > eps_1000);
        assert!(eps_1000 > eps_10000);
    }

    #[test]
    fn test_minimum_samples() {
        let bound = HoeffdingBound::new(0.05, 10, 5, 3);
        let n = bound.min_samples(0.05);
        assert!(n > 0);
        let achieved_eps = bound.epsilon(n);
        assert!(achieved_eps <= 0.05 + 0.01); // allow small rounding
    }

    #[test]
    fn test_per_stratum_bound() {
        let bound = HoeffdingBound::new(0.05, 10, 5, 3);
        let eps_10 = bound.per_stratum_bound(10);
        let eps_100 = bound.per_stratum_bound(100);
        assert!(eps_10 > eps_100);
    }

    #[test]
    fn test_achieved_confidence() {
        let bound = HoeffdingBound::new(0.05, 10, 5, 3);
        let conf = bound.achieved_confidence(10000, 0.1);
        assert!(conf > 0.9);
    }

    #[test]
    fn test_zero_samples() {
        let bound = HoeffdingBound::new(0.05, 10, 5, 3);
        assert_eq!(bound.epsilon(0), 1.0);
        assert_eq!(bound.per_stratum_bound(0), 1.0);
    }

    #[test]
    fn test_stratified_bound() {
        let mut sb = StratifiedBound::new(0.05, 5, 3);
        sb.add_stratum(0.5, 100, 95, 5);
        sb.add_stratum(0.5, 100, 80, 20);

        let eps = sb.overall_epsilon();
        assert!(eps > 0.0 && eps < 1.0);

        let est = sb.estimated_coverage();
        assert!((est - 0.875).abs() < 0.01); // (0.95 + 0.80) / 2

        let lower = sb.coverage_lower_bound();
        let upper = sb.coverage_upper_bound();
        assert!(lower <= est);
        assert!(upper >= est);
    }

    #[test]
    fn test_stratified_bound_empty() {
        let sb = StratifiedBound::new(0.05, 5, 3);
        assert_eq!(sb.overall_epsilon(), 0.0); // sum of nothing
        assert_eq!(sb.estimated_coverage(), 0.0);
        assert_eq!(sb.total_samples(), 0);
    }

    #[test]
    fn test_volume_subtraction_basic() {
        let mut vsm = VolumeSubtractionModel::new(1.0);
        vsm.add_green_volume(0.3);
        vsm.add_red_volume(0.1);

        assert!((vsm.verified_fraction() - 0.3).abs() < 1e-10);
        assert!((vsm.yellow_fraction() - 0.6).abs() < 1e-10);

        // effective_unverified uses accessible volume (total - red)
        let rho = vsm.effective_unverified_fraction();
        // accessible = 0.9, verified = 0.3, so rho = 1 - 0.3/0.9
        assert!((rho - (1.0 - 0.3 / 0.9)).abs() < 1e-10);
    }

    #[test]
    fn test_volume_subtraction_with_smt() {
        let mut vsm = VolumeSubtractionModel::new(1.0);
        vsm.add_green_volume(0.4);
        vsm.add_red_volume(0.1);
        vsm.add_smt_volume(0.2);

        assert!((vsm.verified_fraction() - 0.6).abs() < 1e-10);
        // accessible = 0.9, verified = 0.6
        let rho = vsm.effective_unverified_fraction();
        assert!((rho - (1.0 - 0.6 / 0.9)).abs() < 1e-10);
    }

    #[test]
    fn test_effective_epsilon() {
        let mut vsm = VolumeSubtractionModel::new(1.0);
        vsm.add_green_volume(0.5);
        let eps_sampling = 0.1;
        let eps_eff = vsm.effective_epsilon(eps_sampling);
        assert!(eps_eff < eps_sampling);
        assert!((eps_eff - 0.5 * 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_effective_kappa() {
        let mut vsm = VolumeSubtractionModel::new(1.0);
        vsm.add_green_volume(0.5);
        let kappa = vsm.effective_kappa(0.9);
        assert!(kappa > 0.9);
    }

    #[test]
    fn test_compute_minimum_samples_fn() {
        let n = compute_minimum_samples(0.05, 0.05, 10, 5, 3);
        assert!(n > 0);
        let eps = compute_epsilon(n, 0.05, 10, 5, 3);
        assert!(eps <= 0.06);
    }

    #[test]
    fn test_clopper_pearson() {
        let (lower, upper) = clopper_pearson_approx(90, 100, 0.05);
        assert!(lower < 0.9);
        assert!(upper > 0.9);
        assert!(lower > 0.8);
        assert!(upper < 1.0);
    }

    #[test]
    fn test_wilson_interval() {
        let (lower, upper) = wilson_interval(90, 100, 0.05);
        assert!(lower < 0.9);
        assert!(upper > 0.9);
        assert!(lower > 0.8);
    }

    #[test]
    fn test_normal_quantile() {
        let z = normal_quantile(0.975);
        assert!((z - 1.96).abs() < 0.01);
        let z50 = normal_quantile(0.5);
        assert!(z50.abs() < 0.01);
    }

    #[test]
    fn test_volume_model_summary() {
        let mut vsm = VolumeSubtractionModel::new(1.0);
        vsm.add_green_volume(0.3);
        vsm.add_red_volume(0.1);
        let summary = vsm.summary();
        assert!((summary.total_volume - 1.0).abs() < 1e-10);
        assert!((summary.green_volume - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_per_hypothesis_delta() {
        let bound = HoeffdingBound::new(0.05, 10, 5, 3);
        let per = bound.per_hypothesis_delta();
        assert!((per - 0.05 / 150.0).abs() < 1e-15);
    }

    #[test]
    fn test_fully_verified_volume() {
        let mut vsm = VolumeSubtractionModel::new(1.0);
        vsm.add_green_volume(1.0);
        assert!((vsm.verified_fraction() - 1.0).abs() < 1e-10);
        assert!((vsm.effective_unverified_fraction() - 0.0).abs() < 1e-10);
        assert!((vsm.effective_epsilon(0.5) - 0.0).abs() < 1e-10);
    }
}
