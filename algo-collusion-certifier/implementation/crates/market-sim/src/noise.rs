//! Noise and randomness models for market simulation.
//!
//! - [`DemandShock`]: additive or multiplicative Gaussian demand noise
//! - [`CostShock`]: random cost perturbations
//! - [`ObservationNoise`]: imperfect monitoring of opponents' actions
//! - [`SignalExtraction`]: filtering noisy observations
//! - [`NoiseCalibration`]: setting appropriate noise levels
//! - AR(1) demand shocks for persistent noise
//! - Common random numbers for variance reduction

use crate::types::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

// ════════════════════════════════════════════════════════════════════════════
// DemandShock
// ════════════════════════════════════════════════════════════════════════════

/// Type of demand shock.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ShockType {
    /// Q_i = Q_i + ε  where ε ~ N(0, σ²)
    Additive,
    /// Q_i = Q_i * (1 + ε)  where ε ~ N(0, σ²)
    Multiplicative,
}

/// Demand shock model perturbing quantities after demand computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandShock {
    pub shock_type: ShockType,
    pub std_dev: f64,
    pub per_player: bool,
    seed: u64,
    #[serde(skip)]
    rng: Option<StdRng>,
}

impl DemandShock {
    pub fn new(shock_type: ShockType, std_dev: f64, per_player: bool, seed: u64) -> Self {
        Self {
            shock_type,
            std_dev,
            per_player,
            seed,
            rng: Some(StdRng::seed_from_u64(seed)),
        }
    }

    pub fn additive(std_dev: f64, seed: u64) -> Self {
        Self::new(ShockType::Additive, std_dev, true, seed)
    }

    pub fn multiplicative(std_dev: f64, seed: u64) -> Self {
        Self::new(ShockType::Multiplicative, std_dev, true, seed)
    }

    fn ensure_rng(&mut self) {
        if self.rng.is_none() {
            self.rng = Some(StdRng::seed_from_u64(self.seed));
        }
    }

    /// Apply demand shock to a vector of quantities.
    pub fn apply(&mut self, quantities: &mut [f64]) {
        self.ensure_rng();
        let rng = self.rng.as_mut().unwrap();
        let dist = Normal::new(0.0, self.std_dev).unwrap();

        if self.per_player {
            for q in quantities.iter_mut() {
                let eps = dist.sample(rng);
                match self.shock_type {
                    ShockType::Additive => *q = (*q + eps).max(0.0),
                    ShockType::Multiplicative => *q = (*q * (1.0 + eps)).max(0.0),
                }
            }
        } else {
            let eps = dist.sample(rng);
            for q in quantities.iter_mut() {
                match self.shock_type {
                    ShockType::Additive => *q = (*q + eps).max(0.0),
                    ShockType::Multiplicative => *q = (*q * (1.0 + eps)).max(0.0),
                }
            }
        }
    }

    /// Reset the RNG to the original seed.
    pub fn reset(&mut self) {
        self.rng = Some(StdRng::seed_from_u64(self.seed));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// CostShock
// ════════════════════════════════════════════════════════════════════════════

/// Cost shock model perturbing marginal costs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostShock {
    pub std_dev: f64,
    pub correlated: bool,
    seed: u64,
    #[serde(skip)]
    rng: Option<StdRng>,
}

impl CostShock {
    pub fn new(std_dev: f64, correlated: bool, seed: u64) -> Self {
        Self {
            std_dev,
            correlated,
            seed,
            rng: Some(StdRng::seed_from_u64(seed)),
        }
    }

    fn ensure_rng(&mut self) {
        if self.rng.is_none() {
            self.rng = Some(StdRng::seed_from_u64(self.seed));
        }
    }

    /// Apply cost shock: returns perturbed marginal costs.
    pub fn apply(&mut self, base_costs: &[f64]) -> Vec<f64> {
        self.ensure_rng();
        let rng = self.rng.as_mut().unwrap();
        let dist = Normal::new(0.0, self.std_dev).unwrap();

        if self.correlated {
            let eps = dist.sample(rng);
            base_costs.iter().map(|&c| (c + eps).max(0.0)).collect()
        } else {
            base_costs
                .iter()
                .map(|&c| {
                    let eps = dist.sample(rng);
                    (c + eps).max(0.0)
                })
                .collect()
        }
    }

    pub fn reset(&mut self) {
        self.rng = Some(StdRng::seed_from_u64(self.seed));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ObservationNoise
// ════════════════════════════════════════════════════════════════════════════

/// Imperfect monitoring: players observe noisy signals of opponents' actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationNoise {
    pub std_dev: f64,
    seed: u64,
    #[serde(skip)]
    rng: Option<StdRng>,
}

impl ObservationNoise {
    pub fn new(std_dev: f64, seed: u64) -> Self {
        Self {
            std_dev,
            seed,
            rng: Some(StdRng::seed_from_u64(seed)),
        }
    }

    fn ensure_rng(&mut self) {
        if self.rng.is_none() {
            self.rng = Some(StdRng::seed_from_u64(self.seed));
        }
    }

    /// Generate noisy observations of the true actions.
    /// Returns a matrix: observed[observer][observed_player] = noisy value.
    pub fn observe(
        &mut self,
        true_actions: &[f64],
        num_players: usize,
    ) -> Vec<Vec<f64>> {
        self.ensure_rng();
        let rng = self.rng.as_mut().unwrap();
        let dist = Normal::new(0.0, self.std_dev).unwrap();

        let mut observations = Vec::with_capacity(num_players);
        for _observer in 0..num_players {
            let obs: Vec<f64> = true_actions
                .iter()
                .map(|&a| a + dist.sample(rng))
                .collect();
            observations.push(obs);
        }
        observations
    }

    /// Generate a single player's noisy observation of opponents.
    pub fn observe_others(
        &mut self,
        observer: PlayerId,
        true_actions: &[f64],
    ) -> Vec<f64> {
        self.ensure_rng();
        let rng = self.rng.as_mut().unwrap();
        let dist = Normal::new(0.0, self.std_dev).unwrap();

        true_actions
            .iter()
            .enumerate()
            .map(|(j, &a)| {
                if j == observer {
                    a // Own action observed perfectly
                } else {
                    a + dist.sample(rng)
                }
            })
            .collect()
    }

    pub fn reset(&mut self) {
        self.rng = Some(StdRng::seed_from_u64(self.seed));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SignalExtraction
// ════════════════════════════════════════════════════════════════════════════

/// Signal extraction from noisy observations using exponential smoothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalExtraction {
    /// Smoothing parameter α ∈ (0, 1]. Higher = more weight on recent.
    pub alpha: f64,
    /// Current estimates for each player's action.
    pub estimates: Vec<f64>,
    pub initialized: bool,
}

impl SignalExtraction {
    pub fn new(alpha: f64, num_players: usize) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
        Self {
            alpha,
            estimates: vec![0.0; num_players],
            initialized: false,
        }
    }

    /// Update estimates with new noisy observations.
    pub fn update(&mut self, observations: &[f64]) -> &[f64] {
        if !self.initialized {
            self.estimates = observations.to_vec();
            self.initialized = true;
        } else {
            for (i, &obs) in observations.iter().enumerate() {
                self.estimates[i] = self.alpha * obs + (1.0 - self.alpha) * self.estimates[i];
            }
        }
        &self.estimates
    }

    /// Get current filtered estimates.
    pub fn current_estimates(&self) -> &[f64] {
        &self.estimates
    }

    /// Reset the filter.
    pub fn reset(&mut self) {
        self.estimates.fill(0.0);
        self.initialized = false;
    }

    /// Compute the signal-to-noise ratio given true variance and noise variance.
    pub fn snr(signal_variance: f64, noise_variance: f64) -> f64 {
        if noise_variance.abs() < 1e-15 {
            return f64::INFINITY;
        }
        signal_variance / noise_variance
    }

    /// Optimal smoothing parameter (Kalman-like) for given signal and noise variances.
    /// For a random walk + noise model: α_opt ≈ σ_signal / (σ_signal + σ_noise).
    pub fn optimal_alpha(signal_std: f64, noise_std: f64) -> f64 {
        if signal_std + noise_std < 1e-15 {
            return 0.5;
        }
        signal_std / (signal_std + noise_std)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// NoiseCalibration
// ════════════════════════════════════════════════════════════════════════════

/// Calibration of noise levels relative to market parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseCalibration {
    /// Demand noise as fraction of base demand.
    pub demand_noise_fraction: f64,
    /// Cost noise as fraction of base cost.
    pub cost_noise_fraction: f64,
    /// Observation noise as fraction of price range.
    pub observation_noise_fraction: f64,
    /// Reference price level for scaling.
    pub reference_price: f64,
    /// Reference demand level for scaling.
    pub reference_demand: f64,
}

impl NoiseCalibration {
    pub fn new(
        demand_noise_fraction: f64,
        cost_noise_fraction: f64,
        observation_noise_fraction: f64,
        reference_price: f64,
        reference_demand: f64,
    ) -> Self {
        Self {
            demand_noise_fraction,
            cost_noise_fraction,
            observation_noise_fraction,
            reference_price,
            reference_demand,
        }
    }

    /// Low noise: 1% of reference levels.
    pub fn low(reference_price: f64, reference_demand: f64) -> Self {
        Self::new(0.01, 0.01, 0.01, reference_price, reference_demand)
    }

    /// Medium noise: 5% of reference levels.
    pub fn medium(reference_price: f64, reference_demand: f64) -> Self {
        Self::new(0.05, 0.05, 0.05, reference_price, reference_demand)
    }

    /// High noise: 15% of reference levels.
    pub fn high(reference_price: f64, reference_demand: f64) -> Self {
        Self::new(0.15, 0.10, 0.10, reference_price, reference_demand)
    }

    /// Create a DemandShock from calibration.
    pub fn create_demand_shock(&self, seed: u64) -> DemandShock {
        let std_dev = self.demand_noise_fraction * self.reference_demand;
        DemandShock::additive(std_dev, seed)
    }

    /// Create a CostShock from calibration.
    pub fn create_cost_shock(&self, seed: u64) -> CostShock {
        let std_dev = self.cost_noise_fraction * self.reference_price;
        CostShock::new(std_dev, false, seed)
    }

    /// Create an ObservationNoise from calibration.
    pub fn create_observation_noise(&self, seed: u64) -> ObservationNoise {
        let std_dev = self.observation_noise_fraction * self.reference_price;
        ObservationNoise::new(std_dev, seed)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// AR(1) demand shocks
// ════════════════════════════════════════════════════════════════════════════

/// AR(1) demand shock: ε_t = ρ * ε_{t-1} + η_t,  η ~ N(0, σ_η²).
///
/// The persistence parameter ρ ∈ [0, 1) controls autocorrelation.
/// Unconditional variance: Var(ε) = σ_η² / (1 - ρ²).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AR1DemandShock {
    pub rho: f64,
    pub innovation_std: f64,
    pub shock_type: ShockType,
    pub current_shock: Vec<f64>,
    seed: u64,
    #[serde(skip)]
    rng: Option<StdRng>,
}

impl AR1DemandShock {
    pub fn new(
        rho: f64,
        innovation_std: f64,
        shock_type: ShockType,
        num_players: usize,
        seed: u64,
    ) -> Self {
        assert!(rho.abs() < 1.0, "AR(1) rho must have |ρ| < 1");
        Self {
            rho,
            innovation_std,
            shock_type,
            current_shock: vec![0.0; num_players],
            seed,
            rng: Some(StdRng::seed_from_u64(seed)),
        }
    }

    fn ensure_rng(&mut self) {
        if self.rng.is_none() {
            self.rng = Some(StdRng::seed_from_u64(self.seed));
        }
    }

    /// Advance the AR(1) process and apply to quantities.
    pub fn apply(&mut self, quantities: &mut [f64]) {
        self.ensure_rng();
        let rng = self.rng.as_mut().unwrap();
        let dist = Normal::new(0.0, self.innovation_std).unwrap();

        for (i, q) in quantities.iter_mut().enumerate() {
            let eta = dist.sample(rng);
            self.current_shock[i] = self.rho * self.current_shock[i] + eta;
            match self.shock_type {
                ShockType::Additive => *q = (*q + self.current_shock[i]).max(0.0),
                ShockType::Multiplicative => *q = (*q * (1.0 + self.current_shock[i])).max(0.0),
            }
        }
    }

    /// Unconditional standard deviation of the AR(1) process.
    pub fn unconditional_std(&self) -> f64 {
        self.innovation_std / (1.0 - self.rho * self.rho).sqrt()
    }

    /// Autocorrelation at lag k: ρ^k.
    pub fn autocorrelation(&self, lag: usize) -> f64 {
        self.rho.powi(lag as i32)
    }

    pub fn reset(&mut self) {
        self.current_shock.fill(0.0);
        self.rng = Some(StdRng::seed_from_u64(self.seed));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Common Random Numbers (CRN)
// ════════════════════════════════════════════════════════════════════════════

/// Common Random Numbers generator for variance reduction in counterfactual comparison.
///
/// Ensures that two simulations use the same sequence of random shocks,
/// so differences in outcomes are due solely to algorithmic differences.
#[derive(Debug, Clone)]
pub struct CommonRandomNumbers {
    #[allow(dead_code)]
    seed: u64,
    num_players: usize,
    // Pre-generated shock sequences
    demand_shocks: Vec<Vec<f64>>,
    cost_shocks: Vec<Vec<f64>>,
    current_round: usize,
}

impl CommonRandomNumbers {
    pub fn new(seed: u64, num_players: usize, num_rounds: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0, 1.0).unwrap();

        let demand_shocks: Vec<Vec<f64>> = (0..num_rounds)
            .map(|_| (0..num_players).map(|_| dist.sample(&mut rng)).collect())
            .collect();

        let cost_shocks: Vec<Vec<f64>> = (0..num_rounds)
            .map(|_| (0..num_players).map(|_| dist.sample(&mut rng)).collect())
            .collect();

        Self {
            seed,
            num_players,
            demand_shocks,
            cost_shocks,
            current_round: 0,
        }
    }

    /// Get the demand shock for the current round (scaled by std_dev).
    pub fn demand_shock(&self, round: usize, std_dev: f64) -> Vec<f64> {
        if round < self.demand_shocks.len() {
            self.demand_shocks[round].iter().map(|&z| z * std_dev).collect()
        } else {
            vec![0.0; self.num_players]
        }
    }

    /// Get the cost shock for the current round (scaled by std_dev).
    pub fn cost_shock(&self, round: usize, std_dev: f64) -> Vec<f64> {
        if round < self.cost_shocks.len() {
            self.cost_shocks[round].iter().map(|&z| z * std_dev).collect()
        } else {
            vec![0.0; self.num_players]
        }
    }

    /// Apply demand shocks to quantities for the given round.
    pub fn apply_demand_shock(
        &self,
        round: usize,
        quantities: &mut [f64],
        std_dev: f64,
        shock_type: ShockType,
    ) {
        let shocks = self.demand_shock(round, std_dev);
        for (i, q) in quantities.iter_mut().enumerate() {
            match shock_type {
                ShockType::Additive => *q = (*q + shocks[i]).max(0.0),
                ShockType::Multiplicative => *q = (*q * (1.0 + shocks[i])).max(0.0),
            }
        }
    }

    /// Reset the round counter.
    pub fn reset(&mut self) {
        self.current_round = 0;
    }

    pub fn num_rounds_available(&self) -> usize {
        self.demand_shocks.len()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demand_shock_additive() {
        let mut shock = DemandShock::additive(0.5, 42);
        let mut q = vec![10.0, 10.0];
        shock.apply(&mut q);
        // Should be perturbed but still near 10
        assert!(q[0] > 5.0 && q[0] < 15.0);
        assert!(q[1] > 5.0 && q[1] < 15.0);
    }

    #[test]
    fn test_demand_shock_multiplicative() {
        let mut shock = DemandShock::multiplicative(0.1, 42);
        let mut q = vec![10.0, 10.0];
        shock.apply(&mut q);
        assert!(q[0] > 5.0 && q[0] < 15.0);
    }

    #[test]
    fn test_demand_shock_non_negative() {
        let mut shock = DemandShock::additive(100.0, 42);
        let mut q = vec![1.0, 1.0];
        shock.apply(&mut q);
        assert!(q[0] >= 0.0);
        assert!(q[1] >= 0.0);
    }

    #[test]
    fn test_demand_shock_reset() {
        let mut s1 = DemandShock::additive(1.0, 42);
        let mut s2 = DemandShock::additive(1.0, 42);
        let mut q1 = vec![10.0, 10.0];
        let mut q2 = vec![10.0, 10.0];
        s1.apply(&mut q1);
        s2.apply(&mut q2);
        assert!((q1[0] - q2[0]).abs() < 1e-10);
    }

    #[test]
    fn test_cost_shock() {
        let mut shock = CostShock::new(0.5, false, 42);
        let costs = shock.apply(&[1.0, 2.0]);
        assert_eq!(costs.len(), 2);
        assert!(costs[0] >= 0.0);
    }

    #[test]
    fn test_cost_shock_correlated() {
        let mut shock = CostShock::new(0.5, true, 42);
        let costs = shock.apply(&[1.0, 1.0]);
        // Correlated: same shock to both
        // Due to the implementation, both get the same epsilon
        assert!((costs[0] - costs[1]).abs() < 1e-10);
    }

    #[test]
    fn test_observation_noise() {
        let mut noise = ObservationNoise::new(0.5, 42);
        let obs = noise.observe(&[3.0, 4.0], 2);
        assert_eq!(obs.len(), 2);
        assert_eq!(obs[0].len(), 2);
        // Observations should be noisy
        assert!((obs[0][0] - 3.0).abs() < 5.0);
    }

    #[test]
    fn test_observation_noise_others() {
        let mut noise = ObservationNoise::new(0.5, 42);
        let obs = noise.observe_others(0, &[3.0, 4.0]);
        // Own action observed perfectly
        assert!((obs[0] - 3.0).abs() < 1e-10);
        // Other's action noisy
        assert!((obs[1] - 4.0).abs() < 5.0);
    }

    #[test]
    fn test_signal_extraction() {
        let mut se = SignalExtraction::new(0.3, 2);
        se.update(&[5.0, 5.0]);
        assert!((se.estimates[0] - 5.0).abs() < 1e-10); // first = raw
        se.update(&[6.0, 6.0]);
        // Smoothed: 0.3*6 + 0.7*5 = 5.3
        assert!((se.estimates[0] - 5.3).abs() < 1e-10);
    }

    #[test]
    fn test_signal_extraction_snr() {
        let snr = SignalExtraction::snr(4.0, 1.0);
        assert!((snr - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_signal_extraction_optimal_alpha() {
        let alpha = SignalExtraction::optimal_alpha(2.0, 2.0);
        assert!((alpha - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_noise_calibration_low() {
        let cal = NoiseCalibration::low(3.0, 7.0);
        let shock = cal.create_demand_shock(42);
        assert!((shock.std_dev - 0.07).abs() < 1e-10);
    }

    #[test]
    fn test_noise_calibration_medium() {
        let cal = NoiseCalibration::medium(3.0, 7.0);
        let shock = cal.create_demand_shock(42);
        assert!((shock.std_dev - 0.35).abs() < 1e-10);
    }

    #[test]
    fn test_ar1_demand_shock() {
        let mut shock = AR1DemandShock::new(0.8, 0.5, ShockType::Additive, 2, 42);
        let mut q = vec![10.0, 10.0];
        shock.apply(&mut q);
        assert!(q[0] >= 0.0);
        // Apply again - should show persistence
        let mut q2 = vec![10.0, 10.0];
        shock.apply(&mut q2);
        // Hard to test persistence exactly, but at least verify it runs
    }

    #[test]
    fn test_ar1_unconditional_std() {
        let shock = AR1DemandShock::new(0.5, 1.0, ShockType::Additive, 2, 42);
        let std = shock.unconditional_std();
        // σ_unconditional = 1.0 / sqrt(1 - 0.25) = 1.0 / sqrt(0.75) ≈ 1.1547
        assert!((std - 1.0 / 0.75_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_ar1_autocorrelation() {
        let shock = AR1DemandShock::new(0.8, 1.0, ShockType::Additive, 2, 42);
        assert!((shock.autocorrelation(0) - 1.0).abs() < 1e-10);
        assert!((shock.autocorrelation(1) - 0.8).abs() < 1e-10);
        assert!((shock.autocorrelation(2) - 0.64).abs() < 1e-10);
    }

    #[test]
    fn test_common_random_numbers() {
        let crn = CommonRandomNumbers::new(42, 2, 100);
        let s1 = crn.demand_shock(0, 1.0);
        let s2 = crn.demand_shock(0, 1.0);
        // Same round, same shocks
        assert!((s1[0] - s2[0]).abs() < 1e-10);
    }

    #[test]
    fn test_crn_scaling() {
        let crn = CommonRandomNumbers::new(42, 2, 100);
        let s1 = crn.demand_shock(0, 1.0);
        let s2 = crn.demand_shock(0, 2.0);
        // s2 = 2 * s1
        assert!((s2[0] - 2.0 * s1[0]).abs() < 1e-10);
    }

    #[test]
    fn test_crn_apply() {
        let crn = CommonRandomNumbers::new(42, 2, 100);
        let mut q1 = vec![10.0, 10.0];
        let mut q2 = vec![10.0, 10.0];
        crn.apply_demand_shock(0, &mut q1, 0.5, ShockType::Additive);
        crn.apply_demand_shock(0, &mut q2, 0.5, ShockType::Additive);
        // Same shocks applied
        assert!((q1[0] - q2[0]).abs() < 1e-10);
    }

    #[test]
    fn test_crn_different_rounds() {
        let crn = CommonRandomNumbers::new(42, 2, 100);
        let s1 = crn.demand_shock(0, 1.0);
        let s2 = crn.demand_shock(1, 1.0);
        // Different rounds should (almost certainly) have different shocks
        assert!((s1[0] - s2[0]).abs() > 1e-15 || (s1[1] - s2[1]).abs() > 1e-15);
    }

    #[test]
    fn test_noise_calibration_create_all() {
        let cal = NoiseCalibration::high(5.0, 10.0);
        let ds = cal.create_demand_shock(1);
        let cs = cal.create_cost_shock(2);
        let on = cal.create_observation_noise(3);
        assert!(ds.std_dev > 0.0);
        assert!(cs.std_dev > 0.0);
        assert!(on.std_dev > 0.0);
    }
}
