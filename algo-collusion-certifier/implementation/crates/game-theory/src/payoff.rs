//! Payoff computation and analysis.
//!
//! Provides payoff profiles, comparisons, normalization, social welfare,
//! individual rationality, and interpolation between competitive and collusive outcomes.

use serde::{Deserialize, Serialize};
use shared_types::*;
use std::cmp::Ordering;

// ── Payoff Profile ──────────────────────────────────────────────────────────────────

/// Payoff profile for N players.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoffProfile {
    pub payoffs: Vec<f64>,
    pub label: Option<String>,
}

impl PayoffProfile {
    pub fn new(payoffs: Vec<f64>) -> Self {
        Self { payoffs, label: None }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    pub fn num_players(&self) -> usize { self.payoffs.len() }

    pub fn total(&self) -> f64 { self.payoffs.iter().sum() }

    pub fn player(&self, id: PlayerId) -> f64 {
        self.payoffs.get(id.0).copied().unwrap_or(0.0)
    }

    pub fn min_payoff(&self) -> f64 {
        self.payoffs.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn max_payoff(&self) -> f64 {
        self.payoffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Scale all payoffs by a constant factor.
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            payoffs: self.payoffs.iter().map(|&p| p * factor).collect(),
            label: self.label.clone(),
        }
    }

    /// Element-wise addition of two profiles.
    pub fn add(&self, other: &Self) -> Self {
        let payoffs = self.payoffs.iter().zip(&other.payoffs)
            .map(|(a, b)| a + b)
            .collect();
        Self { payoffs, label: None }
    }

    /// Element-wise subtraction.
    pub fn subtract(&self, other: &Self) -> Self {
        let payoffs = self.payoffs.iter().zip(&other.payoffs)
            .map(|(a, b)| a - b)
            .collect();
        Self { payoffs, label: None }
    }
}

// ── Payoff Comparison ───────────────────────────────────────────────────────────────

/// Ordering relationships between payoff profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DominanceRelation {
    /// Profile A Pareto dominates B (all >=, some >).
    Dominates,
    /// Profile A is Pareto dominated by B.
    IsDominated,
    /// Profiles are Pareto incomparable.
    Incomparable,
    /// Profiles are identical.
    Equal,
}

/// Compare payoff profiles.
pub struct PayoffComparison;

impl PayoffComparison {
    /// Check Pareto dominance: a dominates b if all a[i] >= b[i] and some a[i] > b[i].
    pub fn pareto_compare(a: &PayoffProfile, b: &PayoffProfile) -> DominanceRelation {
        if a.payoffs.len() != b.payoffs.len() {
            return DominanceRelation::Incomparable;
        }
        let mut all_ge = true;
        let mut all_le = true;
        let mut some_gt = false;
        let mut some_lt = false;

        for (ai, bi) in a.payoffs.iter().zip(&b.payoffs) {
            if *ai < *bi - 1e-12 { all_ge = false; some_lt = true; }
            if *ai > *bi + 1e-12 { all_le = false; some_gt = true; }
        }

        if all_ge && all_le { return DominanceRelation::Equal; }
        if all_ge && some_gt { return DominanceRelation::Dominates; }
        if all_le && some_lt { return DominanceRelation::IsDominated; }
        DominanceRelation::Incomparable
    }

    /// Find all Pareto-optimal profiles from a set.
    pub fn pareto_frontier(profiles: &[PayoffProfile]) -> Vec<usize> {
        let mut frontier = Vec::new();
        for (i, pi) in profiles.iter().enumerate() {
            let mut dominated = false;
            for (j, pj) in profiles.iter().enumerate() {
                if i == j { continue; }
                if Self::pareto_compare(pj, pi) == DominanceRelation::Dominates {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                frontier.push(i);
            }
        }
        frontier
    }
}

// ── Payoff Normalization ────────────────────────────────────────────────────────────

/// Normalize payoffs to a common scale.
pub struct PayoffNormalization;

impl PayoffNormalization {
    /// Min-max normalization to [0, 1] across a set of profiles.
    pub fn min_max(profiles: &[PayoffProfile]) -> Vec<PayoffProfile> {
        if profiles.is_empty() { return vec![]; }
        let n = profiles[0].num_players();
        let mut mins = vec![f64::INFINITY; n];
        let mut maxs = vec![f64::NEG_INFINITY; n];

        for p in profiles {
            for i in 0..n {
                mins[i] = mins[i].min(p.payoffs[i]);
                maxs[i] = maxs[i].max(p.payoffs[i]);
            }
        }

        profiles.iter().map(|p| {
            let normalized: Vec<f64> = p.payoffs.iter().enumerate().map(|(i, &v)| {
                let range = maxs[i] - mins[i];
                if range.abs() < 1e-12 { 0.5 } else { (v - mins[i]) / range }
            }).collect();
            PayoffProfile::new(normalized)
        }).collect()
    }

    /// Normalize relative to a reference profile (e.g., Nash equilibrium).
    pub fn relative_to(profiles: &[PayoffProfile], reference: &PayoffProfile) -> Vec<PayoffProfile> {
        profiles.iter().map(|p| {
            let normalized: Vec<f64> = p.payoffs.iter().zip(&reference.payoffs)
                .map(|(v, r)| {
                    if r.abs() > 1e-12 { v / r } else { if *v > 1e-12 { f64::INFINITY } else { 1.0 } }
                })
                .collect();
            PayoffProfile::new(normalized)
        }).collect()
    }

    /// Z-score normalization.
    pub fn z_score(profiles: &[PayoffProfile]) -> Vec<PayoffProfile> {
        if profiles.is_empty() { return vec![]; }
        let n = profiles[0].num_players();
        let k = profiles.len() as f64;

        let means: Vec<f64> = (0..n).map(|i| {
            profiles.iter().map(|p| p.payoffs[i]).sum::<f64>() / k
        }).collect();

        let stds: Vec<f64> = (0..n).map(|i| {
            let var = profiles.iter().map(|p| (p.payoffs[i] - means[i]).powi(2)).sum::<f64>() / k;
            var.sqrt()
        }).collect();

        profiles.iter().map(|p| {
            let normalized: Vec<f64> = p.payoffs.iter().enumerate().map(|(i, &v)| {
                if stds[i] > 1e-12 { (v - means[i]) / stds[i] } else { 0.0 }
            }).collect();
            PayoffProfile::new(normalized)
        }).collect()
    }
}

// ── Social Welfare ──────────────────────────────────────────────────────────────────

/// Social welfare computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WelfareType {
    Utilitarian,
    Egalitarian,
    NashProduct,
}

pub struct SocialWelfare;

impl SocialWelfare {
    /// Utilitarian welfare: sum of payoffs.
    pub fn utilitarian(profile: &PayoffProfile) -> f64 {
        profile.total()
    }

    /// Egalitarian welfare: minimum payoff (Rawlsian).
    pub fn egalitarian(profile: &PayoffProfile) -> f64 {
        profile.min_payoff()
    }

    /// Nash bargaining product: product of payoffs relative to disagreement point.
    pub fn nash_product(profile: &PayoffProfile, disagreement: &PayoffProfile) -> f64 {
        profile.payoffs.iter().zip(&disagreement.payoffs)
            .map(|(p, d)| (p - d).max(0.0))
            .product()
    }

    /// Compute welfare of a given type.
    pub fn compute(profile: &PayoffProfile, welfare_type: WelfareType, disagreement: Option<&PayoffProfile>) -> f64 {
        match welfare_type {
            WelfareType::Utilitarian => Self::utilitarian(profile),
            WelfareType::Egalitarian => Self::egalitarian(profile),
            WelfareType::NashProduct => {
                let default_disagreement = PayoffProfile::new(vec![0.0; profile.num_players()]);
                let d = disagreement.unwrap_or(&default_disagreement);
                Self::nash_product(profile, d)
            }
        }
    }

    /// Rank profiles by welfare.
    pub fn rank(profiles: &[PayoffProfile], welfare_type: WelfareType) -> Vec<(usize, f64)> {
        let mut ranked: Vec<(usize, f64)> = profiles.iter().enumerate()
            .map(|(i, p)| (i, Self::compute(p, welfare_type, None)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        ranked
    }
}

// ── Payoff Space ────────────────────────────────────────────────────────────────────

/// Enumeration of achievable payoffs for finite games.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoffSpace {
    pub profiles: Vec<PayoffProfile>,
    pub num_players: usize,
}

impl PayoffSpace {
    pub fn new(profiles: Vec<PayoffProfile>) -> Self {
        let n = profiles.first().map(|p| p.num_players()).unwrap_or(0);
        Self { profiles, num_players: n }
    }

    /// Enumerate payoffs from a discrete Bertrand game on a price grid.
    pub fn from_price_grid(config: &GameConfig, grid_points: usize) -> Self {
        let n = config.num_players;
        let (p_min, p_max) = (config.marginal_costs[0].0, 10.0);
        let step = if grid_points > 1 { (p_max - p_min) / (grid_points - 1) as f64 } else { 0.0 };

        let mut profiles = Vec::new();

        if n == 2 {
            for i in 0..grid_points {
                for j in 0..grid_points {
                    let p1 = p_min + step * i as f64;
                    let p2 = p_min + step * j as f64;
                    let payoffs = Self::compute_bertrand_payoffs(config, &[p1, p2]);
                    profiles.push(PayoffProfile::new(payoffs));
                }
            }
        }
        Self::new(profiles)
    }

    fn compute_bertrand_payoffs(config: &GameConfig, prices: &[f64]) -> Vec<f64> {
        match &config.demand_system {
            DemandSystem::Linear { max_quantity, slope } => {
                let n = prices.len();
                (0..n).map(|i| {
                    let mc = config.marginal_costs.get(i).map(|c| c.0).unwrap_or(1.0);
                    let others_avg: f64 = if n > 1 {
                        prices.iter().enumerate()
                            .filter(|(j, _)| *j != i)
                            .map(|(_, &p)| p)
                            .sum::<f64>() / (n - 1) as f64
                    } else { 0.0 };
                    let q = (max_quantity - slope * prices[i]).max(0.0);
                    (prices[i] - mc) * q
                }).collect()
            }
            _ => vec![0.0; prices.len()],
        }
    }

    /// Get the Pareto frontier.
    pub fn pareto_frontier(&self) -> Vec<&PayoffProfile> {
        let indices = PayoffComparison::pareto_frontier(&self.profiles);
        indices.iter().map(|&i| &self.profiles[i]).collect()
    }

    /// Find the profile with maximum utilitarian welfare.
    pub fn max_welfare_profile(&self) -> Option<&PayoffProfile> {
        self.profiles.iter().max_by(|a, b| {
            a.total().partial_cmp(&b.total()).unwrap_or(Ordering::Equal)
        })
    }
}

// ── Individual Rationality ──────────────────────────────────────────────────────────

/// Check individual rationality of payoff profiles.
pub struct IndividualRationality;

impl IndividualRationality {
    /// Check if all players receive at least their minimax value.
    pub fn check(profile: &PayoffProfile, minimax_values: &[f64]) -> bool {
        profile.payoffs.iter().zip(minimax_values)
            .all(|(&p, &mm)| p >= mm - 1e-10)
    }

    /// Compute the individual rationality surplus for each player.
    pub fn surplus(profile: &PayoffProfile, minimax_values: &[f64]) -> Vec<f64> {
        profile.payoffs.iter().zip(minimax_values)
            .map(|(&p, &mm)| p - mm)
            .collect()
    }

    /// Find the player with the smallest surplus (binding constraint).
    pub fn binding_player(profile: &PayoffProfile, minimax_values: &[f64]) -> (usize, f64) {
        let surpluses = Self::surplus(profile, minimax_values);
        surpluses.iter().enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, &s)| (i, s))
            .unwrap_or((0, 0.0))
    }
}

// ── Cooperative Payoff ──────────────────────────────────────────────────────────────

/// Joint profit maximization (cooperative/collusive payoff).
pub struct CooperativePayoff;

impl CooperativePayoff {
    /// Compute the joint profit-maximizing prices for a Bertrand game.
    pub fn compute(config: &GameConfig) -> PayoffProfile {
        match &config.demand_system {
            DemandSystem::Linear { max_quantity, slope } => {
                let n = config.num_players;
                let mc = config.marginal_costs[0].0;
                let total_slope = *slope;
                let p_m = if total_slope.abs() > 1e-12 {
                    (max_quantity + total_slope * mc) / (2.0 * total_slope)
                } else {
                    mc * 2.0
                };
                let payoffs: Vec<f64> = (0..n).map(|_i| {
                    let q = (max_quantity - slope * p_m).max(0.0);
                    (p_m - mc) * q
                }).collect();
                PayoffProfile::new(payoffs).with_label("cooperative")
            }
            _ => PayoffProfile::new(vec![0.0; config.num_players]),
        }
    }

    /// Compute the efficiency of an observed outcome relative to the cooperative optimum.
    pub fn efficiency(observed: &PayoffProfile, cooperative: &PayoffProfile) -> f64 {
        let coop_total = cooperative.total();
        if coop_total.abs() < 1e-12 { return 1.0; }
        observed.total() / coop_total
    }
}

// ── Payoff Interpolation ────────────────────────────────────────────────────────────

/// Interpolate between competitive and collusive payoffs.
pub struct PayoffInterpolation;

impl PayoffInterpolation {
    /// Linear interpolation: result = (1-alpha)*competitive + alpha*collusive.
    pub fn linear(competitive: &PayoffProfile, collusive: &PayoffProfile, alpha: f64) -> PayoffProfile {
        let alpha = alpha.clamp(0.0, 1.0);
        let payoffs: Vec<f64> = competitive.payoffs.iter().zip(&collusive.payoffs)
            .map(|(c, m)| (1.0 - alpha) * c + alpha * m)
            .collect();
        PayoffProfile::new(payoffs)
    }

    /// Find the alpha that matches an observed payoff profile (least squares).
    pub fn infer_alpha(competitive: &PayoffProfile, collusive: &PayoffProfile, observed: &PayoffProfile) -> f64 {
        let mut numer = 0.0;
        let mut denom = 0.0;
        for i in 0..competitive.num_players() {
            let c = competitive.payoffs[i];
            let m = collusive.payoffs[i];
            let o = observed.payoffs[i];
            numer += (o - c) * (m - c);
            denom += (m - c) * (m - c);
        }
        if denom.abs() < 1e-12 { return 0.0; }
        (numer / denom).clamp(0.0, 1.0)
    }

    /// Compute the "collusion trajectory": sequence of alpha values over time.
    pub fn trajectory(
        competitive: &PayoffProfile,
        collusive: &PayoffProfile,
        observed_sequence: &[PayoffProfile],
    ) -> Vec<f64> {
        observed_sequence.iter()
            .map(|obs| Self::infer_alpha(competitive, collusive, obs))
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payoff_profile_basic() {
        let pp = PayoffProfile::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(pp.num_players(), 3);
        assert!((pp.total() - 6.0).abs() < 1e-10);
        assert!((pp.min_payoff() - 1.0).abs() < 1e-10);
        assert!((pp.max_payoff() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_payoff_profile_scale() {
        let pp = PayoffProfile::new(vec![2.0, 4.0]);
        let scaled = pp.scale(0.5);
        assert!((scaled.payoffs[0] - 1.0).abs() < 1e-10);
        assert!((scaled.payoffs[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_payoff_profile_add() {
        let a = PayoffProfile::new(vec![1.0, 2.0]);
        let b = PayoffProfile::new(vec![3.0, 4.0]);
        let sum = a.add(&b);
        assert!((sum.payoffs[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_pareto_dominance() {
        let a = PayoffProfile::new(vec![3.0, 4.0]);
        let b = PayoffProfile::new(vec![2.0, 3.0]);
        assert_eq!(PayoffComparison::pareto_compare(&a, &b), DominanceRelation::Dominates);
    }

    #[test]
    fn test_pareto_incomparable() {
        let a = PayoffProfile::new(vec![3.0, 2.0]);
        let b = PayoffProfile::new(vec![2.0, 3.0]);
        assert_eq!(PayoffComparison::pareto_compare(&a, &b), DominanceRelation::Incomparable);
    }

    #[test]
    fn test_pareto_equal() {
        let a = PayoffProfile::new(vec![3.0, 3.0]);
        let b = PayoffProfile::new(vec![3.0, 3.0]);
        assert_eq!(PayoffComparison::pareto_compare(&a, &b), DominanceRelation::Equal);
    }

    #[test]
    fn test_pareto_frontier() {
        let profiles = vec![
            PayoffProfile::new(vec![1.0, 4.0]),
            PayoffProfile::new(vec![3.0, 3.0]),
            PayoffProfile::new(vec![4.0, 1.0]),
            PayoffProfile::new(vec![2.0, 2.0]), // dominated
        ];
        let frontier = PayoffComparison::pareto_frontier(&profiles);
        assert_eq!(frontier.len(), 3); // indices 0, 1, 2
    }

    #[test]
    fn test_min_max_normalization() {
        let profiles = vec![
            PayoffProfile::new(vec![1.0, 2.0]),
            PayoffProfile::new(vec![3.0, 4.0]),
        ];
        let normalized = PayoffNormalization::min_max(&profiles);
        assert!((normalized[0].payoffs[0] - 0.0).abs() < 1e-10);
        assert!((normalized[1].payoffs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_utilitarian_welfare() {
        let pp = PayoffProfile::new(vec![3.0, 4.0, 5.0]);
        assert!((SocialWelfare::utilitarian(&pp) - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_egalitarian_welfare() {
        let pp = PayoffProfile::new(vec![3.0, 1.0, 5.0]);
        assert!((SocialWelfare::egalitarian(&pp) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nash_product() {
        let pp = PayoffProfile::new(vec![3.0, 4.0]);
        let d = PayoffProfile::new(vec![1.0, 1.0]);
        let product = SocialWelfare::nash_product(&pp, &d);
        assert!((product - 6.0).abs() < 1e-10); // (3-1)*(4-1) = 6
    }

    #[test]
    fn test_welfare_ranking() {
        let profiles = vec![
            PayoffProfile::new(vec![1.0, 1.0]),
            PayoffProfile::new(vec![3.0, 3.0]),
            PayoffProfile::new(vec![2.0, 2.0]),
        ];
        let ranked = SocialWelfare::rank(&profiles, WelfareType::Utilitarian);
        assert_eq!(ranked[0].0, 1); // (3,3) = 6 is highest
    }

    #[test]
    fn test_individual_rationality_check() {
        let pp = PayoffProfile::new(vec![2.0, 3.0]);
        assert!(IndividualRationality::check(&pp, &[1.0, 1.0]));
        assert!(!IndividualRationality::check(&pp, &[1.0, 4.0]));
    }

    #[test]
    fn test_individual_rationality_surplus() {
        let pp = PayoffProfile::new(vec![3.0, 5.0]);
        let surplus = IndividualRationality::surplus(&pp, &[1.0, 2.0]);
        assert!((surplus[0] - 2.0).abs() < 1e-10);
        assert!((surplus[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cooperative_payoff() {
        let config = GameConfig::symmetric(
            MarketType::Bertrand,
            DemandSystem::Linear { max_quantity: 10.0, slope: 1.0 },
            2, 0.95, Cost(1.0), 1000,
        );
        let coop = CooperativePayoff::compute(&config);
        assert!(coop.total() > 0.0);
    }

    #[test]
    fn test_linear_interpolation() {
        let comp = PayoffProfile::new(vec![1.0, 1.0]);
        let coll = PayoffProfile::new(vec![3.0, 3.0]);
        let mid = PayoffInterpolation::linear(&comp, &coll, 0.5);
        assert!((mid.payoffs[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_infer_alpha() {
        let comp = PayoffProfile::new(vec![1.0, 1.0]);
        let coll = PayoffProfile::new(vec![3.0, 3.0]);
        let obs = PayoffProfile::new(vec![2.0, 2.0]);
        let alpha = PayoffInterpolation::infer_alpha(&comp, &coll, &obs);
        assert!((alpha - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_payoff_space() {
        let config = GameConfig::symmetric(
            MarketType::Bertrand,
            DemandSystem::Linear { max_quantity: 10.0, slope: 1.0 },
            2, 0.95, Cost(1.0), 1000,
        );
        let space = PayoffSpace::from_price_grid(&config, 5);
        assert!(!space.profiles.is_empty());
    }

    #[test]
    fn test_collusion_trajectory() {
        let comp = PayoffProfile::new(vec![1.0, 1.0]);
        let coll = PayoffProfile::new(vec![3.0, 3.0]);
        let observations = vec![
            PayoffProfile::new(vec![1.0, 1.0]),
            PayoffProfile::new(vec![1.5, 1.5]),
            PayoffProfile::new(vec![2.5, 2.5]),
        ];
        let traj = PayoffInterpolation::trajectory(&comp, &coll, &observations);
        assert_eq!(traj.len(), 3);
        assert!((traj[0] - 0.0).abs() < 1e-10);
        assert!((traj[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_cooperative_efficiency() {
        let observed = PayoffProfile::new(vec![2.0, 2.0]);
        let coop = PayoffProfile::new(vec![4.0, 4.0]);
        let eff = CooperativePayoff::efficiency(&observed, &coop);
        assert!((eff - 0.5).abs() < 1e-10);
    }
}
