//! Core domain types for the CollusionProof system.
//!
//! Contains newtype wrappers for numeric values, market types, game
//! configurations, algorithm descriptors, trajectory types with
//! phantom-tagged segments, and simulation configuration.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Macro for float-based newtype wrappers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

macro_rules! define_float_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize, Default)]
        pub struct $name(pub f64);

        impl $name {
            pub const ZERO: Self = $name(0.0);
            pub const ONE: Self = $name(1.0);

            #[inline] pub fn new(value: f64) -> Self { $name(value) }
            #[inline] pub fn value(self) -> f64 { self.0 }
            #[inline] pub fn abs(self) -> Self { $name(self.0.abs()) }
            #[inline] pub fn max(self, other: Self) -> Self { $name(self.0.max(other.0)) }
            #[inline] pub fn min(self, other: Self) -> Self { $name(self.0.min(other.0)) }
            #[inline] pub fn is_finite(self) -> bool { self.0.is_finite() }
            #[inline] pub fn is_nan(self) -> bool { self.0.is_nan() }
            #[inline] pub fn sqrt(self) -> Self { $name(self.0.sqrt()) }
            #[inline] pub fn powi(self, n: i32) -> Self { $name(self.0.powi(n)) }
            #[inline] pub fn ln(self) -> Self { $name(self.0.ln()) }
            #[inline] pub fn exp(self) -> Self { $name(self.0.exp()) }
        }

        impl Eq for $name {}

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        impl Hash for $name {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.0.to_bits().hash(state);
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:.6}", self.0)
            }
        }

        impl ops::Add for $name {
            type Output = Self;
            #[inline] fn add(self, rhs: Self) -> Self { $name(self.0 + rhs.0) }
        }
        impl ops::Sub for $name {
            type Output = Self;
            #[inline] fn sub(self, rhs: Self) -> Self { $name(self.0 - rhs.0) }
        }
        impl ops::Mul<f64> for $name {
            type Output = Self;
            #[inline] fn mul(self, rhs: f64) -> Self { $name(self.0 * rhs) }
        }
        impl ops::Mul<$name> for f64 {
            type Output = $name;
            #[inline] fn mul(self, rhs: $name) -> $name { $name(self * rhs.0) }
        }
        impl ops::Div<f64> for $name {
            type Output = Self;
            #[inline] fn div(self, rhs: f64) -> Self { $name(self.0 / rhs) }
        }
        impl ops::Div<$name> for $name {
            type Output = f64;
            #[inline] fn div(self, rhs: $name) -> f64 { self.0 / rhs.0 }
        }
        impl ops::Neg for $name {
            type Output = Self;
            #[inline] fn neg(self) -> Self { $name(-self.0) }
        }
        impl ops::AddAssign for $name {
            #[inline] fn add_assign(&mut self, rhs: Self) { self.0 += rhs.0; }
        }
        impl ops::SubAssign for $name {
            #[inline] fn sub_assign(&mut self, rhs: Self) { self.0 -= rhs.0; }
        }
        impl ops::MulAssign<f64> for $name {
            #[inline] fn mul_assign(&mut self, rhs: f64) { self.0 *= rhs; }
        }
        impl ops::DivAssign<f64> for $name {
            #[inline] fn div_assign(&mut self, rhs: f64) { self.0 /= rhs; }
        }
        impl From<f64> for $name {
            #[inline] fn from(v: f64) -> Self { $name(v) }
        }
        impl From<$name> for f64 {
            #[inline] fn from(v: $name) -> f64 { v.0 }
        }
        impl std::iter::Sum for $name {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                $name(iter.map(|x| x.0).sum())
            }
        }
    };
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Numeric newtype wrappers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

define_float_newtype!(
    /// A monetary price value in the market.
    Price
);
define_float_newtype!(
    /// A quantity of goods produced or consumed.
    Quantity
);
define_float_newtype!(
    /// A profit value for a player or firm.
    Profit
);
define_float_newtype!(
    /// A cost value for production or operation.
    Cost
);

// ── Integer-based newtypes ──────────────────────────────────────────────────

/// Unique identifier for a player/firm in the market.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PlayerId(pub usize);

impl PlayerId {
    #[inline] pub fn new(id: usize) -> Self { PlayerId(id) }
    #[inline] pub fn value(self) -> usize { self.0 }
}

impl fmt::Display for PlayerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Player({})", self.0)
    }
}

impl From<usize> for PlayerId {
    fn from(v: usize) -> Self { PlayerId(v) }
}

/// A round number in a repeated game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RoundNumber(pub usize);

impl RoundNumber {
    #[inline] pub fn new(n: usize) -> Self { RoundNumber(n) }
    #[inline] pub fn value(self) -> usize { self.0 }

    pub fn next(self) -> Self { RoundNumber(self.0 + 1) }

    pub fn prev(self) -> Option<Self> {
        if self.0 > 0 { Some(RoundNumber(self.0 - 1)) } else { None }
    }
}

impl fmt::Display for RoundNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Round({})", self.0)
    }
}

impl From<usize> for RoundNumber {
    fn from(v: usize) -> Self { RoundNumber(v) }
}

impl ops::Add<usize> for RoundNumber {
    type Output = Self;
    fn add(self, rhs: usize) -> Self { RoundNumber(self.0 + rhs) }
}

impl ops::Sub<RoundNumber> for RoundNumber {
    type Output = usize;
    fn sub(self, rhs: RoundNumber) -> usize { self.0 - rhs.0 }
}

impl RoundNumber {
    pub fn saturating_sub(self, rhs: usize) -> Self {
        RoundNumber(self.0.saturating_sub(rhs))
    }
}

impl ops::Rem<usize> for RoundNumber {
    type Output = usize;
    fn rem(self, rhs: usize) -> usize { self.0 % rhs }
}

impl From<RoundNumber> for usize {
    fn from(v: RoundNumber) -> usize { v.0 }
}

impl<T> ops::Index<RoundNumber> for Vec<T> {
    type Output = T;
    fn index(&self, index: RoundNumber) -> &T { &self[index.0] }
}

impl<T> ops::Index<RoundNumber> for [T] {
    type Output = T;
    fn index(&self, index: RoundNumber) -> &T { &self[index.0] }
}

impl<T> ops::Index<PlayerId> for Vec<T> {
    type Output = T;
    fn index(&self, index: PlayerId) -> &T { &self[index.0] }
}

impl<T> ops::Index<PlayerId> for [T] {
    type Output = T;
    fn index(&self, index: PlayerId) -> &T { &self[index.0] }
}

impl PartialEq<usize> for PlayerId {
    fn eq(&self, other: &usize) -> bool { self.0 == *other }
}

impl PartialEq<PlayerId> for usize {
    fn eq(&self, other: &PlayerId) -> bool { *self == other.0 }
}

impl PartialOrd<usize> for PlayerId {
    fn partial_cmp(&self, other: &usize) -> Option<Ordering> { self.0.partial_cmp(other) }
}

impl PartialOrd<PlayerId> for usize {
    fn partial_cmp(&self, other: &PlayerId) -> Option<Ordering> { self.partial_cmp(&other.0) }
}

impl From<PlayerId> for u64 {
    fn from(v: PlayerId) -> u64 { v.0 as u64 }
}

impl ops::Sub<f64> for Price {
    type Output = Price;
    fn sub(self, rhs: f64) -> Price { Price(self.0 - rhs) }
}

impl ops::Sub<Price> for f64 {
    type Output = f64;
    fn sub(self, rhs: Price) -> f64 { self - rhs.0 }
}

impl ops::Div<Price> for f64 {
    type Output = f64;
    fn div(self, rhs: Price) -> f64 { self / rhs.0 }
}

impl<'a> std::iter::Sum<&'a Price> for f64 {
    fn sum<I: Iterator<Item = &'a Price>>(iter: I) -> f64 {
        iter.map(|p| p.0).sum()
    }
}

impl std::iter::Sum<Price> for f64 {
    fn sum<I: Iterator<Item = Price>>(iter: I) -> f64 {
        iter.map(|p| p.0).sum()
    }
}

impl std::iter::Sum<Profit> for f64 {
    fn sum<I: Iterator<Item = Profit>>(iter: I) -> f64 {
        iter.map(|p| p.0).sum()
    }
}

impl<'a> std::iter::Sum<&'a Profit> for f64 {
    fn sum<I: Iterator<Item = &'a Profit>>(iter: I) -> f64 {
        iter.map(|p| p.0).sum()
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Market types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Type of market competition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketType {
    /// Price competition (Bertrand model).
    Bertrand,
    /// Quantity competition (Cournot model).
    Cournot,
}

impl fmt::Display for MarketType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarketType::Bertrand => write!(f, "Bertrand"),
            MarketType::Cournot => write!(f, "Cournot"),
        }
    }
}

/// Demand system specification with parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DemandSystem {
    /// Linear demand: Q = max_quantity − slope × P, split equally among firms.
    Linear { max_quantity: f64, slope: f64 },
    /// Constant Elasticity of Substitution demand.
    CES {
        elasticity_of_substitution: f64,
        market_size: f64,
        quality_indices: Vec<f64>,
    },
    /// Logit demand model.
    Logit {
        temperature: f64,
        outside_option_value: f64,
        market_size: f64,
    },
}

impl DemandSystem {
    /// Compute demand quantities at given prices for all players.
    pub fn compute_quantities(&self, prices: &[Price], num_players: usize) -> Vec<Quantity> {
        match self {
            DemandSystem::Linear { max_quantity, slope } => {
                prices
                    .iter()
                    .map(|p| {
                        let q = (max_quantity - slope * p.0) / num_players as f64;
                        Quantity::new(q.max(0.0))
                    })
                    .collect()
            }
            DemandSystem::CES {
                elasticity_of_substitution,
                market_size,
                quality_indices,
            } => {
                let sigma = *elasticity_of_substitution;
                let denom: f64 = prices
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let qi = quality_indices.get(i).copied().unwrap_or(1.0);
                        qi * p.0.powf(1.0 - sigma)
                    })
                    .sum();
                if denom.abs() < 1e-30 {
                    return vec![Quantity::ZERO; prices.len()];
                }
                prices
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let qi = quality_indices.get(i).copied().unwrap_or(1.0);
                        let share = qi * p.0.powf(1.0 - sigma) / denom;
                        Quantity::new(market_size * share / p.0)
                    })
                    .collect()
            }
            DemandSystem::Logit {
                temperature,
                outside_option_value,
                market_size,
            } => {
                let exp_outside = (-outside_option_value / temperature).exp();
                let exp_vals: Vec<f64> =
                    prices.iter().map(|p| (-p.0 / temperature).exp()).collect();
                let denom = exp_outside + exp_vals.iter().sum::<f64>();
                if denom.abs() < 1e-30 {
                    return vec![Quantity::ZERO; prices.len()];
                }
                exp_vals
                    .iter()
                    .map(|ev| Quantity::new(market_size * ev / denom))
                    .collect()
            }
        }
    }

    /// Competitive (Nash) price for a symmetric Bertrand game.
    pub fn competitive_price(&self, marginal_cost: Cost) -> Price {
        match self {
            DemandSystem::Linear { .. } => Price::new(marginal_cost.0),
            DemandSystem::CES { elasticity_of_substitution, .. } => {
                let sigma = *elasticity_of_substitution;
                if sigma <= 1.0 { return Price::new(marginal_cost.0); }
                Price::new(marginal_cost.0 * sigma / (sigma - 1.0))
            }
            DemandSystem::Logit { temperature, .. } => {
                Price::new(marginal_cost.0 + *temperature)
            }
        }
    }

    /// Monopoly price for a single firm.
    pub fn monopoly_price(&self, marginal_cost: Cost) -> Price {
        match self {
            DemandSystem::Linear { max_quantity, slope } => {
                Price::new((max_quantity / slope + marginal_cost.0) / 2.0)
            }
            DemandSystem::CES { elasticity_of_substitution, .. } => {
                let sigma = *elasticity_of_substitution;
                if sigma <= 1.0 { return Price::new(marginal_cost.0); }
                Price::new(marginal_cost.0 * sigma / (sigma - 1.0))
            }
            DemandSystem::Logit { temperature, .. } => {
                Price::new(marginal_cost.0 + 2.0 * *temperature)
            }
        }
    }
}

impl fmt::Display for DemandSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DemandSystem::Linear { max_quantity, slope } =>
                write!(f, "Linear(max_q={}, slope={})", max_quantity, slope),
            DemandSystem::CES { elasticity_of_substitution, .. } =>
                write!(f, "CES(σ={})", elasticity_of_substitution),
            DemandSystem::Logit { temperature, .. } =>
                write!(f, "Logit(τ={})", temperature),
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Player actions and market outcomes
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// An action taken by a player in a single round.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlayerAction {
    pub player_id: PlayerId,
    pub price: Option<Price>,
    pub quantity: Option<Quantity>,
}

impl PlayerAction {
    pub fn new(player_id: PlayerId, price: Price) -> Self {
        PlayerAction { player_id, price: Some(price), quantity: None }
    }

    pub fn bertrand(player_id: PlayerId, price: Price) -> Self {
        PlayerAction { player_id, price: Some(price), quantity: None }
    }

    pub fn cournot(player_id: PlayerId, quantity: Quantity) -> Self {
        PlayerAction { player_id, price: None, quantity: Some(quantity) }
    }

    pub fn action_value(&self) -> f64 {
        self.price.map(|p| p.0)
            .or_else(|| self.quantity.map(|q| q.0))
            .unwrap_or(0.0)
    }
}

impl fmt::Display for PlayerAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(p) = self.price {
            write!(f, "{}→p={:.4}", self.player_id, p.0)
        } else if let Some(q) = self.quantity {
            write!(f, "{}→q={:.4}", self.player_id, q.0)
        } else {
            write!(f, "{}→?", self.player_id)
        }
    }
}

/// The outcome of a single round of market interaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarketOutcome {
    pub round: RoundNumber,
    pub actions: Vec<PlayerAction>,
    pub prices: Vec<Price>,
    pub quantities: Vec<Quantity>,
    pub profits: Vec<Profit>,
    pub total_surplus: Profit,
    pub consumer_surplus: Profit,
}

impl MarketOutcome {
    pub fn new(
        round: RoundNumber,
        actions: Vec<PlayerAction>,
        prices: Vec<Price>,
        quantities: Vec<Quantity>,
        profits: Vec<Profit>,
    ) -> Self {
        let total_surplus = Profit::new(profits.iter().map(|p| p.0).sum::<f64>());
        MarketOutcome {
            round, actions, prices, quantities, profits,
            total_surplus,
            consumer_surplus: Profit::ZERO,
        }
    }

    pub fn num_players(&self) -> usize { self.prices.len() }

    pub fn mean_price(&self) -> Price {
        if self.prices.is_empty() { return Price::ZERO; }
        let s: f64 = self.prices.iter().map(|p| p.0).sum();
        Price::new(s / self.prices.len() as f64)
    }

    pub fn max_price(&self) -> Price {
        self.prices.iter().copied().max().unwrap_or(Price::ZERO)
    }

    pub fn min_price(&self) -> Price {
        self.prices.iter().copied().min().unwrap_or(Price::ZERO)
    }

    pub fn total_profit(&self) -> Profit { self.profits.iter().copied().sum() }

    pub fn player_profit(&self, p: PlayerId) -> Profit {
        self.profits.get(p.0).copied().unwrap_or(Profit::ZERO)
    }

    pub fn player_price(&self, p: PlayerId) -> Price {
        self.prices.get(p.0).copied().unwrap_or(Price::ZERO)
    }

    pub fn player_quantity(&self, p: PlayerId) -> Quantity {
        self.quantities.get(p.0).copied().unwrap_or(Quantity::ZERO)
    }

    pub fn price_dispersion(&self) -> f64 {
        if self.prices.len() < 2 { return 0.0; }
        let mean = self.mean_price().0;
        let var: f64 = self.prices.iter()
            .map(|p| (p.0 - mean).powi(2)).sum::<f64>()
            / (self.prices.len() - 1) as f64;
        var.sqrt()
    }
}

impl fmt::Display for MarketOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Round {} | mean_p={:.4} | total_π={:.4}",
            self.round.0, self.mean_price().0, self.total_profit().0)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Price trajectory
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Metadata attached to a trajectory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryMetadata {
    pub algorithm_type: AlgorithmType,
    pub seed: u64,
    pub convergence_round: Option<RoundNumber>,
    pub description: String,
}

/// A sequence of market outcomes over time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriceTrajectory {
    pub outcomes: Vec<MarketOutcome>,
    pub market_type: MarketType,
    pub num_players: usize,
    pub metadata: TrajectoryMetadata,
}

impl PriceTrajectory {
    pub fn new(
        outcomes: Vec<MarketOutcome>,
        market_type: MarketType,
        num_players: usize,
        algorithm_type: AlgorithmType,
        seed: u64,
    ) -> Self {
        PriceTrajectory {
            outcomes, market_type, num_players,
            metadata: TrajectoryMetadata {
                algorithm_type, seed,
                convergence_round: None,
                description: String::new(),
            },
        }
    }

    pub fn len(&self) -> usize { self.outcomes.len() }
    pub fn is_empty(&self) -> bool { self.outcomes.is_empty() }

    pub fn get(&self, round: RoundNumber) -> Option<&MarketOutcome> {
        self.outcomes.get(round.0)
    }

    pub fn slice(&self, start: RoundNumber, end: RoundNumber) -> Vec<&MarketOutcome> {
        self.outcomes.iter()
            .filter(|o| o.round >= start && o.round < end)
            .collect()
    }

    pub fn player_price_series(&self, player: PlayerId) -> Vec<Price> {
        self.outcomes.iter().map(|o| o.player_price(player)).collect()
    }

    pub fn player_profit_series(&self, player: PlayerId) -> Vec<Profit> {
        self.outcomes.iter().map(|o| o.player_profit(player)).collect()
    }

    pub fn overall_mean_price(&self) -> Price {
        if self.outcomes.is_empty() { return Price::ZERO; }
        let s: f64 = self.outcomes.iter().map(|o| o.mean_price().0).sum();
        Price::new(s / self.outcomes.len() as f64)
    }

    pub fn mean_price(&self) -> Price {
        self.overall_mean_price()
    }

    pub fn prices_for_player(&self, player: PlayerId) -> Vec<Price> {
        self.player_price_series(player)
    }

    pub fn mean_price_series(&self) -> Vec<Price> {
        self.outcomes.iter().map(|o| o.mean_price()).collect()
    }

    /// Detect convergence: first round where the rolling standard deviation
    /// stays below `threshold` for `window` consecutive rounds.
    pub fn detect_convergence(&self, threshold: f64, window: usize) -> Option<RoundNumber> {
        if self.outcomes.len() < window { return None; }
        let mp: Vec<f64> = self.outcomes.iter().map(|o| o.mean_price().0).collect();
        for start in 0..mp.len().saturating_sub(window) {
            let seg = &mp[start..start + window];
            let mean = seg.iter().sum::<f64>() / window as f64;
            let var = seg.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (window - 1) as f64;
            if var.sqrt() < threshold {
                return Some(RoundNumber(start));
            }
        }
        None
    }

    pub fn tail(&self, n: usize) -> &[MarketOutcome] {
        let start = self.outcomes.len().saturating_sub(n);
        &self.outcomes[start..]
    }
}

impl fmt::Display for PriceTrajectory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trajectory({}, {} rounds, {} players, {})",
            self.market_type, self.outcomes.len(),
            self.num_players, self.metadata.algorithm_type)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Phantom-tagged trajectory segments
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Marker trait for segment phase tags.
pub trait SegmentPhase: fmt::Debug + Clone + Send + Sync + 'static {
    fn phase_name() -> &'static str;
}

/// Training phase — used for algorithm convergence.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrainingSegment;
impl SegmentPhase for TrainingSegment { fn phase_name() -> &'static str { "training" } }

/// Testing phase — used for collusion detection tests.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TestingSegment;
impl SegmentPhase for TestingSegment { fn phase_name() -> &'static str { "testing" } }

/// Validation phase — cross-validation of results.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ValidationSegment;
impl SegmentPhase for ValidationSegment { fn phase_name() -> &'static str { "validation" } }

/// Holdout phase — never seen during analysis, used for final confirmation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HoldoutSegment;
impl SegmentPhase for HoldoutSegment { fn phase_name() -> &'static str { "holdout" } }

/// A trajectory segment tagged with a phantom type to prevent cross-segment
/// data reuse at compile time.
#[derive(Debug, Clone)]
pub struct TrajectorySegment<S: SegmentPhase> {
    pub outcomes: Vec<MarketOutcome>,
    pub start_round: RoundNumber,
    pub end_round: RoundNumber,
    _phase: PhantomData<S>,
}

impl<S: SegmentPhase> TrajectorySegment<S> {
    pub fn new(outcomes: Vec<MarketOutcome>, start: RoundNumber, end: RoundNumber) -> Self {
        TrajectorySegment { outcomes, start_round: start, end_round: end, _phase: PhantomData }
    }

    pub fn phase_name(&self) -> &'static str { S::phase_name() }
    pub fn len(&self) -> usize { self.outcomes.len() }
    pub fn is_empty(&self) -> bool { self.outcomes.is_empty() }

    pub fn mean_price(&self) -> Price {
        if self.outcomes.is_empty() { return Price::ZERO; }
        let s: f64 = self.outcomes.iter().map(|o| o.mean_price().0).sum();
        Price::new(s / self.outcomes.len() as f64)
    }

    pub fn price_series(&self, player: PlayerId) -> Vec<Price> {
        self.outcomes.iter().map(|o| o.player_price(player)).collect()
    }

    pub fn profit_series(&self, player: PlayerId) -> Vec<Profit> {
        self.outcomes.iter().map(|o| o.player_profit(player)).collect()
    }

    pub fn mean_prices(&self) -> Vec<Price> {
        self.outcomes.iter().map(|o| o.mean_price()).collect()
    }

    pub fn num_rounds(&self) -> usize {
        self.end_round.0.saturating_sub(self.start_round.0)
    }
}

impl<S: SegmentPhase> fmt::Display for TrajectorySegment<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Segment<{}>[{}..{}] ({} outcomes)",
            S::phase_name(), self.start_round.0, self.end_round.0, self.outcomes.len())
    }
}

/// Split a trajectory into four phantom-tagged segments.
pub fn split_trajectory(
    trajectory: &PriceTrajectory,
    training_end: usize,
    testing_end: usize,
    validation_end: usize,
) -> (
    TrajectorySegment<TrainingSegment>,
    TrajectorySegment<TestingSegment>,
    TrajectorySegment<ValidationSegment>,
    TrajectorySegment<HoldoutSegment>,
) {
    let n = trajectory.outcomes.len();
    let te = training_end.min(n);
    let tse = testing_end.min(n);
    let ve = validation_end.min(n);

    let training = TrajectorySegment::new(
        trajectory.outcomes[..te].to_vec(), RoundNumber(0), RoundNumber(te));
    let testing = TrajectorySegment::new(
        trajectory.outcomes[te..tse].to_vec(), RoundNumber(te), RoundNumber(tse));
    let validation = TrajectorySegment::new(
        trajectory.outcomes[tse..ve].to_vec(), RoundNumber(tse), RoundNumber(ve));
    let holdout = TrajectorySegment::new(
        trajectory.outcomes[ve..].to_vec(), RoundNumber(ve), RoundNumber(n));

    (training, testing, validation, holdout)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Game configuration
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Discretised price grid for tabular RL algorithms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriceGrid {
    pub min_price: Price,
    pub max_price: Price,
    pub num_points: usize,
}

impl PriceGrid {
    pub fn new(min_price: Price, max_price: Price, num_points: usize) -> Self {
        PriceGrid { min_price, max_price, num_points }
    }

    pub fn step_size(&self) -> f64 {
        if self.num_points <= 1 { return 0.0; }
        (self.max_price.0 - self.min_price.0) / (self.num_points - 1) as f64
    }

    pub fn price_at(&self, index: usize) -> Price {
        Price::new(self.min_price.0 + self.step_size() * index as f64)
    }

    pub fn nearest_index(&self, price: Price) -> usize {
        if self.num_points == 0 { return 0; }
        let step = self.step_size();
        if step <= 0.0 { return 0; }
        let raw = (price.0 - self.min_price.0) / step;
        raw.round().max(0.0).min((self.num_points - 1) as f64) as usize
    }

    pub fn prices(&self) -> Vec<Price> {
        (0..self.num_points).map(|i| self.price_at(i)).collect()
    }
}

/// Complete configuration for a market game.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GameConfig {
    pub market_type: MarketType,
    pub demand_system: DemandSystem,
    pub num_players: usize,
    pub discount_factor: f64,
    pub marginal_costs: Vec<Cost>,
    pub price_grid: Option<PriceGrid>,
    pub max_rounds: usize,
    pub description: String,
}

impl GameConfig {
    /// Create a symmetric game where all players share the same marginal cost.
    pub fn symmetric(
        market_type: MarketType,
        demand_system: DemandSystem,
        num_players: usize,
        discount_factor: f64,
        marginal_cost: Cost,
        max_rounds: usize,
    ) -> Self {
        GameConfig {
            market_type, demand_system, num_players, discount_factor,
            marginal_costs: vec![marginal_cost; num_players],
            price_grid: None, max_rounds, description: String::new(),
        }
    }

    pub fn with_price_grid(mut self, grid: PriceGrid) -> Self {
        self.price_grid = Some(grid); self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into(); self
    }

    pub fn is_symmetric(&self) -> bool {
        if self.marginal_costs.is_empty() { return true; }
        let first = self.marginal_costs[0];
        self.marginal_costs.iter().all(|c| *c == first)
    }

    pub fn competitive_price(&self) -> Price {
        let mc = self.marginal_costs.first().copied().unwrap_or(Cost::ZERO);
        self.demand_system.competitive_price(mc)
    }

    pub fn monopoly_price(&self) -> Price {
        let mc = self.marginal_costs.first().copied().unwrap_or(Cost::ZERO);
        self.demand_system.monopoly_price(mc)
    }

    pub fn player_ids(&self) -> Vec<PlayerId> {
        (0..self.num_players).map(PlayerId).collect()
    }
}

impl fmt::Display for GameConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Game({}, {}, n={}, δ={:.3})",
            self.market_type, self.demand_system, self.num_players, self.discount_factor)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Algorithm types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Types of pricing/quantity-setting algorithms.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlgorithmType {
    QLearning, GrimTrigger, TitForTat, DQN, PPO, SARSA,
    EpsilonGreedy, MyopicBestResponse, Edgeworth,
    Bandit, NashEquilibrium, Custom(String),
}

impl AlgorithmType {
    pub fn is_reinforcement_learning(&self) -> bool {
        matches!(self,
            AlgorithmType::QLearning | AlgorithmType::DQN | AlgorithmType::PPO
            | AlgorithmType::SARSA | AlgorithmType::EpsilonGreedy | AlgorithmType::Bandit)
    }

    pub fn is_trigger_strategy(&self) -> bool {
        matches!(self, AlgorithmType::GrimTrigger | AlgorithmType::TitForTat)
    }

    pub fn requires_price_grid(&self) -> bool {
        matches!(self,
            AlgorithmType::QLearning | AlgorithmType::SARSA | AlgorithmType::EpsilonGreedy
            | AlgorithmType::Bandit)
    }

    pub fn name(&self) -> &str {
        match self {
            AlgorithmType::QLearning => "Q-Learning",
            AlgorithmType::GrimTrigger => "Grim Trigger",
            AlgorithmType::TitForTat => "Tit-for-Tat",
            AlgorithmType::DQN => "DQN",
            AlgorithmType::PPO => "PPO",
            AlgorithmType::SARSA => "SARSA",
            AlgorithmType::EpsilonGreedy => "ε-Greedy",
            AlgorithmType::MyopicBestResponse => "Myopic Best Response",
            AlgorithmType::Edgeworth => "Edgeworth",
            AlgorithmType::Bandit => "Bandit",
            AlgorithmType::NashEquilibrium => "Nash Equilibrium",
            AlgorithmType::Custom(name) => name.as_str(),
        }
    }
}

impl fmt::Display for AlgorithmType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Hyperparameters for an algorithm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    pub algorithm_type: AlgorithmType,
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub epsilon: f64,
    pub epsilon_decay: f64,
    pub epsilon_min: f64,
    pub batch_size: usize,
    pub memory_size: usize,
    pub hidden_layers: Vec<usize>,
    pub exploration_rounds: usize,
    pub extra_params: std::collections::HashMap<String, f64>,
}

impl AlgorithmConfig {
    pub fn new(algorithm_type: AlgorithmType) -> Self {
        AlgorithmConfig {
            algorithm_type,
            learning_rate: 0.1, discount_factor: 0.95, epsilon: 1.0,
            epsilon_decay: 0.99995, epsilon_min: 0.01,
            batch_size: 1, memory_size: 0,
            hidden_layers: vec![], exploration_rounds: 100_000,
            extra_params: std::collections::HashMap::new(),
        }
    }

    pub fn get_param(&self, key: &str) -> Option<f64> {
        self.extra_params.get(key).copied()
    }

    pub fn q_learning(lr: f64, gamma: f64, epsilon: f64) -> Self {
        AlgorithmConfig {
            algorithm_type: AlgorithmType::QLearning,
            learning_rate: lr, discount_factor: gamma, epsilon,
            epsilon_decay: 0.99995, epsilon_min: 0.01,
            batch_size: 1, memory_size: 0,
            hidden_layers: vec![], exploration_rounds: 100_000,
            extra_params: std::collections::HashMap::new(),
        }
    }

    pub fn dqn(lr: f64, gamma: f64, hidden: Vec<usize>) -> Self {
        AlgorithmConfig {
            algorithm_type: AlgorithmType::DQN,
            learning_rate: lr, discount_factor: gamma,
            epsilon: 1.0, epsilon_decay: 0.9999, epsilon_min: 0.01,
            batch_size: 32, memory_size: 10_000,
            hidden_layers: hidden, exploration_rounds: 50_000,
            extra_params: std::collections::HashMap::new(),
        }
    }

    pub fn grim_trigger() -> Self {
        AlgorithmConfig {
            algorithm_type: AlgorithmType::GrimTrigger,
            learning_rate: 0.0, discount_factor: 0.0, epsilon: 0.0,
            epsilon_decay: 0.0, epsilon_min: 0.0,
            batch_size: 0, memory_size: 0,
            hidden_layers: vec![], exploration_rounds: 0,
            extra_params: std::collections::HashMap::new(),
        }
    }

    pub fn tit_for_tat() -> Self {
        AlgorithmConfig {
            algorithm_type: AlgorithmType::TitForTat,
            learning_rate: 0.0, discount_factor: 0.0, epsilon: 0.0,
            epsilon_decay: 0.0, epsilon_min: 0.0,
            batch_size: 0, memory_size: 0,
            hidden_layers: vec![], exploration_rounds: 0,
            extra_params: std::collections::HashMap::new(),
        }
    }

    pub fn with_param(mut self, key: impl Into<String>, value: f64) -> Self {
        self.extra_params.insert(key.into(), value); self
    }
}

impl fmt::Display for AlgorithmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(α={}, γ={}, ε={})",
            self.algorithm_type, self.learning_rate, self.discount_factor, self.epsilon)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Oracle access levels
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Level of access to the algorithm's internal state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OracleAccessLevel {
    /// Passive observation: only market outcomes visible.
    Layer0Passive,
    /// Checkpoint inspection: internal state snapshots accessible.
    Layer1Checkpoint,
    /// Full rewind: re-run algorithm from arbitrary states.
    Layer2FullRewind,
}

impl OracleAccessLevel {
    pub const Layer0: OracleAccessLevel = OracleAccessLevel::Layer0Passive;
    pub const Layer1: OracleAccessLevel = OracleAccessLevel::Layer1Checkpoint;
    pub const Layer2: OracleAccessLevel = OracleAccessLevel::Layer2FullRewind;

    pub fn level(&self) -> u8 {
        match self {
            OracleAccessLevel::Layer0Passive => 0,
            OracleAccessLevel::Layer1Checkpoint => 1,
            OracleAccessLevel::Layer2FullRewind => 2,
        }
    }

    pub fn can_inspect_state(&self) -> bool {
        matches!(self, OracleAccessLevel::Layer1Checkpoint | OracleAccessLevel::Layer2FullRewind)
    }

    pub fn can_rewind(&self) -> bool {
        matches!(self, OracleAccessLevel::Layer2FullRewind)
    }
}

impl fmt::Display for OracleAccessLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OracleAccessLevel::Layer0Passive => write!(f, "Layer0(Passive)"),
            OracleAccessLevel::Layer1Checkpoint => write!(f, "Layer1(Checkpoint)"),
            OracleAccessLevel::Layer2FullRewind => write!(f, "Layer2(FullRewind)"),
        }
    }
}

impl PartialOrd for OracleAccessLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl Ord for OracleAccessLevel {
    fn cmp(&self, other: &Self) -> Ordering { self.level().cmp(&other.level()) }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Evaluation mode & simulation config
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Evaluation mode controlling thoroughness vs speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvaluationMode {
    Smoke,
    Standard,
    Full,
}

impl EvaluationMode {
    pub fn simulation_rounds(&self) -> usize {
        match self { EvaluationMode::Smoke => 10_000, EvaluationMode::Standard => 100_000, EvaluationMode::Full => 1_000_000 }
    }

    pub fn bootstrap_iterations(&self) -> usize {
        match self { EvaluationMode::Smoke => 100, EvaluationMode::Standard => 1_000, EvaluationMode::Full => 10_000 }
    }

    pub fn monte_carlo_iterations(&self) -> usize {
        match self { EvaluationMode::Smoke => 50, EvaluationMode::Standard => 500, EvaluationMode::Full => 5_000 }
    }

    pub fn is_full(&self) -> bool { matches!(self, EvaluationMode::Full) }
}

impl fmt::Display for EvaluationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { EvaluationMode::Smoke => write!(f, "Smoke"), EvaluationMode::Standard => write!(f, "Standard"), EvaluationMode::Full => write!(f, "Full") }
    }
}

/// Complete simulation configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub game: GameConfig,
    pub algorithm: AlgorithmConfig,
    pub oracle_access: OracleAccessLevel,
    pub evaluation_mode: EvaluationMode,
    pub num_episodes: usize,
    pub training_fraction: f64,
    pub testing_fraction: f64,
    pub validation_fraction: f64,
    pub holdout_fraction: f64,
    pub random_seed: u64,
}

impl SimulationConfig {
    pub fn new(game: GameConfig, algorithm: AlgorithmConfig, mode: EvaluationMode) -> Self {
        SimulationConfig {
            game, algorithm,
            oracle_access: OracleAccessLevel::Layer0Passive,
            evaluation_mode: mode,
            num_episodes: mode.simulation_rounds(),
            training_fraction: 0.40,
            testing_fraction: 0.20,
            validation_fraction: 0.20,
            holdout_fraction: 0.20,
            random_seed: 42,
        }
    }

    pub fn with_oracle(mut self, level: OracleAccessLevel) -> Self {
        self.oracle_access = level; self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed; self
    }

    pub fn validate_fractions(&self) -> bool {
        let sum = self.training_fraction + self.testing_fraction
            + self.validation_fraction + self.holdout_fraction;
        (sum - 1.0).abs() < 1e-9
    }

    pub fn segment_boundaries(&self) -> (usize, usize, usize) {
        let n = self.num_episodes;
        let te = (n as f64 * self.training_fraction).round() as usize;
        let tse = te + (n as f64 * self.testing_fraction).round() as usize;
        let ve = tse + (n as f64 * self.validation_fraction).round() as usize;
        (te, tse, ve)
    }

    // Convenience accessors that delegate to nested fields
    pub fn num_players(&self) -> usize { self.game.num_players }
    pub fn num_rounds(&self) -> usize { self.game.max_rounds }
    pub fn demand_system(&self) -> &DemandSystem { &self.game.demand_system }
    pub fn marginal_costs(&self) -> &[Cost] { &self.game.marginal_costs }
    pub fn marginal_cost(&self) -> Cost {
        self.game.marginal_costs.first().copied().unwrap_or(Cost::ZERO)
    }
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig::new(
            GameConfig::symmetric(
                MarketType::Bertrand,
                DemandSystem::Linear { max_quantity: 10.0, slope: 1.0 },
                2,
                0.95,
                Cost::new(1.0),
                1000,
            ),
            AlgorithmConfig::new(AlgorithmType::QLearning),
            EvaluationMode::Standard,
        )
    }
}

impl fmt::Display for SimulationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sim({}, {}, {}, {})",
            self.game, self.algorithm, self.oracle_access, self.evaluation_mode)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Collusion index
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Normalized collusion index in [0, 1]: 0 = competitive, 1 = monopoly.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize, Default)]
pub struct CollusionIndex(pub f64);

impl CollusionIndex {
    pub fn new(value: f64) -> Self { CollusionIndex(value.clamp(0.0, 1.0)) }
    pub fn value(self) -> f64 { self.0 }

    pub fn from_profits(observed: Profit, competitive: Profit, monopoly: Profit) -> Self {
        let d = monopoly.0 - competitive.0;
        if d.abs() < 1e-12 { return CollusionIndex(0.0); }
        CollusionIndex::new((observed.0 - competitive.0) / d)
    }

    pub fn from_prices(observed: Price, competitive: Price, monopoly: Price) -> Self {
        let d = monopoly.0 - competitive.0;
        if d.abs() < 1e-12 { return CollusionIndex(0.0); }
        CollusionIndex::new((observed.0 - competitive.0) / d)
    }

    pub fn is_collusive(self, threshold: f64) -> bool { self.0 > threshold }
}

impl Eq for CollusionIndex {}
impl Ord for CollusionIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}
impl Hash for CollusionIndex {
    fn hash<H: Hasher>(&self, state: &mut H) { self.0.to_bits().hash(state); }
}
impl fmt::Display for CollusionIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "CI={:.4}", self.0) }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_arithmetic() {
        let a = Price::new(1.5);
        let b = Price::new(2.5);
        assert_eq!((a + b).0, 4.0);
        assert_eq!((b - a).0, 1.0);
        assert_eq!((a * 3.0).0, 4.5);
        assert_eq!((a / 2.0).0, 0.75);
    }

    #[test]
    fn test_price_ordering() {
        let mut v = vec![Price::new(3.0), Price::new(1.0), Price::new(2.0)];
        v.sort();
        assert_eq!(v[0], Price::new(1.0));
        assert_eq!(v[2], Price::new(3.0));
    }

    #[test]
    fn test_price_hash_set() {
        use std::collections::HashSet;
        let mut s = HashSet::new();
        s.insert(Price::new(1.0));
        s.insert(Price::new(1.0));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn test_player_id_display() {
        assert_eq!(format!("{}", PlayerId::new(5)), "Player(5)");
    }

    #[test]
    fn test_round_number_navigation() {
        let r = RoundNumber::new(10);
        assert_eq!(r.next().value(), 11);
        assert_eq!(r.prev().unwrap().value(), 9);
        assert_eq!(RoundNumber::new(0).prev(), None);
    }

    #[test]
    fn test_market_outcome_aggregates() {
        let o = MarketOutcome::new(
            RoundNumber(0), vec![],
            vec![Price::new(2.0), Price::new(4.0)],
            vec![Quantity::new(10.0), Quantity::new(8.0)],
            vec![Profit::new(5.0), Profit::new(3.0)],
        );
        assert!((o.mean_price().0 - 3.0).abs() < 1e-10);
        assert!((o.total_profit().0 - 8.0).abs() < 1e-10);
        assert!((o.price_dispersion() - (2.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_price_grid() {
        let g = PriceGrid::new(Price::new(1.0), Price::new(5.0), 5);
        assert_eq!(g.step_size(), 1.0);
        assert_eq!(g.price_at(0), Price::new(1.0));
        assert_eq!(g.price_at(4), Price::new(5.0));
        assert_eq!(g.nearest_index(Price::new(3.5)), 3);
        assert_eq!(g.prices().len(), 5);
    }

    #[test]
    fn test_game_config_symmetric() {
        let g = GameConfig::symmetric(
            MarketType::Bertrand,
            DemandSystem::Linear { max_quantity: 100.0, slope: 1.0 },
            2, 0.95, Cost::new(1.0), 1000,
        );
        assert!(g.is_symmetric());
        assert_eq!(g.player_ids().len(), 2);
    }

    #[test]
    fn test_algorithm_classification() {
        assert!(AlgorithmType::QLearning.is_reinforcement_learning());
        assert!(!AlgorithmType::GrimTrigger.is_reinforcement_learning());
        assert!(AlgorithmType::GrimTrigger.is_trigger_strategy());
        assert!(AlgorithmType::QLearning.requires_price_grid());
    }

    #[test]
    fn test_oracle_ordering() {
        assert!(OracleAccessLevel::Layer0Passive < OracleAccessLevel::Layer1Checkpoint);
        assert!(OracleAccessLevel::Layer1Checkpoint < OracleAccessLevel::Layer2FullRewind);
        assert!(OracleAccessLevel::Layer2FullRewind.can_rewind());
        assert!(!OracleAccessLevel::Layer0Passive.can_inspect_state());
    }

    #[test]
    fn test_evaluation_mode_scaling() {
        assert!(EvaluationMode::Smoke.simulation_rounds() < EvaluationMode::Standard.simulation_rounds());
        assert!(EvaluationMode::Standard.simulation_rounds() < EvaluationMode::Full.simulation_rounds());
    }

    #[test]
    fn test_simulation_config_fractions() {
        let g = GameConfig::symmetric(
            MarketType::Bertrand,
            DemandSystem::Linear { max_quantity: 100.0, slope: 1.0 },
            2, 0.95, Cost::new(1.0), 1000,
        );
        let cfg = SimulationConfig::new(g, AlgorithmConfig::q_learning(0.1, 0.95, 1.0), EvaluationMode::Standard);
        assert!(cfg.validate_fractions());
    }

    #[test]
    fn test_phantom_segment_tagging() {
        let outcomes = vec![MarketOutcome::new(
            RoundNumber(0), vec![],
            vec![Price::new(2.0)], vec![Quantity::new(5.0)], vec![Profit::new(3.0)],
        )];
        let train: TrajectorySegment<TrainingSegment> =
            TrajectorySegment::new(outcomes.clone(), RoundNumber(0), RoundNumber(1));
        let test: TrajectorySegment<TestingSegment> =
            TrajectorySegment::new(outcomes, RoundNumber(1), RoundNumber(2));
        assert_eq!(train.phase_name(), "training");
        assert_eq!(test.phase_name(), "testing");
    }

    #[test]
    fn test_collusion_index_from_prices() {
        let ci = CollusionIndex::from_prices(Price::new(3.0), Price::new(1.0), Price::new(5.0));
        assert!((ci.value() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_collusion_index_clamped() {
        assert_eq!(CollusionIndex::new(2.0).value(), 1.0);
        assert_eq!(CollusionIndex::new(-1.0).value(), 0.0);
    }

    #[test]
    fn test_linear_demand() {
        let ds = DemandSystem::Linear { max_quantity: 100.0, slope: 2.0 };
        let qs = ds.compute_quantities(&[Price::new(10.0), Price::new(10.0)], 2);
        assert!((qs[0].0 - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_split_trajectory() {
        let outcomes: Vec<MarketOutcome> = (0..100).map(|i| MarketOutcome::new(
            RoundNumber(i), vec![], vec![Price::new(i as f64)],
            vec![Quantity::new(1.0)], vec![Profit::new(1.0)],
        )).collect();
        let traj = PriceTrajectory::new(outcomes, MarketType::Bertrand, 1, AlgorithmType::QLearning, 42);
        let (tr, te, va, ho) = split_trajectory(&traj, 40, 60, 80);
        assert_eq!(tr.len(), 40);
        assert_eq!(te.len(), 20);
        assert_eq!(va.len(), 20);
        assert_eq!(ho.len(), 20);
    }

    #[test]
    fn test_price_sum_iterator() {
        let total: Price = vec![Price::new(1.0), Price::new(2.0), Price::new(3.0)].into_iter().sum();
        assert!((total.0 - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_detect_convergence() {
        let outcomes: Vec<MarketOutcome> = (0..100).map(|i| {
            let p = if i < 50 { i as f64 } else { 50.0 };
            MarketOutcome::new(RoundNumber(i), vec![], vec![Price::new(p)],
                vec![Quantity::new(1.0)], vec![Profit::new(1.0)])
        }).collect();
        let traj = PriceTrajectory::new(outcomes, MarketType::Bertrand, 1, AlgorithmType::QLearning, 0);
        assert!(traj.detect_convergence(0.1, 10).is_some());
    }
}
