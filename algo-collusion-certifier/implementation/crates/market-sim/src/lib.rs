//! Market simulation engine for the CollusionProof algorithmic collusion certification system.
//!
//! This crate provides Bertrand and Cournot competition models, configurable demand systems,
//! a high-performance simulation engine, trajectory analysis, and game orchestration.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
//! │  Demand      │────▶│  Bertrand /   │────▶│  Simulation  │
//! │  Systems     │     │  Cournot      │     │  Engine      │
//! └─────────────┘     └──────────────┘     └──────────────┘
//!       ▲                    │                     │
//!       │                    ▼                     ▼
//! ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
//! │  Noise       │     │  Market       │     │  Trajectory  │
//! │  Models      │     │  Interface    │     │  Analysis    │
//! └─────────────┘     └──────────────┘     └──────────────┘
//!                            │
//!                            ▼
//!                     ┌──────────────┐
//!                     │ Orchestrator  │
//!                     └──────────────┘
//! ```

pub mod demand;
pub mod bertrand;
pub mod cournot;
pub mod market;
pub mod simulation;
pub mod trajectory;
pub mod noise;
pub mod orchestrator;

// ── Local type definitions ──────────────────────────────────────────────────
// These mirror the types in `shared-types`. Once that crate is fully implemented,
// replace this module body with re-exports:
//   pub use shared_types::{Price, Quantity, Profit, Cost, …};

/// Foundational types used throughout the market-sim crate.
pub mod types {
    use serde::{Deserialize, Serialize};
    use std::fmt;

    // ── Newtypes ────────────────────────────────────────────────────────

    /// Price in currency units.
    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize, Default)]
    pub struct Price(pub f64);

    /// Quantity of goods.
    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize, Default)]
    pub struct Quantity(pub f64);

    /// Profit in currency units.
    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize, Default)]
    pub struct Profit(pub f64);

    /// Cost in currency units.
    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize, Default)]
    pub struct Cost(pub f64);

    macro_rules! impl_newtype {
        ($T:ident) => {
            impl fmt::Display for $T {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{:.6}", self.0)
                }
            }
            impl std::ops::Add for $T {
                type Output = Self;
                fn add(self, rhs: Self) -> Self { $T(self.0 + rhs.0) }
            }
            impl std::ops::Sub for $T {
                type Output = Self;
                fn sub(self, rhs: Self) -> Self { $T(self.0 - rhs.0) }
            }
            impl std::ops::Mul<f64> for $T {
                type Output = Self;
                fn mul(self, rhs: f64) -> Self { $T(self.0 * rhs) }
            }
            impl std::ops::Div<f64> for $T {
                type Output = Self;
                fn div(self, rhs: f64) -> Self { $T(self.0 / rhs) }
            }
            impl std::ops::Neg for $T {
                type Output = Self;
                fn neg(self) -> Self { $T(-self.0) }
            }
            impl From<f64> for $T {
                fn from(v: f64) -> Self { $T(v) }
            }
            impl From<$T> for f64 {
                fn from(v: $T) -> f64 { v.0 }
            }
            impl std::ops::AddAssign for $T {
                fn add_assign(&mut self, rhs: Self) { self.0 += rhs.0; }
            }
            impl std::ops::SubAssign for $T {
                fn sub_assign(&mut self, rhs: Self) { self.0 -= rhs.0; }
            }
        };
    }

    impl_newtype!(Price);
    impl_newtype!(Quantity);
    impl_newtype!(Profit);
    impl_newtype!(Cost);

    // ── Identifiers ─────────────────────────────────────────────────────

    /// Zero-based player index.
    pub type PlayerId = usize;
    /// Monotonically increasing round counter.
    pub type RoundNumber = u64;

    // ── Enums ───────────────────────────────────────────────────────────

    /// Type of market competition.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum MarketType {
        Bertrand,
        Cournot,
    }

    /// Demand system functional form.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum DemandSystemType {
        Linear,
        CES,
        Logit,
    }

    // ── Shared data structures ──────────────────────────────────────────

    /// A single action taken by a player in a round.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PlayerAction {
        pub player_id: PlayerId,
        /// Price (Bertrand) or quantity (Cournot).
        pub value: f64,
    }

    impl PlayerAction {
        pub fn new(player_id: PlayerId, value: f64) -> Self {
            Self { player_id, value }
        }
    }

    /// Outcome of a single market round.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MarketOutcome {
        pub round: RoundNumber,
        pub prices: Vec<f64>,
        pub quantities: Vec<f64>,
        pub profits: Vec<f64>,
    }

    impl MarketOutcome {
        pub fn new(
            round: RoundNumber,
            prices: Vec<f64>,
            quantities: Vec<f64>,
            profits: Vec<f64>,
        ) -> Self {
            Self { round, prices, quantities, profits }
        }
    }

    /// Full trajectory of market outcomes over many rounds.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PriceTrajectory {
        pub num_players: usize,
        pub outcomes: Vec<MarketOutcome>,
    }

    impl PriceTrajectory {
        pub fn new(num_players: usize) -> Self {
            Self { num_players, outcomes: Vec::new() }
        }

        pub fn with_outcomes(num_players: usize, outcomes: Vec<MarketOutcome>) -> Self {
            Self { num_players, outcomes }
        }

        pub fn push(&mut self, outcome: MarketOutcome) {
            self.outcomes.push(outcome);
        }

        pub fn len(&self) -> usize {
            self.outcomes.len()
        }

        pub fn is_empty(&self) -> bool {
            self.outcomes.is_empty()
        }

        pub fn prices_for_player(&self, player: PlayerId) -> Vec<f64> {
            self.outcomes.iter().map(|o| o.prices[player]).collect()
        }

        pub fn profits_for_player(&self, player: PlayerId) -> Vec<f64> {
            self.outcomes.iter().map(|o| o.profits[player]).collect()
        }

        pub fn quantities_for_player(&self, player: PlayerId) -> Vec<f64> {
            self.outcomes.iter().map(|o| o.quantities[player]).collect()
        }

        pub fn last_outcome(&self) -> Option<&MarketOutcome> {
            self.outcomes.last()
        }

        pub fn mean_prices(&self) -> Vec<f64> {
            if self.outcomes.is_empty() {
                return vec![];
            }
            let n = self.num_players;
            let t = self.outcomes.len() as f64;
            let mut sums = vec![0.0; n];
            for o in &self.outcomes {
                for i in 0..n {
                    sums[i] += o.prices[i];
                }
            }
            sums.iter().map(|s| s / t).collect()
        }

        pub fn mean_profits(&self) -> Vec<f64> {
            if self.outcomes.is_empty() {
                return vec![];
            }
            let n = self.num_players;
            let t = self.outcomes.len() as f64;
            let mut sums = vec![0.0; n];
            for o in &self.outcomes {
                for i in 0..n {
                    sums[i] += o.profits[i];
                }
            }
            sums.iter().map(|s| s / t).collect()
        }
    }

    /// Full game configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GameConfig {
        pub market_type: MarketType,
        pub demand_system: DemandSystemType,
        pub num_players: usize,
        pub num_rounds: u64,
        pub demand_intercept: f64,
        pub demand_slope: f64,
        pub demand_cross_slope: f64,
        pub substitution_elasticity: f64,
        pub price_sensitivity: f64,
        pub outside_option_value: f64,
        pub market_size: f64,
        pub marginal_costs: Vec<f64>,
        pub price_min: f64,
        pub price_max: f64,
        pub price_grid_size: usize,
        pub quantity_min: f64,
        pub quantity_max: f64,
        pub quantity_grid_size: usize,
    }

    impl Default for GameConfig {
        fn default() -> Self {
            Self {
                market_type: MarketType::Bertrand,
                demand_system: DemandSystemType::Linear,
                num_players: 2,
                num_rounds: 1000,
                demand_intercept: 10.0,
                demand_slope: 1.0,
                demand_cross_slope: 0.5,
                substitution_elasticity: 2.0,
                price_sensitivity: 1.0,
                outside_option_value: 1.0,
                market_size: 100.0,
                marginal_costs: vec![1.0, 1.0],
                price_min: 0.0,
                price_max: 20.0,
                price_grid_size: 100,
                quantity_min: 0.0,
                quantity_max: 20.0,
                quantity_grid_size: 100,
            }
        }
    }

    // ── Interval arithmetic ─────────────────────────────────────────────

    /// Closed interval [lo, hi] for rigorous bound propagation.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct Interval {
        pub lo: f64,
        pub hi: f64,
    }

    impl Interval {
        pub fn new(lo: f64, hi: f64) -> Self {
            debug_assert!(lo <= hi, "Interval lo ({lo}) must be <= hi ({hi})");
            Self { lo, hi }
        }

        pub fn point(v: f64) -> Self {
            Self { lo: v, hi: v }
        }

        pub fn midpoint(&self) -> f64 {
            (self.lo + self.hi) * 0.5
        }

        pub fn width(&self) -> f64 {
            self.hi - self.lo
        }

        pub fn contains(&self, v: f64) -> bool {
            self.lo <= v && v <= self.hi
        }

        pub fn overlaps(&self, other: &Interval) -> bool {
            self.lo <= other.hi && other.lo <= self.hi
        }

        pub fn hull(a: &Interval, b: &Interval) -> Interval {
            Interval::new(a.lo.min(b.lo), a.hi.max(b.hi))
        }

        pub fn intersection(&self, other: &Interval) -> Option<Interval> {
            let lo = self.lo.max(other.lo);
            let hi = self.hi.min(other.hi);
            if lo <= hi { Some(Interval::new(lo, hi)) } else { None }
        }

        pub fn abs(&self) -> Interval {
            if self.lo >= 0.0 {
                *self
            } else if self.hi <= 0.0 {
                Interval::new(-self.hi, -self.lo)
            } else {
                Interval::new(0.0, self.lo.abs().max(self.hi.abs()))
            }
        }
    }

    impl std::ops::Add for Interval {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Interval::new(self.lo + rhs.lo, self.hi + rhs.hi)
        }
    }

    impl std::ops::Sub for Interval {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Interval::new(self.lo - rhs.hi, self.hi - rhs.lo)
        }
    }

    impl std::ops::Mul for Interval {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            let prods = [
                self.lo * rhs.lo,
                self.lo * rhs.hi,
                self.hi * rhs.lo,
                self.hi * rhs.hi,
            ];
            let lo = prods.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = prods.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            Interval::new(lo, hi)
        }
    }

    impl std::ops::Mul<f64> for Interval {
        type Output = Self;
        fn mul(self, rhs: f64) -> Self {
            if rhs >= 0.0 {
                Interval::new(self.lo * rhs, self.hi * rhs)
            } else {
                Interval::new(self.hi * rhs, self.lo * rhs)
            }
        }
    }

    impl std::ops::Div for Interval {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            assert!(
                rhs.lo > 0.0 || rhs.hi < 0.0,
                "Division by interval containing zero"
            );
            let inv = Interval::new(1.0 / rhs.hi, 1.0 / rhs.lo);
            self * inv
        }
    }
}

// ── Error types ─────────────────────────────────────────────────────────────

use thiserror::Error;

/// Errors specific to the market-sim crate.
#[derive(Error, Debug)]
pub enum MarketSimError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Market configuration error: {0}")]
    ConfigError(String),

    #[error("Simulation error: {0}")]
    SimulationError(String),

    #[error("Demand computation error: {0}")]
    DemandError(String),

    #[error("Equilibrium not found: {0}")]
    EquilibriumNotFound(String),

    #[error("Player error: player {player_id} – {message}")]
    PlayerError { player_id: usize, message: String },

    #[error("Timeout after {0}")]
    Timeout(String),

    #[error("Trajectory error: {0}")]
    TrajectoryError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),
}

pub type MarketSimResult<T> = Result<T, MarketSimError>;

// ── Re-exports ──────────────────────────────────────────────────────────────

pub use types::*;

pub use demand::{CESDemand, DemandFunction, LinearDemand, LipschitzBound, LogitDemand};
pub use bertrand::BertrandMarket;
pub use cournot::CournotMarket;
pub use market::{Market, MarketFactory, MarketState};
pub use simulation::{SimulationEngine, SimulationResult};
pub use trajectory::{TrajectoryAnalyzer, TrajectoryBuilder};
pub use noise::{CostShock, DemandShock, NoiseCalibration};
pub use orchestrator::GameOrchestrator;
