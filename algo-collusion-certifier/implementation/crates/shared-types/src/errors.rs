//! Comprehensive error types for the CollusionProof system.
//!
//! Uses `thiserror` for ergonomic error derivation. Each subsystem has its own
//! error type, all unified under [`CollusionError`].

use std::fmt;
use thiserror::Error;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Top-level error
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// The unified error type for the entire CollusionProof system.
#[derive(Debug, Error)]
pub enum CollusionError {
    #[error("market error: {0}")]
    Market(#[from] MarketError),

    #[error("simulation error: {0}")]
    Simulation(#[from] SimulationError),

    #[error("statistical error: {0}")]
    Statistical(#[from] StatisticalError),

    #[error("certificate error: {0}")]
    Certificate(#[from] CertificateError),

    #[error("game theory error: {0}")]
    GameTheory(#[from] GameTheoryError),

    #[error("counterfactual error: {0}")]
    Counterfactual(#[from] CounterfactualError),

    #[error("oracle error: {0}")]
    Oracle(#[from] OracleError),

    #[error("statistical test error: {0}")]
    StatisticalTest(String),

    #[error("resource limit exceeded: {0}")]
    ResourceLimit(String),

    #[error("timeout: {0}")]
    Timeout(String),

    #[error("invalid state: {0}")]
    InvalidState(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("interval arithmetic error: {0}")]
    IntervalArithmetic(String),

    #[error("rational arithmetic error: {0}")]
    RationalArithmetic(String),

    #[error("validation error: {0}")]
    Validation(String),

    #[error("internal error: {0}")]
    Internal(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Convenience result type alias.
pub type CollusionResult<T> = Result<T, CollusionError>;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Subsystem error types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Error)]
pub enum MarketError {
    #[error("invalid price {price}: {reason}")]
    InvalidPrice { price: f64, reason: String },

    #[error("invalid quantity {quantity}: {reason}")]
    InvalidQuantity { quantity: f64, reason: String },

    #[error("player {player_id} not found in market with {num_players} players")]
    PlayerNotFound { player_id: usize, num_players: usize },

    #[error("demand system error: {0}")]
    DemandSystemError(String),

    #[error("price grid error: {0}")]
    PriceGridError(String),

    #[error("market configuration invalid: {0}")]
    InvalidConfig(String),

    #[error("negative profit of {profit} for player {player_id}")]
    NegativeProfit { player_id: usize, profit: f64 },

    #[error("market not at equilibrium: deviation={deviation:.6}")]
    NotAtEquilibrium { deviation: f64 },

    #[error("empty market: no players configured")]
    EmptyMarket,

    #[error("cost exceeds price: cost={cost:.4}, price={price:.4}")]
    CostExceedsPrice { cost: f64, price: f64 },
}

#[derive(Debug, Error)]
pub enum SimulationError {
    #[error("simulation did not converge after {rounds} rounds (threshold={threshold})")]
    ConvergenceFailure { rounds: usize, threshold: f64 },

    #[error("invalid simulation config: {0}")]
    InvalidConfig(String),

    #[error("algorithm error: {0}")]
    AlgorithmError(String),

    #[error("trajectory too short: got {actual} rounds, need at least {minimum}")]
    TrajectoryTooShort { actual: usize, minimum: usize },

    #[error("segment split error: fractions do not sum to 1.0 (sum={sum})")]
    SegmentSplitError { sum: f64 },

    #[error("random seed collision: seed {seed} already used")]
    SeedCollision { seed: u64 },

    #[error("out of memory: attempted to allocate {bytes} bytes")]
    OutOfMemory { bytes: usize },

    #[error("timeout: simulation exceeded {seconds}s limit")]
    Timeout { seconds: f64 },

    #[error("numerical instability at round {round}: {details}")]
    NumericalInstability { round: usize, details: String },
}

#[derive(Debug, Error)]
pub enum StatisticalError {
    #[error("invalid p-value {value}: must be in [0, 1]")]
    InvalidPValue { value: f64 },

    #[error("invalid significance level {alpha}: must be in (0, 1)")]
    InvalidAlpha { alpha: f64 },

    #[error("insufficient sample size: got {actual}, need at least {minimum}")]
    InsufficientSample { actual: usize, minimum: usize },

    #[error("zero variance in sample of size {n}")]
    ZeroVariance { n: usize },

    #[error("alpha budget exhausted: spent {spent:.6} of {total:.6}")]
    AlphaBudgetExhausted { spent: f64, total: f64 },

    #[error("bootstrap failure: {0}")]
    BootstrapFailure(String),

    #[error("permutation test failure: {0}")]
    PermutationFailure(String),

    #[error("confidence interval invalid: lo={lo} > hi={hi}")]
    InvalidConfidenceInterval { lo: f64, hi: f64 },

    #[error("distribution error: {0}")]
    DistributionError(String),

    #[error("numerical overflow in test statistic computation")]
    NumericalOverflow,

    #[error("FWER correction failed: {0}")]
    FWERCorrectionFailed(String),

    #[error("multiple testing error: {num_tests} tests with alpha={alpha} is infeasible")]
    MultipleTesting { num_tests: usize, alpha: f64 },
}

#[derive(Debug, Error)]
pub enum CertificateError {
    #[error("certificate generation failed: {0}")]
    GenerationFailed(String),

    #[error("certificate verification failed: {0}")]
    VerificationFailed(String),

    #[error("integrity check failed: computed={computed}, expected={expected}")]
    IntegrityCheckFailed { computed: String, expected: String },

    #[error("insufficient evidence: got {num_items} items, need at least {minimum}")]
    InsufficientEvidence { num_items: usize, minimum: usize },

    #[error("certificate expired: generated={generated}, now={current}")]
    CertificateExpired { generated: String, current: String },

    #[error("invalid certificate format: {0}")]
    InvalidFormat(String),

    #[error("signing error: {0}")]
    SigningError(String),

    #[error("merkle proof invalid at depth {depth}: {reason}")]
    MerkleProofInvalid { depth: usize, reason: String },

    #[error("duplicate evidence item: {item_id}")]
    DuplicateEvidence { item_id: String },
}

#[derive(Debug, Error)]
pub enum GameTheoryError {
    #[error("no Nash equilibrium found after {iterations} iterations")]
    NoNashEquilibrium { iterations: usize },

    #[error("multiple equilibria found ({count}), expected unique")]
    MultipleEquilibria { count: usize },

    #[error("best response computation failed for player {player}: {reason}")]
    BestResponseFailed { player: usize, reason: String },

    #[error("deviation incentive computation failed: {0}")]
    DeviationIncentiveFailed(String),

    #[error("discount factor out of range: {delta} not in [0, 1)")]
    InvalidDiscountFactor { delta: f64 },

    #[error("payoff matrix dimension mismatch: expected {expected}×{expected}, got {rows}×{cols}")]
    PayoffDimensionMismatch { expected: usize, rows: usize, cols: usize },

    #[error("invalid strategy profile: {0}")]
    InvalidStrategyProfile(String),

    #[error("cooperation not sustainable: critical δ={critical_delta:.4} > δ={actual_delta:.4}")]
    CooperationNotSustainable { critical_delta: f64, actual_delta: f64 },

    #[error("repeated game analysis failed: {0}")]
    RepeatedGameError(String),
}

#[derive(Debug, Error)]
pub enum CounterfactualError {
    #[error("counterfactual simulation failed: {0}")]
    SimulationFailed(String),

    #[error("deviation scenario invalid: {0}")]
    InvalidScenario(String),

    #[error("punishment phase not detected after deviation at round {round}")]
    PunishmentNotDetected { round: usize },

    #[error("reward comparison failed: baseline={baseline:.4}, cf={counterfactual:.4}: {reason}")]
    RewardComparisonFailed { baseline: f64, counterfactual: f64, reason: String },

    #[error("insufficient counterfactual samples: got {actual}, need {minimum}")]
    InsufficientSamples { actual: usize, minimum: usize },

    #[error("oracle access insufficient: need {required}, have {available}")]
    InsufficientAccess { required: String, available: String },

    #[error("rewind failed at round {round}: {reason}")]
    RewindFailed { round: usize, reason: String },

    #[error("counterfactual timeout: scenario {scenario} exceeded {seconds}s")]
    Timeout { scenario: String, seconds: f64 },
}

#[derive(Debug, Error)]
pub enum OracleError {
    #[error("oracle query failed: {0}")]
    QueryFailed(String),

    #[error("state extraction failed at round {round}: {reason}")]
    StateExtractionFailed { round: usize, reason: String },

    #[error("checkpoint not available at round {round}")]
    CheckpointNotAvailable { round: usize },

    #[error("access level {requested} exceeds granted level {granted}")]
    AccessDenied { requested: String, granted: String },

    #[error("oracle not initialized")]
    NotInitialized,

    #[error("state snapshot corrupted at round {round}: {details}")]
    CorruptedSnapshot { round: usize, details: String },

    #[error("rewind not supported for algorithm {algorithm}")]
    RewindNotSupported { algorithm: String },
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Error context trait
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Trait for adding context to errors.
pub trait ErrorContext<T> {
    fn context(self, msg: impl Into<String>) -> CollusionResult<T>;
    fn with_context(self, f: impl FnOnce() -> String) -> CollusionResult<T>;
}

impl<T, E> ErrorContext<T> for Result<T, E>
where
    E: Into<CollusionError>,
{
    fn context(self, msg: impl Into<String>) -> CollusionResult<T> {
        self.map_err(|e| {
            let inner = e.into();
            CollusionError::Internal(format!("{}: {}", msg.into(), inner))
        })
    }

    fn with_context(self, f: impl FnOnce() -> String) -> CollusionResult<T> {
        self.map_err(|e| {
            let inner = e.into();
            CollusionError::Internal(format!("{}: {}", f(), inner))
        })
    }
}

impl<T> ErrorContext<T> for Option<T> {
    fn context(self, msg: impl Into<String>) -> CollusionResult<T> {
        self.ok_or_else(|| CollusionError::Internal(msg.into()))
    }

    fn with_context(self, f: impl FnOnce() -> String) -> CollusionResult<T> {
        self.ok_or_else(|| CollusionError::Internal(f()))
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Typed result aliases
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub type MarketResult<T> = Result<T, MarketError>;
pub type SimulationResult<T> = Result<T, SimulationError>;
pub type StatisticalResult<T> = Result<T, StatisticalError>;
pub type CertificateResult<T> = Result<T, CertificateError>;
pub type GameTheoryResult<T> = Result<T, GameTheoryError>;
pub type CounterfactualResult<T> = Result<T, CounterfactualError>;
pub type OracleResult<T> = Result<T, OracleError>;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Helper constructors
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl CollusionError {
    pub fn config(msg: impl Into<String>) -> Self { CollusionError::Config(msg.into()) }
    pub fn validation(msg: impl Into<String>) -> Self { CollusionError::Validation(msg.into()) }
    pub fn internal(msg: impl Into<String>) -> Self { CollusionError::Internal(msg.into()) }
    pub fn serialization(msg: impl Into<String>) -> Self { CollusionError::Serialization(msg.into()) }
    pub fn interval(msg: impl Into<String>) -> Self { CollusionError::IntervalArithmetic(msg.into()) }
    pub fn rational(msg: impl Into<String>) -> Self { CollusionError::RationalArithmetic(msg.into()) }

    pub fn is_recoverable(&self) -> bool {
        matches!(self,
            CollusionError::Statistical(_) | CollusionError::Config(_) | CollusionError::Validation(_))
    }

    pub fn is_internal(&self) -> bool {
        matches!(self, CollusionError::Internal(_))
    }
}

/// Structured error wrapper with context chain and source location.
#[derive(Debug)]
pub struct ErrorWithContext {
    pub error: CollusionError,
    pub context: Vec<String>,
    pub source_location: Option<String>,
}

impl ErrorWithContext {
    pub fn new(error: CollusionError) -> Self {
        ErrorWithContext { error, context: Vec::new(), source_location: None }
    }

    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.context.push(ctx.into()); self
    }

    pub fn with_location(mut self, loc: impl Into<String>) -> Self {
        self.source_location = Some(loc.into()); self
    }
}

impl fmt::Display for ErrorWithContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)?;
        for ctx in &self.context {
            write!(f, "\n  context: {}", ctx)?;
        }
        if let Some(ref loc) = self.source_location {
            write!(f, "\n  at: {}", loc)?;
        }
        Ok(())
    }
}

impl std::error::Error for ErrorWithContext {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_error_display() {
        let err = MarketError::InvalidPrice { price: -1.0, reason: "negative".into() };
        assert!(format!("{}", err).contains("negative"));
    }

    #[test]
    fn test_collusion_error_from_market() {
        let err: CollusionError = MarketError::EmptyMarket.into();
        assert!(format!("{}", err).contains("empty market"));
    }

    #[test]
    fn test_collusion_error_from_simulation() {
        let err: CollusionError = SimulationError::ConvergenceFailure { rounds: 1000, threshold: 0.01 }.into();
        assert!(format!("{}", err).contains("1000"));
    }

    #[test]
    fn test_statistical_error_display() {
        let err = StatisticalError::InvalidPValue { value: 1.5 };
        assert!(format!("{}", err).contains("1.5"));
    }

    #[test]
    fn test_certificate_error_integrity() {
        let err = CertificateError::IntegrityCheckFailed { computed: "abc".into(), expected: "def".into() };
        let msg = format!("{}", err);
        assert!(msg.contains("abc") && msg.contains("def"));
    }

    #[test]
    fn test_game_theory_error_display() {
        let err = GameTheoryError::InvalidDiscountFactor { delta: 1.5 };
        assert!(format!("{}", err).contains("1.5"));
    }

    #[test]
    fn test_counterfactual_error_display() {
        let err = CounterfactualError::PunishmentNotDetected { round: 42 };
        assert!(format!("{}", err).contains("42"));
    }

    #[test]
    fn test_oracle_error_display() {
        let err = OracleError::NotInitialized;
        assert!(format!("{}", err).contains("not initialized"));
    }

    #[test]
    fn test_error_context_result() {
        let r: Result<(), MarketError> = Err(MarketError::EmptyMarket);
        let c = r.context("during setup");
        assert!(format!("{}", c.unwrap_err()).contains("during setup"));
    }

    #[test]
    fn test_error_context_option() {
        let o: Option<i32> = None;
        let r = o.context("missing value");
        assert!(format!("{}", r.unwrap_err()).contains("missing value"));
    }

    #[test]
    fn test_error_is_recoverable() {
        assert!(CollusionError::config("bad").is_recoverable());
        assert!(!CollusionError::internal("bug").is_recoverable());
        assert!(!CollusionError::internal("bug").is_recoverable());
        assert!(CollusionError::internal("bug").is_internal());
    }

    #[test]
    fn test_error_with_context_display() {
        let ewc = ErrorWithContext::new(CollusionError::internal("boom"))
            .with_context("in test")
            .with_location("errors.rs:42");
        let msg = format!("{}", ewc);
        assert!(msg.contains("boom") && msg.contains("in test") && msg.contains("errors.rs:42"));
    }

    #[test]
    fn test_lazy_context() {
        let r: Result<(), MarketError> = Err(MarketError::EmptyMarket);
        let c = r.with_context(|| format!("step {}", 3));
        assert!(format!("{}", c.unwrap_err()).contains("step 3"));
    }

    #[test]
    fn test_oracle_error_conversion() {
        let err: CollusionError = OracleError::AccessDenied {
            requested: "Layer2".into(), granted: "Layer0".into(),
        }.into();
        assert!(format!("{}", err).contains("Layer2"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let err: CollusionError = io_err.into();
        assert!(format!("{}", err).contains("missing"));
    }
}
