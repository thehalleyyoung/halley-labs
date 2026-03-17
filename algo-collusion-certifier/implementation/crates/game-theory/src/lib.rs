//! Game-theoretic models, equilibrium computation, automaton theory, and the Folk Theorem
//! for the CollusionProof algorithmic collusion certification system.
//!
//! This crate provides:
//! - Finite-state automaton models for pricing strategies
//! - Concrete strategy implementations (Grim Trigger, Tit-for-Tat, etc.)
//! - Nash equilibrium computation (Bertrand, Cournot, mixed)
//! - Folk theorem analysis and the C3'/M8 dichotomy theorems
//! - Repeated game framework with discounting
//! - Collusion index and premium measurement
//! - Payoff computation and analysis
//! - Learning dynamics theory

pub mod automaton;
pub mod c3_validator;
pub mod collusion_index;
pub mod equilibrium;
pub mod folk_theorem;
pub mod learning;
pub mod payoff;
pub mod repeated_game;
pub mod strategies;

// ── Convenience re-exports ──────────────────────────────────────────────────

pub use automaton::{
    AutomatonBuilder, AutomatonMinimizer, AutomatonState, CycleDetector, FiniteStateStrategy,
    MealyMachine, ProductAutomaton, RecallBound, Transition,
};

pub use strategies::{
    AlwaysCooperate, AlwaysDefect, BoundedRecallStrategy, GrimTriggerStrategy, SoftMajority,
    Strategy, TitForTatStrategy, TitForTwoTatsStrategy, WinStayLoseShift,
};

pub use equilibrium::{
    BertrandNashSolver, BestResponseDynamics, CournotNashSolver, DominanceElimination,
    EquilibriumVerifier, IteratedBestResponse, MixedStrategyNE, NashEquilibrium, PayoffMatrix,
};

pub use folk_theorem::{
    CollusionDetectionTheorem, DeviationBound, DeviationProof, DiscountFactorAnalysis,
    FeasiblePayoffSet, FolkTheoremRegion, ImpossibilityProof, IndividuallyRationalPayoffSet,
    MinimaxComputation, PunishmentStrategy, StealthCollusionConstruction,
};

pub use c3_validator::{
    C3ValidationConfig, C3ValidationResult, C3Counterexample,
    validate_c3_exhaustive, validate_c3_default, format_validation_report,
};

pub use repeated_game::{
    AveragePayoff, DiscountedPayoff, PunishmentSeverity, RepeatedGame, RepeatedGameSimulator,
    StageGame, SubgamePerfectCheck, TriggerEquilibrium,
};

pub use collusion_index::{
    AbsoluteMargin, CollusionIndex, CollusionPremium, CollusionPremiumCI, CollusionSeverity,
    IntervalCollusionPremium, MonopolyBenchmark, SmoothCPTransition,
};

pub use payoff::{
    CooperativePayoff, IndividualRationality, PayoffComparison, PayoffInterpolation,
    PayoffNormalization, PayoffProfile, PayoffSpace, SocialWelfare,
};

pub use learning::{
    ConvergenceRate, CorrelationUnderCompetition, IndependentLearnerModel, LearningEquilibrium,
    NoRegretLearner, PriceCorrelationBound, QValueDynamics, RegretBound,
};
