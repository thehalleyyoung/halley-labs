//! # BiCut Compiler
//!
//! The `bicut-compiler` crate implements a bilevel optimization compiler pipeline
//! that transforms [`BilevelProblem`](bicut_types::BilevelProblem) instances into
//! single-level mathematical programming reformulations suitable for standard
//! MIP/LP solvers.
//!
//! ## Architecture
//!
//! The compiler is organized as a multi-pass pipeline:
//!
//! 1. **Validation & Preprocessing** — structural analysis, bound tightening,
//!    constraint-qualification verification via [`bicut_core`].
//! 2. **Reformulation Pass** — one of four strategies selected by
//!    [`ReformulationType`]: KKT, strong duality, value function, or CCG.
//! 3. **Complementarity Encoding** — linearization of complementarity constraints
//!    using big-M, SOS1, or indicator constraints (see [`ComplementarityEncoding`]).
//! 4. **Emission** — lowering the reformulated model to a solver-specific
//!    representation selected by [`BackendTarget`].
//! 5. **Certificate Generation** — optional proof artifacts for verifiable
//!    optimization results.
//!
//! ## Supported Reformulations
//!
//! | Strategy | Module | Applicability |
//! |----------|--------|---------------|
//! | KKT | [`kkt_pass`] | Linear lower level with satisfied LICQ/MFCQ |
//! | Strong Duality | [`strong_duality_pass`] | Convex lower level (LP) |
//! | Value Function | [`value_function_pass`] | Parametric lower level with bounded value function |
//! | CCG | [`ccg_pass`] | General bilevel; iterative column-and-constraint generation |
//!
//! ## Quick Start
//!
//! ```no_run
//! use bicut_compiler::{compile, CompilerConfig, ReformulationType, BackendTarget};
//! use bicut_types::BilevelProblem;
//!
//! # fn example(problem: BilevelProblem) {
//! let config = CompilerConfig::new(ReformulationType::KKT, BackendTarget::Gurobi)
//!     .with_tolerance(1e-7)
//!     .with_certificate(true);
//! let result = compile(&problem, config).expect("compilation failed");
//! println!("Reformulated model has {} vars, {} constraints",
//!     result.stats.num_vars, result.stats.num_constraints);
//! # }
//! ```

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------

/// The top-level compiler pipeline that orchestrates all passes.
pub mod pipeline;

/// KKT (Karush-Kuhn-Tucker) reformulation pass.
pub mod kkt_pass;

/// Strong-duality reformulation pass.
pub mod strong_duality_pass;

/// Parametric value-function reformulation pass.
pub mod value_function_pass;

/// Column-and-constraint generation (CCG) iterative pass.
pub mod ccg_pass;

/// Model emission and format lowering.
pub mod emission;

/// Gurobi solver backend.
pub mod backend_gurobi;

/// SCIP solver backend.
pub mod backend_scip;

/// HiGHS solver backend.
pub mod backend_highs;

/// Verification certificate generation.
pub mod certificate_gen;

/// Big-M computation and estimation utilities.
pub mod bigm;

// ---------------------------------------------------------------------------
// Re-exports from submodules
// ---------------------------------------------------------------------------

pub use pipeline::{
    milp_to_lp, CompilerPipeline, IndicatorConstraint, MilpConstraint, MilpProblem, MilpVariable,
    PipelineResult, PipelineStage, Sos1Set, VarType,
};

pub use ccg_pass::CcgPass;
pub use kkt_pass::{KktPass, KktReformulation};
pub use strong_duality_pass::StrongDualityPass;
pub use value_function_pass::ValueFunctionPass;

pub use emission::{emit, EmissionConfig, EmissionResult, LpWriter, MpsWriter, OutputFormat};

pub use backend_gurobi::GurobiEmitter;
pub use backend_highs::HighsEmitter;
pub use backend_scip::ScipEmitter;

pub use certificate_gen::{CertificateGenerator, CompilerCertificate};

pub use bigm::{BigMComputer, BigMEstimate};

// ---------------------------------------------------------------------------
// Dependency re-exports (convenience for downstream consumers)
// ---------------------------------------------------------------------------

pub use bicut_core::{ProblemValidator, StructuralAnalysis, ValidationReport};
pub use bicut_lp::{solve_lp, LpError, LpSolver, SimplexSolver};
pub use bicut_types::{
    AffineFunction, BasisStatus, BilevelProblem, ConstraintIndex, ConstraintSense, Halfspace,
    LpProblem, LpSolution, LpStatus, OptDirection, Polyhedron, SparseEntry, SparseMatrix,
    ValidInequality, VarBound, VarIndex, DEFAULT_TOLERANCE,
};

// ---------------------------------------------------------------------------
// Imports used in this module
// ---------------------------------------------------------------------------

use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Crate-level error type
// ---------------------------------------------------------------------------

/// Errors that can arise during the bilevel compilation pipeline.
///
/// Each variant captures a distinct failure mode so that callers can
/// programmatically decide how to recover or report the problem.
#[derive(Debug, Error)]
pub enum CompilerError {
    /// The input bilevel problem failed structural validation.
    #[error("validation error: {0}")]
    Validation(String),

    /// The selected reformulation strategy is not applicable to the problem
    /// (e.g., KKT chosen but LICQ is violated).
    #[error("reformulation not applicable: {0}")]
    ReformulationNotApplicable(String),

    /// A numerical issue was encountered (ill-conditioning, near-zero pivots, etc.).
    #[error("numerical error: {0}")]
    Numerical(String),

    /// The lower-level problem is infeasible for the current leader decision.
    #[error("infeasible lower level: {0}")]
    InfeasibleLowerLevel(String),

    /// The lower-level problem is unbounded.
    #[error("unbounded lower level: {0}")]
    UnboundedLowerLevel(String),

    /// Big-M computation failed or produced unreliable estimates.
    #[error("big-M computation error: {0}")]
    BigMError(String),

    /// The target backend solver is unavailable or returned an error.
    #[error("backend error ({backend}): {detail}")]
    BackendError {
        /// Which backend was targeted.
        backend: BackendTarget,
        /// Description of the failure.
        detail: String,
    },

    /// Certificate generation or verification failed.
    #[error("certificate error: {0}")]
    CertificateError(String),

    /// The CCG iterative loop did not converge within the configured limit.
    #[error("CCG did not converge after {iterations} iterations (gap={gap:.2e})")]
    CcgNotConverged {
        /// Number of iterations executed.
        iterations: usize,
        /// Optimality gap at termination.
        gap: f64,
    },

    /// An LP sub-problem error propagated from [`bicut_lp`].
    #[error("LP error: {0}")]
    LpError(#[from] bicut_lp::LpError),

    /// An error propagated from an upstream dependency crate.
    #[error("upstream: {0}")]
    Upstream(#[from] anyhow::Error),

    /// An internal compiler error indicating a bug.
    #[error("internal compiler error: {0}")]
    Internal(String),

    /// The input MILP/problem is structurally invalid.
    #[error("invalid problem: {0}")]
    InvalidProblem(String),

    /// Emission to a solver file format failed.
    #[error("emission error: {0}")]
    Emission(String),

    /// An iterative algorithm did not converge.
    #[error("convergence error: {0}")]
    Convergence(String),
}

impl CompilerError {
    /// Creates a validation error from any displayable message.
    pub fn validation(msg: impl std::fmt::Display) -> Self {
        Self::Validation(msg.to_string())
    }

    /// Creates a numerical error from any displayable message.
    pub fn numerical(msg: impl std::fmt::Display) -> Self {
        Self::Numerical(msg.to_string())
    }

    /// Creates a backend error for the given target.
    pub fn backend(target: BackendTarget, detail: impl std::fmt::Display) -> Self {
        Self::BackendError {
            backend: target,
            detail: detail.to_string(),
        }
    }

    /// Creates an internal-error variant.
    pub fn internal(msg: impl std::fmt::Display) -> Self {
        Self::Internal(msg.to_string())
    }

    /// Returns `true` if this error indicates that an alternative reformulation
    /// strategy might succeed.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::ReformulationNotApplicable(_) | Self::CcgNotConverged { .. }
        )
    }
}

/// Convenience result alias for compiler operations.
pub type CompilerResult<T> = Result<T, CompilerError>;

// ---------------------------------------------------------------------------
// Shared enums
// ---------------------------------------------------------------------------

/// The reformulation strategy applied by the compiler to eliminate the
/// bilevel structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReformulationType {
    /// Replace the lower level with its Karush-Kuhn-Tucker optimality conditions,
    /// yielding complementarity constraints.
    KKT,
    /// Replace the lower level with a primal-dual strong-duality equality,
    /// valid when the lower level is a linear program.
    StrongDuality,
    /// Replace the lower level with its parametric value function, decomposing
    /// the problem into a single-level program over the leader variables.
    ValueFunction,
    /// Column-and-constraint generation: iteratively solve a relaxed master
    /// problem and a bilevel sub-problem, adding cuts until convergence.
    CCG,
}

impl std::fmt::Display for ReformulationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KKT => f.write_str("KKT"),
            Self::StrongDuality => f.write_str("Strong Duality"),
            Self::ValueFunction => f.write_str("Value Function"),
            Self::CCG => f.write_str("CCG"),
        }
    }
}

impl ReformulationType {
    /// Returns `true` if this strategy produces a single-shot (non-iterative)
    /// reformulation.
    pub fn is_single_shot(&self) -> bool {
        matches!(self, Self::KKT | Self::StrongDuality | Self::ValueFunction)
    }

    /// Returns `true` if this strategy requires an iterative solve loop.
    pub fn is_iterative(&self) -> bool {
        matches!(self, Self::CCG)
    }

    /// Lists all available reformulation types.
    pub fn all() -> &'static [ReformulationType] {
        &[
            Self::KKT,
            Self::StrongDuality,
            Self::ValueFunction,
            Self::CCG,
        ]
    }
}

/// Target solver backend for the emitted reformulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendTarget {
    /// Gurobi Optimizer (requires a Gurobi license).
    Gurobi,
    /// SCIP Optimization Suite.
    SCIP,
    /// HiGHS open-source LP/MIP solver.
    HiGHS,
    /// Generic MPS file output (solver-agnostic).
    GenericMps,
    /// Generic LP file output (solver-agnostic).
    GenericLp,
}

impl std::fmt::Display for BackendTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gurobi => f.write_str("Gurobi"),
            Self::SCIP => f.write_str("SCIP"),
            Self::HiGHS => f.write_str("HiGHS"),
            Self::GenericMps => f.write_str("MPS"),
            Self::GenericLp => f.write_str("LP"),
        }
    }
}

impl BackendTarget {
    /// Returns `true` if the backend writes a file rather than calling a solver API.
    pub fn is_file_based(&self) -> bool {
        matches!(self, Self::GenericMps | Self::GenericLp)
    }

    /// Returns `true` if the backend supports indicator constraints natively.
    pub fn supports_indicators(&self) -> bool {
        matches!(self, Self::Gurobi | Self::SCIP)
    }

    /// Returns `true` if the backend supports SOS1 constraints natively.
    pub fn supports_sos1(&self) -> bool {
        matches!(self, Self::Gurobi | Self::SCIP | Self::HiGHS)
    }

    /// Lists all available backend targets.
    pub fn all() -> &'static [BackendTarget] {
        &[
            Self::Gurobi,
            Self::SCIP,
            Self::HiGHS,
            Self::GenericMps,
            Self::GenericLp,
        ]
    }
}

/// Encoding strategy for complementarity constraints arising from KKT
/// reformulations.
///
/// The choice of encoding affects both model size and solver performance.
/// Big-M is the most portable but introduces large constants; SOS1 and
/// indicator constraints avoid big-M values but require solver support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplementarityEncoding {
    /// Big-M linearization: `s <= M(1 - z)` and `lambda <= Mz` for each pair.
    BigM,
    /// SOS1 (Special Ordered Set Type 1) encoding: `{s, lambda}` in an SOS1 set.
    SOS1,
    /// Indicator constraints: `z = 0 -> s = 0` and `z = 1 -> lambda = 0`.
    Indicator,
}

impl std::fmt::Display for ComplementarityEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BigM => f.write_str("Big-M"),
            Self::SOS1 => f.write_str("SOS1"),
            Self::Indicator => f.write_str("Indicator"),
        }
    }
}

impl ComplementarityEncoding {
    /// Returns `true` if this encoding requires big-M constants.
    pub fn needs_big_m(&self) -> bool {
        matches!(self, Self::BigM)
    }

    /// Returns the best encoding supported by the given backend.
    pub fn best_for(backend: BackendTarget) -> Self {
        if backend.supports_indicators() {
            Self::Indicator
        } else if backend.supports_sos1() {
            Self::SOS1
        } else {
            Self::BigM
        }
    }
}

// ---------------------------------------------------------------------------
// Big-M strategy (configuration-level enum)
// ---------------------------------------------------------------------------

/// Strategy for computing or supplying big-M values used in
/// complementarity linearization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BigMStrategy {
    /// Use a single fixed big-M value for every complementarity constraint.
    Fixed(f64),
    /// Automatically compute per-constraint big-M values from bound
    /// propagation and LP relaxations.
    Computed,
    /// Adaptive strategy: start from an initial estimate and refine
    /// iteratively by solving tightening sub-problems.
    Adaptive {
        /// Initial big-M estimate used as a starting point.
        initial: f64,
        /// Maximum number of tightening iterations.
        max_refinements: usize,
    },
}

impl Default for BigMStrategy {
    fn default() -> Self {
        Self::Computed
    }
}

impl std::fmt::Display for BigMStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(val) => write!(f, "Fixed({})", val),
            Self::Computed => f.write_str("Computed"),
            Self::Adaptive {
                initial,
                max_refinements,
            } => write!(f, "Adaptive(init={}, iters={})", initial, max_refinements),
        }
    }
}

// ---------------------------------------------------------------------------
// Compiler configuration
// ---------------------------------------------------------------------------

/// Configuration for the bilevel compiler pipeline.
///
/// Use the builder methods to customise individual settings while keeping
/// sensible defaults for the rest.
///
/// # Defaults
///
/// | Field | Default |
/// |-------|---------|
/// | `reformulation` | [`ReformulationType::KKT`] |
/// | `backend` | [`BackendTarget::GenericMps`] |
/// | `tolerance` | [`DEFAULT_TOLERANCE`] (1e-8) |
/// | `big_m_strategy` | [`BigMStrategy::Computed`] |
/// | `complementarity_encoding` | [`ComplementarityEncoding::BigM`] |
/// | `generate_certificate` | `false` |
/// | `max_iterations` | `100` |
/// | `verbosity` | `1` |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerConfig {
    /// The reformulation strategy to apply.
    pub reformulation: ReformulationType,
    /// The target solver backend for emission.
    pub backend: BackendTarget,
    /// Numerical tolerance for feasibility and optimality checks.
    pub tolerance: f64,
    /// Strategy for computing big-M values in complementarity encoding.
    pub big_m_strategy: BigMStrategy,
    /// Encoding method for complementarity constraints (relevant for KKT).
    pub complementarity_encoding: ComplementarityEncoding,
    /// Whether to generate a verification certificate alongside the
    /// reformulation.
    pub generate_certificate: bool,
    /// Maximum number of iterations for iterative methods (CCG).
    pub max_iterations: usize,
    /// Verbosity level: 0 = silent, 1 = summary, 2 = detailed, 3 = trace.
    pub verbosity: u8,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            reformulation: ReformulationType::KKT,
            backend: BackendTarget::GenericMps,
            tolerance: DEFAULT_TOLERANCE,
            big_m_strategy: BigMStrategy::default(),
            complementarity_encoding: ComplementarityEncoding::BigM,
            generate_certificate: false,
            max_iterations: 100,
            verbosity: 1,
        }
    }
}

impl CompilerConfig {
    /// Creates a new configuration with the given reformulation and backend,
    /// inheriting defaults for all other fields.
    pub fn new(reformulation: ReformulationType, backend: BackendTarget) -> Self {
        Self {
            reformulation,
            backend,
            ..Self::default()
        }
    }

    /// Sets the numerical tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Sets the big-M strategy.
    pub fn with_big_m_strategy(mut self, strategy: BigMStrategy) -> Self {
        self.big_m_strategy = strategy;
        self
    }

    /// Sets the complementarity encoding method.
    pub fn with_complementarity_encoding(mut self, encoding: ComplementarityEncoding) -> Self {
        self.complementarity_encoding = encoding;
        self
    }

    /// Enables or disables certificate generation.
    pub fn with_certificate(mut self, generate: bool) -> Self {
        self.generate_certificate = generate;
        self
    }

    /// Sets the maximum iteration count for iterative methods.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Sets the verbosity level.
    pub fn with_verbosity(mut self, level: u8) -> Self {
        self.verbosity = level;
        self
    }

    /// Validates this configuration, returning an error if any field is
    /// nonsensical (e.g. zero tolerance, zero max iterations for CCG).
    pub fn validate(&self) -> CompilerResult<()> {
        if self.tolerance <= 0.0 {
            return Err(CompilerError::validation(format!(
                "tolerance must be positive, got {}",
                self.tolerance
            )));
        }
        if self.tolerance > 1.0 {
            return Err(CompilerError::validation(format!(
                "tolerance {} is unreasonably large (> 1.0)",
                self.tolerance
            )));
        }
        if self.max_iterations == 0 && self.reformulation.is_iterative() {
            return Err(CompilerError::validation(
                "max_iterations must be > 0 for iterative reformulations",
            ));
        }
        if let BigMStrategy::Fixed(m) = &self.big_m_strategy {
            if *m <= 0.0 {
                return Err(CompilerError::validation(format!(
                    "fixed big-M must be positive, got {}",
                    m
                )));
            }
        }
        if let BigMStrategy::Adaptive { initial, .. } = &self.big_m_strategy {
            if *initial <= 0.0 {
                return Err(CompilerError::validation(format!(
                    "adaptive big-M initial value must be positive, got {}",
                    initial
                )));
            }
        }
        if self.complementarity_encoding == ComplementarityEncoding::Indicator
            && !self.backend.supports_indicators()
        {
            return Err(CompilerError::validation(format!(
                "backend {} does not support indicator constraints",
                self.backend
            )));
        }
        if self.complementarity_encoding == ComplementarityEncoding::SOS1
            && !self.backend.supports_sos1()
        {
            return Err(CompilerError::validation(format!(
                "backend {} does not support SOS1 constraints",
                self.backend
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Compilation result types
// ---------------------------------------------------------------------------

/// Summary statistics from a compilation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationStats {
    /// The reformulation strategy that was applied.
    pub reformulation_type: ReformulationType,
    /// Number of variables in the reformulated single-level model.
    pub num_vars: usize,
    /// Number of constraints in the reformulated single-level model.
    pub num_constraints: usize,
    /// Wall-clock compilation time in milliseconds.
    pub compilation_time_ms: u128,
    /// Number of iterations executed (for iterative methods such as CCG).
    pub iterations: Option<usize>,
    /// Number of big-M linearized complementarity constraints introduced.
    pub num_big_m_constraints: usize,
    /// Whether a verification certificate was generated.
    pub certificate_generated: bool,
}

impl std::fmt::Display for CompilationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CompilationStats {{ reformulation: {}, vars: {}, constraints: {}, time: {}ms",
            self.reformulation_type, self.num_vars, self.num_constraints, self.compilation_time_ms,
        )?;
        if let Some(iters) = self.iterations {
            write!(f, ", iterations: {}", iters)?;
        }
        if self.num_big_m_constraints > 0 {
            write!(f, ", big-M constraints: {}", self.num_big_m_constraints)?;
        }
        if self.certificate_generated {
            f.write_str(", certificate: yes")?;
        }
        f.write_str(" }")
    }
}

/// The complete result of compiling a bilevel problem into a single-level
/// reformulation.
#[derive(Debug, Clone)]
pub struct CompilationResult {
    /// The single-level reformulated LP/MIP problem, ready for a solver.
    pub reformulated_problem: LpProblem,
    /// Optional verification certificate attesting to the correctness of the
    /// reformulation.
    pub certificate: Option<certificate_gen::CompilerCertificate>,
    /// Compilation statistics and metrics.
    pub stats: CompilationStats,
    /// The compiler configuration that produced this result.
    pub config: CompilerConfig,
}

impl CompilationResult {
    /// Returns a reference to the reformulated problem.
    pub fn problem(&self) -> &LpProblem {
        &self.reformulated_problem
    }

    /// Returns `true` if a verification certificate is attached.
    pub fn has_certificate(&self) -> bool {
        self.certificate.is_some()
    }

    /// Returns the number of variables in the reformulated model.
    pub fn num_vars(&self) -> usize {
        self.reformulated_problem.num_vars
    }

    /// Returns the number of constraints in the reformulated model.
    pub fn num_constraints(&self) -> usize {
        self.reformulated_problem.num_constraints
    }
}

// ---------------------------------------------------------------------------
// Top-level compile entry point
// ---------------------------------------------------------------------------

/// Compiles a [`BilevelProblem`] into a single-level reformulation.
///
/// This is the primary entry point for the compiler. It validates the
/// configuration, constructs a [`CompilerPipeline`], executes all
/// compilation passes in sequence, and returns a [`CompilationResult`]
/// containing the reformulated model and optional verification certificate.
///
/// # Arguments
///
/// * `problem` - The bilevel optimization problem to compile.
/// * `config` - Compiler configuration controlling the reformulation strategy,
///   backend target, tolerances, and other options.
///
/// # Errors
///
/// Returns [`CompilerError`] if:
/// - The configuration is invalid ([`CompilerError::Validation`]).
/// - The input problem fails structural validation.
/// - The chosen reformulation is inapplicable to the problem structure.
/// - A numerical issue is encountered during compilation.
/// - The CCG loop fails to converge within the iteration limit.
pub fn compile(
    problem: &BilevelProblem,
    config: CompilerConfig,
) -> CompilerResult<CompilationResult> {
    config.validate()?;

    log::info!(
        "Starting bilevel compilation: reformulation={}, backend={}, tol={:.0e}",
        config.reformulation,
        config.backend,
        config.tolerance,
    );

    let start = Instant::now();

    let mut pipeline = pipeline::CompilerPipeline::new(config.clone());
    let pipeline_result = pipeline.run(problem)?;

    let elapsed = start.elapsed();

    log::info!(
        "Compilation completed in {}ms: {} vars, {} constraints",
        elapsed.as_millis(),
        pipeline_result.reformulated_problem.num_vars,
        pipeline_result.reformulated_problem.num_constraints,
    );

    let stats = CompilationStats {
        reformulation_type: config.reformulation,
        num_vars: pipeline_result.reformulated_problem.num_vars,
        num_constraints: pipeline_result.reformulated_problem.num_constraints,
        compilation_time_ms: elapsed.as_millis(),
        iterations: pipeline_result.iterations,
        num_big_m_constraints: pipeline_result.num_big_m_constraints,
        certificate_generated: pipeline_result.certificate.is_some(),
    };

    Ok(CompilationResult {
        reformulated_problem: pipeline_result.reformulated_problem,
        certificate: pipeline_result.certificate,
        stats,
        config,
    })
}
