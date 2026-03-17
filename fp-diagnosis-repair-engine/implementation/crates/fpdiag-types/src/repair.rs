//! Repair strategies, candidates, and certifications.
//!
//! Defines the vocabulary for expressing floating-point repairs: algebraic
//! rewrites, precision promotions, and composite repair plans.  These types
//! are produced by `fpdiag-repair` and consumed by `fpdiag-report`.

use crate::diagnosis::{DiagnosisCategory, DiagnosisSeverity};
use crate::eag::EagNodeId;
use crate::error_bounds::{ErrorBound, ErrorInterval, ErrorMetric};
use crate::source::SourceSpan;
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── RepairStrategy ─────────────────────────────────────────────────────────

/// A specific repair strategy from the Penumbra pattern library.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RepairStrategy {
    /// Kahan compensated summation for long reductions.
    KahanSummation,
    /// Pairwise (cascade) summation for array reductions.
    PairwiseSummation,
    /// Log-sum-exp stabilization for softmax-like patterns.
    LogSumExp,
    /// Compensated dot product (Ogita-Rump-Oishi).
    CompensatedDot,
    /// Welford's online algorithm for numerically stable variance.
    WelfordVariance,
    /// Reformulated quadratic formula: 2c / (−b ∓ √(b²−4ac)).
    StableQuadratic,
    /// Use `expm1(x)` instead of `exp(x) − 1` for small x.
    Expm1,
    /// Use `log1p(x)` instead of `log(1 + x)` for small x.
    Log1p,
    /// Use `hypot(a, b)` instead of `sqrt(a² + b²)`.
    Hypot,
    /// Mixed-precision promotion (promote specific operations to higher precision).
    PrecisionPromotion { target_bits: u32 },
    /// General algebraic rewrite with a named pattern.
    AlgebraicRewrite { pattern_name: String },
    /// Preconditioning for ill-conditioned linear systems.
    Preconditioning,
    /// Iterative refinement for linear solves.
    IterativeRefinement,
    /// No repair available; fall back to higher precision everywhere.
    FallbackPromotion,
}

impl RepairStrategy {
    /// Human-readable name.
    pub fn name(&self) -> &str {
        match self {
            Self::KahanSummation => "Kahan compensated summation",
            Self::PairwiseSummation => "pairwise summation",
            Self::LogSumExp => "log-sum-exp stabilization",
            Self::CompensatedDot => "compensated dot product (Ogita-Rump-Oishi)",
            Self::WelfordVariance => "Welford's algorithm",
            Self::StableQuadratic => "stable quadratic formula",
            Self::Expm1 => "expm1 substitution",
            Self::Log1p => "log1p substitution",
            Self::Hypot => "hypot substitution",
            Self::PrecisionPromotion { target_bits } => {
                // Return a static str for common cases
                match target_bits {
                    128 => "promote to quad precision (128-bit)",
                    106 => "promote to double-double",
                    _ => "precision promotion",
                }
            }
            Self::AlgebraicRewrite { .. } => "algebraic rewrite",
            Self::Preconditioning => "preconditioning",
            Self::IterativeRefinement => "iterative refinement",
            Self::FallbackPromotion => "fallback: full precision promotion",
        }
    }

    /// Which diagnosis categories this strategy addresses.
    pub fn addresses(&self) -> Vec<DiagnosisCategory> {
        match self {
            Self::KahanSummation | Self::PairwiseSummation => {
                vec![DiagnosisCategory::Absorption, DiagnosisCategory::Smearing]
            }
            Self::LogSumExp | Self::Expm1 | Self::Log1p | Self::StableQuadratic => {
                vec![DiagnosisCategory::CatastrophicCancellation]
            }
            Self::CompensatedDot => {
                vec![
                    DiagnosisCategory::Absorption,
                    DiagnosisCategory::AmplifiedRounding,
                ]
            }
            Self::WelfordVariance => {
                vec![
                    DiagnosisCategory::CatastrophicCancellation,
                    DiagnosisCategory::Absorption,
                ]
            }
            Self::Hypot => {
                vec![DiagnosisCategory::AmplifiedRounding]
            }
            Self::PrecisionPromotion { .. } | Self::FallbackPromotion => {
                vec![DiagnosisCategory::AmplifiedRounding]
            }
            Self::AlgebraicRewrite { .. } => {
                vec![DiagnosisCategory::CatastrophicCancellation]
            }
            Self::Preconditioning | Self::IterativeRefinement => {
                vec![DiagnosisCategory::IllConditionedSubproblem]
            }
        }
    }
}

impl fmt::Display for RepairStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ─── RepairCandidate ────────────────────────────────────────────────────────

/// A candidate repair: a strategy targeted at one or more EAG nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairCandidate {
    /// The repair strategy to apply.
    pub strategy: RepairStrategy,
    /// EAG nodes targeted by this repair.
    pub target_nodes: Vec<EagNodeId>,
    /// Source locations affected.
    pub source_spans: Vec<SourceSpan>,
    /// Estimated error reduction factor (e.g., 10.0 means 10× reduction).
    pub estimated_reduction: f64,
    /// Priority score (higher = more impactful).
    pub priority: f64,
    /// The diagnosis category that motivated this repair.
    pub motivated_by: DiagnosisCategory,
    /// Whether this is a primary repair or a fallback.
    pub is_fallback: bool,
}

impl RepairCandidate {
    /// Create a new candidate.
    pub fn new(
        strategy: RepairStrategy,
        target_nodes: Vec<EagNodeId>,
        motivated_by: DiagnosisCategory,
    ) -> Self {
        Self {
            strategy,
            target_nodes,
            source_spans: Vec::new(),
            estimated_reduction: 1.0,
            priority: 0.0,
            motivated_by,
            is_fallback: false,
        }
    }

    /// Mark as a fallback repair.
    pub fn as_fallback(mut self) -> Self {
        self.is_fallback = true;
        self
    }
}

// ─── RepairCertification ────────────────────────────────────────────────────

/// Certification result for a single repair application.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairCertification {
    /// Error interval before repair.
    pub original_error: ErrorInterval,
    /// Error interval after repair.
    pub repaired_error: ErrorInterval,
    /// Error reduction factor: original / repaired.
    pub reduction_factor: f64,
    /// Whether the certification is formal (interval arithmetic) or empirical.
    pub is_formal: bool,
    /// Certified error bound on the repaired computation.
    pub certified_bound: Option<ErrorBound>,
    /// Domain over which certification holds.
    pub certified_domain: Option<ErrorInterval>,
    /// Tier-1 coverage fraction in the certified region.
    pub tier1_coverage: f64,
}

impl RepairCertification {
    /// Whether the repair provably reduces error.
    pub fn is_improvement(&self) -> bool {
        self.repaired_error.width() < self.original_error.width() && self.reduction_factor > 1.0
    }
}

// ─── RepairResult ───────────────────────────────────────────────────────────

/// The result of applying a repair plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairResult {
    /// The repairs that were applied.
    pub applied_repairs: Vec<RepairCandidate>,
    /// Certification for each applied repair.
    pub certifications: Vec<RepairCertification>,
    /// Overall error reduction factor (geometric mean over certifications).
    pub overall_reduction: f64,
    /// Whether all repairs are formally certified.
    pub fully_certified: bool,
    /// Fraction of repairs that improved error.
    pub success_rate: f64,
    /// Wall-clock time for repair synthesis and certification (ms).
    pub repair_time_ms: u64,
}

impl RepairResult {
    /// Create an empty result.
    pub fn new() -> Self {
        Self {
            applied_repairs: Vec::new(),
            certifications: Vec::new(),
            overall_reduction: 1.0,
            fully_certified: true,
            success_rate: 0.0,
            repair_time_ms: 0,
        }
    }

    /// Add a repair and its certification.
    pub fn add(&mut self, repair: RepairCandidate, cert: RepairCertification) {
        if !cert.is_formal {
            self.fully_certified = false;
        }
        self.applied_repairs.push(repair);
        self.certifications.push(cert);
        self.recompute_stats();
    }

    /// Recompute aggregate statistics.
    fn recompute_stats(&mut self) {
        let n = self.certifications.len();
        if n == 0 {
            return;
        }
        let successes = self
            .certifications
            .iter()
            .filter(|c| c.is_improvement())
            .count();
        self.success_rate = successes as f64 / n as f64;
        // Geometric mean of reduction factors
        let log_sum: f64 = self
            .certifications
            .iter()
            .map(|c| c.reduction_factor.max(1.0).ln())
            .sum();
        self.overall_reduction = (log_sum / n as f64).exp();
    }
}

impl Default for RepairResult {
    fn default() -> Self {
        Self::new()
    }
}
