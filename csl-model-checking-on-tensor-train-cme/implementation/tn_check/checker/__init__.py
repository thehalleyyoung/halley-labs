"""
CSL model checker on tensor-train compressed probability vectors.

Implements satisfaction-set algebra, projected rate matrices,
fixpoint iteration with spectral-gap-informed convergence diagnostics,
three-valued semantics for nested probability operators, and
independent certificate verification via verification traces.
"""

from tn_check.checker.csl_ast import (
    CSLFormula,
    AtomicProp,
    TrueFormula,
    Negation,
    Conjunction,
    ProbabilityOp,
    SteadyStateOp,
    BoundedUntil,
    UnboundedUntil,
    Next,
    ComparisonOp,
)
from tn_check.checker.satisfaction import (
    SatisfactionResult,
    ThreeValued,
    compute_satisfaction_set,
)
from tn_check.checker.model_checker import CSLModelChecker, ConvergenceDiagnostics
from tn_check.checker.spectral import (
    estimate_spectral_gap,
    SpectralGapEstimate,
    adaptive_fallback_time_bound,
)

__all__ = [
    "CSLFormula",
    "AtomicProp",
    "TrueFormula",
    "Negation",
    "Conjunction",
    "ProbabilityOp",
    "SteadyStateOp",
    "BoundedUntil",
    "UnboundedUntil",
    "Next",
    "ComparisonOp",
    "SatisfactionResult",
    "ThreeValued",
    "compute_satisfaction_set",
    "CSLModelChecker",
    "ConvergenceDiagnostics",
    "estimate_spectral_gap",
    "SpectralGapEstimate",
    "adaptive_fallback_time_bound",
]
