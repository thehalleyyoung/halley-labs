"""
usability_oracle.algebra — Compositional cost algebra for usability analysis.

Implements the cost tuple algebra ``(μ, σ², κ, λ)`` with operators:

* **⊕** (sequential composition) — :mod:`.sequential`
* **⊗** (parallel composition) — :mod:`.parallel`
* **Δ** (context modulation) — :mod:`.context`

and supporting infrastructure:

* :mod:`.models` — :class:`CostElement`, :class:`CostExpression` tree
* :mod:`.composer` — task-graph-level composition using networkx
* :mod:`.soundness` — axiomatic verification of compositions
* :mod:`.optimizer` — algebraic simplification of expression trees

Category-theoretic and algebraic extensions:

* :mod:`.category` — monoidal category of cognitive states & cost morphisms
* :mod:`.semiring` — semiring family for all-pairs cost computation
* :mod:`.operad` — operadic task decomposition and cost propagation
* :mod:`.differential` — automatic differentiation through cost computations
* :mod:`.lattice` — cost lattice with fixed-point computation
* :mod:`.homomorphism` — structure-preserving maps between cost algebras
"""

from __future__ import annotations

from usability_oracle.algebra.models import (
    CostElement,
    CostExpression,
    Leaf,
    Sequential,
    Parallel,
    ContextMod,
)
from usability_oracle.algebra.sequential import SequentialComposer
from usability_oracle.algebra.parallel import ParallelComposer
from usability_oracle.algebra.context import ContextModulator, CognitiveContext
from usability_oracle.algebra.composer import TaskGraphComposer
from usability_oracle.algebra.soundness import SoundnessVerifier, VerificationResult
from usability_oracle.algebra.optimizer import AlgebraicOptimizer

# Category-theoretic foundations
from usability_oracle.algebra.category import (
    CognitiveState,
    CostMorphism,
    CostCategory,
)
from usability_oracle.algebra.semiring import (
    Semiring,
    TropicalSemiring,
    MaxPlusSemiring,
    LogSemiring,
    ViterbiSemiring,
    BooleanSemiring,
    ExpectedCostSemiring,
    ExpectedCostValue,
    IntervalSemiring,
    IntervalValue,
    SemiringMatrix,
    all_pairs_cost,
    cost_element_all_pairs,
)
from usability_oracle.algebra.operad import (
    Operation,
    OperadTerm,
    CognitiveOperad,
)
from usability_oracle.algebra.differential import (
    DualNumber,
    DualCostElement,
    dual_sequential_compose,
    dual_parallel_compose,
    HyperDualNumber,
    ReverseModeAD,
    cost_jacobian,
    sensitivity_report,
)
from usability_oracle.algebra.lattice import (
    cost_leq,
    cost_lt,
    cost_eq,
    cost_join,
    cost_meet,
    cost_join_many,
    cost_meet_many,
    cost_bottom,
    cost_top,
    kleene_fixpoint,
    tarski_fixpoint,
    widen,
    narrow,
    widened_fixpoint,
    GaloisConnection,
    variance_abstraction,
    tail_risk_abstraction,
)
from usability_oracle.algebra.homomorphism import (
    CostAlgebraHomomorphism,
    verify_sequential_homomorphism,
    verify_parallel_homomorphism,
    verify_identity_preservation,
    mean_projection,
    moment_projection,
    scale_homomorphism,
    log_transform,
    bits_to_seconds,
    CostGaloisConnection,
    moment_galois_connection,
    QuotientAlgebra,
    mean_quotient,
    moment_quotient,
    kernel,
    image,
    image_unique,
    prove_abstraction_soundness,
)

__all__ = [
    # Core models
    "CostElement",
    "CostExpression",
    "Leaf",
    "Sequential",
    "Parallel",
    "ContextMod",
    # Composition operators
    "SequentialComposer",
    "ParallelComposer",
    "ContextModulator",
    "CognitiveContext",
    "TaskGraphComposer",
    # Verification and optimisation
    "SoundnessVerifier",
    "VerificationResult",
    "AlgebraicOptimizer",
    # Category
    "CognitiveState",
    "CostMorphism",
    "CostCategory",
    # Semiring
    "Semiring",
    "TropicalSemiring",
    "MaxPlusSemiring",
    "LogSemiring",
    "ViterbiSemiring",
    "BooleanSemiring",
    "ExpectedCostSemiring",
    "ExpectedCostValue",
    "IntervalSemiring",
    "IntervalValue",
    "SemiringMatrix",
    "all_pairs_cost",
    "cost_element_all_pairs",
    # Operad
    "Operation",
    "OperadTerm",
    "CognitiveOperad",
    # Differential
    "DualNumber",
    "DualCostElement",
    "dual_sequential_compose",
    "dual_parallel_compose",
    "HyperDualNumber",
    "ReverseModeAD",
    "cost_jacobian",
    "sensitivity_report",
    # Lattice
    "cost_leq",
    "cost_lt",
    "cost_eq",
    "cost_join",
    "cost_meet",
    "cost_join_many",
    "cost_meet_many",
    "cost_bottom",
    "cost_top",
    "kleene_fixpoint",
    "tarski_fixpoint",
    "widen",
    "narrow",
    "widened_fixpoint",
    "GaloisConnection",
    "variance_abstraction",
    "tail_risk_abstraction",
    # Homomorphism
    "CostAlgebraHomomorphism",
    "verify_sequential_homomorphism",
    "verify_parallel_homomorphism",
    "verify_identity_preservation",
    "mean_projection",
    "moment_projection",
    "scale_homomorphism",
    "log_transform",
    "bits_to_seconds",
    "CostGaloisConnection",
    "moment_galois_connection",
    "QuotientAlgebra",
    "mean_quotient",
    "moment_quotient",
    "kernel",
    "image",
    "image_unique",
    "prove_abstraction_soundness",
]
