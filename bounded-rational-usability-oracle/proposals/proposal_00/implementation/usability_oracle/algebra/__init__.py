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

__all__ = [
    "CostElement",
    "CostExpression",
    "Leaf",
    "Sequential",
    "Parallel",
    "ContextMod",
    "SequentialComposer",
    "ParallelComposer",
    "ContextModulator",
    "CognitiveContext",
    "TaskGraphComposer",
    "SoundnessVerifier",
    "VerificationResult",
    "AlgebraicOptimizer",
]
