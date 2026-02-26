"""
Bound composition engine for aggregating subgraph bounds into global network bounds.

Implements the composition theorem: given bounds [L_i, U_i] on individual subgraph
causal effects, compose them into valid global bounds [L, U] on the full network
causal effect, accounting for overlapping subgraphs and shared separator variables.
"""

from .composer import BoundComposer
from .gap_estimation import GapEstimator
from .consistency import SeparatorConsistencyChecker
from .propagation import MonotoneBoundPropagator
from .aggregation import GlobalBoundAggregator
from .theorem import CompositionTheorem
from .formal_proof import FormalProofEngine, FormalProofResult

__all__ = [
    "BoundComposer",
    "GapEstimator",
    "SeparatorConsistencyChecker",
    "MonotoneBoundPropagator",
    "GlobalBoundAggregator",
    "CompositionTheorem",
    "FormalProofEngine",
    "FormalProofResult",
]
