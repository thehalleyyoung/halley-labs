"""
HB-aware abstract interpretation framework for MARACE.

Zonotope abstract domains with happens-before consistency constraints
for sound verification of multi-agent reinforcement learning systems.
"""

from marace.abstract.zonotope import Zonotope
from marace.abstract.zonotope_reduction import (
    BoundedReduction,
    PCAMerging,
    ReductionChain,
    ReductionErrorCertificate,
    ChainErrorCertificate,
)
from marace.abstract.hb_constraints import (
    HBConstraint,
    HBConstraintSet,
    ConstraintGenerator,
    TimingConstraint,
    OrderingConstraint,
    ConstraintPropagation,
    ConsistencyChecker,
    ConstraintStrengthening,
)
from marace.abstract.transfer import (
    AbstractTransferFunction,
    ReLUTransfer,
    TanhTransfer,
    LinearTransfer,
    HBPruningTransfer,
    CompositionalTransfer,
    PrecisionTracker,
)
from marace.abstract.fixpoint import (
    FixpointEngine,
    IterationState,
    WideningStrategy,
    ConvergenceChecker,
    BoundedAscendingChain,
    FixpointResult,
    SoundOverApproximation,
    ParallelFixpoint,
)
from marace.abstract.domains import (
    IntervalDomain,
    HybridDomain,
    ConstrainedZonotope,
    AbstractDomainFactory,
    ProductDomain,
    interval_to_zonotope,
    zonotope_to_interval,
)

__all__ = [
    "Zonotope",
    "BoundedReduction",
    "PCAMerging",
    "ReductionChain",
    "ReductionErrorCertificate",
    "ChainErrorCertificate",
    "HBConstraint",
    "HBConstraintSet",
    "ConstraintGenerator",
    "TimingConstraint",
    "OrderingConstraint",
    "ConstraintPropagation",
    "ConsistencyChecker",
    "ConstraintStrengthening",
    "AbstractTransferFunction",
    "ReLUTransfer",
    "TanhTransfer",
    "LinearTransfer",
    "HBPruningTransfer",
    "CompositionalTransfer",
    "PrecisionTracker",
    "FixpointEngine",
    "IterationState",
    "WideningStrategy",
    "ConvergenceChecker",
    "BoundedAscendingChain",
    "FixpointResult",
    "SoundOverApproximation",
    "ParallelFixpoint",
    "IntervalDomain",
    "HybridDomain",
    "ConstrainedZonotope",
    "AbstractDomainFactory",
    "ProductDomain",
    "interval_to_zonotope",
    "zonotope_to_interval",
]
