"""
Deployable mechanism implementations for DP-Forge.

This subpackage provides concrete mechanism classes that wrap the
probability tables, Gaussian covariance structures, and composed pipelines
produced by the DP-Forge CEGIS synthesis engine.  Each class exposes a
standard interface for sampling, density evaluation, loss computation,
and validity checking.

Modules:
    discrete    — :class:`DiscreteMechanism` for finite-output mechanisms.
    gaussian    — :class:`GaussianWorkloadMechanism` for Gaussian workload mechanisms.
    composed    — :class:`ComposedMechanism` for multi-mechanism composition.
    staircase   — :class:`StaircaseMechanism` for optimal pure-DP counting queries.
    matrix_mechanism — :class:`MatrixMechanism` for workload factorization.
    sparse_vector — Sparse vector technique (SVT) variants.
    pufferfish  — :class:`PufferfishMechanism` for Pufferfish privacy.
    truncated_laplace — Truncated and concentrated Laplace mechanisms.

Usage::

    from dp_forge.mechanisms import DiscreteMechanism, GaussianWorkloadMechanism
    from dp_forge.mechanisms import ComposedMechanism, StaircaseMechanism
    from dp_forge.mechanisms import MatrixMechanism, AboveThreshold

    mech = DiscreteMechanism(probability_table, output_grid, epsilon=1.0)
    sample = mech.sample(input_value=0)
    
    stair = StaircaseMechanism(epsilon=1.0)
    noisy = stair.sample(true_value=100)
"""

from dp_forge.mechanisms.discrete import DiscreteMechanism
from dp_forge.mechanisms.gaussian import GaussianWorkloadMechanism
from dp_forge.mechanisms.composed import ComposedMechanism
from dp_forge.mechanisms.staircase import (
    StaircaseMechanism,
    ProductStaircaseMechanism,
)
from dp_forge.mechanisms.matrix_mechanism import (
    MatrixMechanism,
    WorkloadFactorization,
)
from dp_forge.mechanisms.sparse_vector import (
    SparseVectorTechnique,
    AboveThreshold,
    NumericSVT,
    AdaptiveSVT,
    GapSVT,
    SVTComposition,
)
from dp_forge.mechanisms.pufferfish import (
    PufferfishFramework,
    PufferfishMechanism,
    DiscriminativePair,
    WassersteinMechanism,
)
from dp_forge.mechanisms.truncated_laplace import (
    TruncatedLaplaceMechanism,
    ConcentratedLaplace,
    CensoredLaplaceMechanism,
)

__all__ = [
    # Core mechanisms
    "DiscreteMechanism",
    "GaussianWorkloadMechanism",
    "ComposedMechanism",
    # Staircase
    "StaircaseMechanism",
    "ProductStaircaseMechanism",
    # Matrix mechanism
    "MatrixMechanism",
    "WorkloadFactorization",
    # Sparse vector technique
    "SparseVectorTechnique",
    "AboveThreshold",
    "NumericSVT",
    "AdaptiveSVT",
    "GapSVT",
    "SVTComposition",
    # Pufferfish
    "PufferfishFramework",
    "PufferfishMechanism",
    "DiscriminativePair",
    "WassersteinMechanism",
    # Truncated Laplace
    "TruncatedLaplaceMechanism",
    "ConcentratedLaplace",
    "CensoredLaplaceMechanism",
]
