"""Causal discovery adapters and built-in structure learning.

Provides a unified interface for running causal discovery algorithms,
whether via external libraries (causal-learn, lingam) or built-in
implementations that require only numpy/scipy.

Modules
-------
adapters
    Wrappers for external causal discovery libraries.
estimator
    Parameter estimation for structural causal models.
structure_learning
    Built-in structure learning (constraint-based, score-based, hybrid).
"""

from cpa.discovery.adapters import (
    DiscoveryAdapter,
    PCAdapter,
    GESAdapter,
    LiNGAMAdapter,
    FallbackDiscovery,
)
from cpa.discovery.estimator import ParameterEstimator
from cpa.discovery.structure_learning import (
    ConstraintBasedLearner,
    ScoreBasedLearner,
    HybridLearner,
)

__all__ = [
    "DiscoveryAdapter",
    "PCAdapter",
    "GESAdapter",
    "LiNGAMAdapter",
    "FallbackDiscovery",
    "ParameterEstimator",
    "ConstraintBasedLearner",
    "ScoreBasedLearner",
    "HybridLearner",
]
