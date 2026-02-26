"""
Portfolio optimization module for Causal-Shielded Adaptive Trading.

Provides mean-variance optimization under shield constraints, causal feature
selection for return estimation, discrete action space management, and
formal composition theorem verification.
"""

from .mean_variance import ShieldedMeanVarianceOptimizer
from .causal_features import CausalFeatureSelector
from .action_space import ActionSpace
from .composition import CompositionTheorem

__all__ = [
    "ShieldedMeanVarianceOptimizer",
    "CausalFeatureSelector",
    "ActionSpace",
    "CompositionTheorem",
]
