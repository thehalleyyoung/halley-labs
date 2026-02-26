"""
Coupled regime-causal inference module.

Provides EM-style alternation between regime estimation (Sticky HDP-HMM)
and causal discovery (PC algorithm with HSIC), with convergence monitoring,
identifiability analysis, and joint posterior computation.
"""

from .em_alternation import CoupledInference
from .convergence import ConvergenceAnalyzer
from .identifiability import IdentifiabilityAnalyzer
from .joint_posterior import JointPosterior

__all__ = [
    "CoupledInference",
    "ConvergenceAnalyzer",
    "IdentifiabilityAnalyzer",
    "JointPosterior",
]
