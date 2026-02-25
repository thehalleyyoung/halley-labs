"""Kernel engine for NTK computation, Nyström approximation, and kernel operations.

Provides exact and approximate NTK computation, spectral analysis,
kernel alignment, and low-rank approximation methods for finite-width
neural network analysis.
"""

from .ntk import NTKComputer, AnalyticNTK, EmpiricalNTK, NTKTracker
from .nystrom import NystromApproximation, LandmarkSelector, AdaptiveRankSelector
from .kernel_ops import (
    KernelMatrix,
    KernelAlignment,
    KernelPCA,
    KernelSpectralAnalysis,
)
