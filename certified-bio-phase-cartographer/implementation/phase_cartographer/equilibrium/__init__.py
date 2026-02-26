"""
Equilibrium certification module.

Provides the Krawczyk operator for rigorous equilibrium verification,
parameter-dependent interval Newton methods, eigenvalue enclosure
for stability classification, and handling of near-singular Jacobians.
"""

from .krawczyk import KrawczykOperator, KrawczykResult
from .newton import IntervalNewton, ParametricNewton
from .stability import StabilityClassifier, StabilityType, EigenvalueEnclosure
from .certification import EquilibriumCertifier, EquilibriumCertificate

__all__ = [
    "KrawczykOperator", "KrawczykResult",
    "IntervalNewton", "ParametricNewton",
    "StabilityClassifier", "StabilityType", "EigenvalueEnclosure",
    "EquilibriumCertifier", "EquilibriumCertificate",
]
