"""Numerical linear algebra routines for phase diagram computation.

Provides spectral methods, matrix functions, and structured
matrix operations for NTK analysis and ODE integration.
"""

from .spectral import (
    SpectralDecomposition,
    RandomizedSVD,
    MatrixFunction,
    PseudoSpectrum,
)
from .matrix_ops import (
    KroneckerProduct,
    SylvesterSolver,
    LyapunovSolver,
    LowRankUpdate,
    MatrixBalancer,
)
