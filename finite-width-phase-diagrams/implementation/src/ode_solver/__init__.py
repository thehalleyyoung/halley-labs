"""ODE solvers for kernel evolution and bifurcation analysis.

Provides adaptive integration of kernel ODEs, eigenvalue tracking,
and bifurcation detection for phase diagram computation.
"""

from .kernel_ode import KernelODESolver, ODETrajectory, IntegrationScheme
from .eigenvalue_tracker import EigenvalueTracker, SpectralPath, ZeroCrossing
from .bifurcation import (
    BifurcationDetector,
    BifurcationPoint,
    BifurcationType,
    NormalForm,
)
