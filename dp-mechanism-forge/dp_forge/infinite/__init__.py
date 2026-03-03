"""
Infinite-dimensional LP subpackage for DP-Forge.

Implements cutting-plane methods for solving the continuous relaxation of the
mechanism design LP, where the output space is a continuous interval rather
than a finite grid.  The key idea: maintain a finite grid of output points,
solve the LP on that grid, use a dual oracle to find the most-violated
continuous point, add it to the grid, and repeat until convergence.

Modules
-------
- :mod:`cutting_plane` — InfiniteLPSolver: master LP + iterative grid enrichment.
- :mod:`dual_oracle` — DualOracle: continuous search for most-violated output.
- :mod:`convergence_monitor` — ConvergenceMonitor: bound tracking and termination.
- :mod:`continuous_relaxation` — ContinuousMechanism: density/CDF representation.
- :mod:`optimal_transport` — DPTransport: Wasserstein distance and transport plans.
- :mod:`duality` — InfiniteDualityChecker: dual construction and gap certification.
"""

from dp_forge.infinite.convergence_monitor import ConvergenceMonitor
from dp_forge.infinite.continuous_relaxation import ContinuousMechanism
from dp_forge.infinite.cutting_plane import InfiniteLPResult, InfiniteLPSolver
from dp_forge.infinite.dual_oracle import DualOracle
from dp_forge.infinite.duality import InfiniteDualityChecker
from dp_forge.infinite.optimal_transport import DPTransport

__all__ = [
    "InfiniteLPSolver",
    "InfiniteLPResult",
    "DualOracle",
    "ConvergenceMonitor",
    "ContinuousMechanism",
    "DPTransport",
    "InfiniteDualityChecker",
]
