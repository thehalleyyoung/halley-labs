"""
Time integration engine for CME on tensor-train format.

Implements multiple integration methods:
- TDVP (Time-Dependent Variational Principle) 1-site and 2-site
- TT-Krylov exponential integrator
- Numerical uniformization with Fox-Glynn truncation
- Adaptive Euler forward
- DMRG-like steady-state solver
"""

from tn_check.integrator.tdvp import (
    TDVPIntegrator,
    tdvp_one_site_sweep,
    tdvp_two_site_sweep,
)
from tn_check.integrator.krylov import (
    KrylovIntegrator,
    tt_krylov_step,
    lanczos_iteration,
)
from tn_check.integrator.uniformization import (
    UniformizationIntegrator,
    fox_glynn_weights,
    uniformization_step,
)
from tn_check.integrator.euler import (
    EulerIntegrator,
    euler_step,
    adaptive_euler_step,
)
from tn_check.integrator.steady_state import (
    SteadyStateSolver,
    dmrg_steady_state,
)
from tn_check.integrator.base import (
    IntegratorBase,
    IntegrationResult,
    TimePoint,
)

__all__ = [
    "TDVPIntegrator", "tdvp_one_site_sweep", "tdvp_two_site_sweep",
    "KrylovIntegrator", "tt_krylov_step", "lanczos_iteration",
    "UniformizationIntegrator", "fox_glynn_weights", "uniformization_step",
    "EulerIntegrator", "euler_step", "adaptive_euler_step",
    "SteadyStateSolver", "dmrg_steady_state",
    "IntegratorBase", "IntegrationResult", "TimePoint",
]
