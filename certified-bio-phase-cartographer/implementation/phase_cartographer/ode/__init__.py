"""
Validated ODE integration engine.

Provides Taylor model ODE solver with flowpipe computation,
wrapping effect mitigation, and adaptive step-size control.
Specialized for biological rational right-hand sides.
"""

from .taylor_ode import TaylorODESolver, ODEResult
from .flowpipe import FlowpipeSegment, Flowpipe, compute_flowpipe
from .lohner import LohnerMethod
from .step_control import AdaptiveStepController
from .positivity import PositivityExploiter
from .rhs import ODERightHandSide, RationalRHS, PolynomialRHS

__all__ = [
    "TaylorODESolver", "ODEResult",
    "FlowpipeSegment", "Flowpipe", "compute_flowpipe",
    "LohnerMethod",
    "AdaptiveStepController",
    "PositivityExploiter",
    "ODERightHandSide", "RationalRHS", "PolynomialRHS",
]
