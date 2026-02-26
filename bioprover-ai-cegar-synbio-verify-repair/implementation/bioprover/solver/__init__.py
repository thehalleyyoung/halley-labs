"""
BioProver Validated ODE Solver Module.

Provides interval arithmetic ODE integration and flowpipe computation
with mathematically rigorous enclosures for biological system verification.
"""

from bioprover.solver.interval import (
    Interval,
    IntervalVector,
    IntervalMatrix,
    hull,
    intersection,
    midpoint,
    width,
    radius,
    hausdorff_distance,
    subdivision,
)
from bioprover.solver.taylor_model import (
    TaylorModel,
    picard_lindelof,
    shrink_wrap,
)
from bioprover.solver.ode_integrator import (
    ValidatedODEIntegrator,
    IntegratorConfig,
)
from bioprover.solver.flowpipe import (
    FlowpipeSegment,
    Flowpipe,
)
from bioprover.solver.biology_extensions import (
    hill_function_interval,
    MonotoneSystemSolver,
    GRNSparseSolver,
    PositivityEnforcer,
    ConservationLawReducer,
    SteadyStateDetector,
    ContractionDetector,
    AdaptivePrecisionController,
)
from bioprover.solver.proof_certificate import (
    ProofCertificate,
    FlowpipeCertificate,
    InvariantCertificate,
)

__all__ = [
    "Interval",
    "IntervalVector",
    "IntervalMatrix",
    "hull",
    "intersection",
    "midpoint",
    "width",
    "radius",
    "hausdorff_distance",
    "subdivision",
    "TaylorModel",
    "picard_lindelof",
    "shrink_wrap",
    "ValidatedODEIntegrator",
    "IntegratorConfig",
    "FlowpipeSegment",
    "Flowpipe",
    "hill_function_interval",
    "MonotoneSystemSolver",
    "GRNSparseSolver",
    "PositivityEnforcer",
    "ConservationLawReducer",
    "SteadyStateDetector",
    "ContractionDetector",
    "AdaptivePrecisionController",
    "ProofCertificate",
    "FlowpipeCertificate",
    "InvariantCertificate",
]
