"""
SMT Encoding & Z3 Interface.

Encodes refinement subtyping judgments as QF_UFLIA formulas with
finite-domain Tag sort. Interfaces with Z3 for satisfiability,
interpolant extraction, and model generation.
"""

from src.smt.encoder import (
    SMTEncoder,
    SMTContext,
    SubtypingQuery,
    SMTResult,
)

from src.smt.device_theory import HAS_Z3 as _HAS_Z3_DEVICE
from src.smt.phase_theory import HAS_Z3 as _HAS_Z3_PHASE

__all__ = [
    "SMTEncoder",
    "SMTContext",
    "SubtypingQuery",
    "SMTResult",
]

if _HAS_Z3_DEVICE:
    from src.smt.device_theory import (
        DevicePropagator,
        DeviceTheoryPlugin,
        DeviceSort,
        DEVICE_VALS,
        DEVICE_NAMES,
    )

    __all__ += [
        "DevicePropagator",
        "DeviceTheoryPlugin",
        "DeviceSort",
        "DEVICE_VALS",
        "DEVICE_NAMES",
    ]

if _HAS_Z3_PHASE:
    from src.smt.phase_theory import (
        PhasePropagator,
        PhaseTheoryPlugin,
    )

    __all__ += [
        "PhasePropagator",
        "PhaseTheoryPlugin",
    ]
