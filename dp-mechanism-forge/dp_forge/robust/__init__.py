"""
Robust CEGIS subpackage for numerical stability guarantees.

Provides a modified CEGIS loop that inflates privacy constraints by a safety
margin proportional to solver tolerance ν, guaranteeing (ε+ε_ν, δ+δ_ν)-DP
output where ε_ν = O(ν·e^ε) and δ_ν = O(ν).

Modules:
    - ``robust_cegis``: RobustCEGIS engine wrapping the standard CEGIS loop.
    - ``interval_arithmetic``: Interval number type with rigorous rounding.
    - ``perturbation_analysis``: LP perturbation bounds on privacy loss.
    - ``constraint_inflation``: Systematic constraint tightening.
    - ``solver_diagnostics``: Post-solve constraint auditing and refinement.
    - ``certified_output``: Certified mechanism wrapper with error bounds.

Public API
----------
- :class:`RobustCEGISEngine` — Main engine for robust synthesis.
- :class:`Interval` — Interval arithmetic type.
- :class:`IntervalMatrix` — Batch interval matrix operations.
- :class:`PerturbationAnalyzer` — LP perturbation theory analysis.
- :class:`ConstraintInflator` — Constraint tightening for safety margins.
- :class:`SolverDiagnostics` — Post-solve diagnostics and refinement.
- :class:`CertifiedMechanism` — Mechanism with numerical certificate.
"""

from dp_forge.robust.interval_arithmetic import Interval, IntervalMatrix
from dp_forge.robust.constraint_inflation import ConstraintInflator
from dp_forge.robust.perturbation_analysis import PerturbationAnalyzer
from dp_forge.robust.solver_diagnostics import SolverDiagnostics
from dp_forge.robust.certified_output import CertifiedMechanism
from dp_forge.robust.robust_cegis import RobustCEGISEngine

__all__ = [
    "RobustCEGISEngine",
    "Interval",
    "IntervalMatrix",
    "PerturbationAnalyzer",
    "ConstraintInflator",
    "SolverDiagnostics",
    "CertifiedMechanism",
]
