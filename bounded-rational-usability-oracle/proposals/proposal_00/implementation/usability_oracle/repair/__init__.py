"""
usability_oracle.repair — SMT-backed UI repair synthesis.

Provides automated generation of usability fixes using constraint solving
(Z3), cognitive-law-based constraint encoding, and mutation operators over
accessibility trees.
"""

from usability_oracle.repair.models import (
    RepairCandidate,
    RepairConstraint,
    RepairResult,
    UIMutation,
)
from usability_oracle.repair.synthesizer import RepairSynthesizer
from usability_oracle.repair.constraints import ConstraintEncoder
from usability_oracle.repair.mutations import MutationOperator
from usability_oracle.repair.validator import RepairValidator, ValidationResult
from usability_oracle.repair.strategies import RepairStrategySelector

__all__ = [
    "RepairCandidate",
    "RepairConstraint",
    "RepairResult",
    "UIMutation",
    "RepairSynthesizer",
    "ConstraintEncoder",
    "MutationOperator",
    "RepairValidator",
    "ValidationResult",
    "RepairStrategySelector",
]
