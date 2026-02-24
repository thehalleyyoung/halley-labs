"""Statistics subpackage for finite-width phase diagram analysis.

Exports key classes for uncertainty quantification and hypothesis testing.
"""

from __future__ import annotations

from .hypothesis_tests import (
    MultipleTestCorrection,
    PhaseTransitionTester,
    TestResult,
)
from .uncertainty import (
    BoundaryUncertainty,
    UncertaintyBudget,
    UncertaintyQuantifier,
    UncertaintySource,
)

__all__ = [
    "BoundaryUncertainty",
    "MultipleTestCorrection",
    "PhaseTransitionTester",
    "TestResult",
    "UncertaintyBudget",
    "UncertaintyQuantifier",
    "UncertaintySource",
]
