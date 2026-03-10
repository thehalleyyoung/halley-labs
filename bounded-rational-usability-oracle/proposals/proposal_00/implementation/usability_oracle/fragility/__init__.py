"""
usability_oracle.fragility — Cognitive fragility analysis module.

Analyzes how sensitive usability is to changes in the rationality
parameter β, detecting *cliffs* (phase transitions in optimal strategy),
computing fragility scores, and assessing population impact.

Re-exports
----------
- :class:`FragilityResult`, :class:`CliffLocation`, :class:`SensitivityResult`
- :class:`InclusiveDesignResult`
- :class:`FragilityAnalyzer`
- :class:`CliffDetector`
- :class:`AdversarialAnalyzer`
- :class:`SensitivityAnalyzer`
- :class:`InclusiveDesignAnalyzer`
"""

from __future__ import annotations

from usability_oracle.fragility.models import (
    CliffLocation,
    FragilityResult,
    InclusiveDesignResult,
    Interval,
    SensitivityResult,
)
from usability_oracle.fragility.analyzer import FragilityAnalyzer
from usability_oracle.fragility.cliff import CliffDetector
from usability_oracle.fragility.adversarial import AdversarialAnalyzer
from usability_oracle.fragility.sensitivity import SensitivityAnalyzer
from usability_oracle.fragility.inclusive import InclusiveDesignAnalyzer

__all__ = [
    # models
    "FragilityResult",
    "CliffLocation",
    "SensitivityResult",
    "InclusiveDesignResult",
    "Interval",
    # analyzers
    "FragilityAnalyzer",
    "CliffDetector",
    "AdversarialAnalyzer",
    "SensitivityAnalyzer",
    "InclusiveDesignAnalyzer",
]
