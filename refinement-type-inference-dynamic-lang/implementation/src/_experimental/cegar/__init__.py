"""
CEGAR Engine — Counterexample-Guided Abstraction Refinement.

Two-phase loop: (1) abstract interpretation with current predicates,
(2) counterexample analysis and predicate refinement via SMT.
"""

from src.cegar.engine import CEGAREngine, CEGARResult, CEGARConfig
from src.cegar.guard_harvesting import GuardHarvester, HarvestedGuard

__all__ = [
    "CEGAREngine",
    "CEGARResult",
    "CEGARConfig",
    "GuardHarvester",
    "HarvestedGuard",
]
