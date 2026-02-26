"""Soundness level tracking for verification results.

BioProver operates at multiple soundness levels depending on the
verification technique and underlying solver:

- SOUND: Full mathematical guarantee. Uses validated interval arithmetic
  and exact SMT solving. No approximation errors.
- DELTA_SOUND: Sound up to delta perturbation (dReal delta-satisfiability).
  The result holds for all states within delta of the boundary.
- BOUNDED: Sound within a bounded time horizon or bounded state space.
  Uses bounded model checking or finite-horizon reachability.
- APPROXIMATE: Uses approximations (GP surrogates, moment closure,
  linearization) that may introduce errors. Results are informative
  but not formally guaranteed.
"""

import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ErrorBudget:
    """End-to-end error budget tracking all approximation sources.

    Attributes:
        delta: dReal delta-satisfiability precision.
        epsilon: CEGIS counterexample-guided loop tolerance.
        truncation: Moment closure truncation error bound.
        discretization: ODE integration discretization error.
    """
    delta: float = 0.0
    epsilon: float = 0.0
    truncation: float = 0.0
    discretization: float = 0.0

    @property
    def combined(self) -> float:
        """Combined error bound (additive composition)."""
        return propagate_errors(self)

    def to_dict(self) -> dict:
        return {
            "delta": self.delta,
            "epsilon": self.epsilon,
            "truncation": self.truncation,
            "discretization": self.discretization,
            "combined": self.combined,
        }


def propagate_errors(budget: ErrorBudget) -> float:
    """Compute combined error from independent error sources.

    Uses root-sum-of-squares for independent errors, which gives a
    tighter bound than pure additive composition when errors are
    uncorrelated.
    """
    return math.sqrt(
        budget.delta ** 2
        + budget.epsilon ** 2
        + budget.truncation ** 2
        + budget.discretization ** 2
    )


class SoundnessLevel(Enum):
    SOUND = auto()
    DELTA_SOUND = auto()
    BOUNDED = auto()
    APPROXIMATE = auto()

    def __le__(self, other):
        if not isinstance(other, SoundnessLevel):
            return NotImplemented
        order = {SoundnessLevel.SOUND: 0, SoundnessLevel.DELTA_SOUND: 1,
                 SoundnessLevel.BOUNDED: 2, SoundnessLevel.APPROXIMATE: 3}
        return order[self] <= order[other]

    def __lt__(self, other):
        if not isinstance(other, SoundnessLevel):
            return NotImplemented
        return self <= other and self != other

    def __ge__(self, other):
        if not isinstance(other, SoundnessLevel):
            return NotImplemented
        return not self < other

    def __gt__(self, other):
        if not isinstance(other, SoundnessLevel):
            return NotImplemented
        return not self <= other

    @staticmethod
    def meet(a: 'SoundnessLevel', b: 'SoundnessLevel') -> 'SoundnessLevel':
        """Weakest (least sound) of two levels."""
        order = {SoundnessLevel.SOUND: 0, SoundnessLevel.DELTA_SOUND: 1,
                 SoundnessLevel.BOUNDED: 2, SoundnessLevel.APPROXIMATE: 3}
        if order[a] >= order[b]:
            return a
        return b


@dataclass
class SoundnessAnnotation:
    """Tracks soundness assumptions for a verification result."""
    level: SoundnessLevel
    assumptions: List[str] = field(default_factory=list)
    delta: Optional[float] = None  # For DELTA_SOUND
    time_bound: Optional[float] = None  # For BOUNDED
    approximation_error: Optional[float] = None  # For APPROXIMATE
    error_budget: Optional[ErrorBudget] = None

    def weaken_to(self, level: SoundnessLevel, reason: str) -> 'SoundnessAnnotation':
        new_level = SoundnessLevel.meet(self.level, level)
        return SoundnessAnnotation(
            level=new_level,
            assumptions=self.assumptions + [reason],
            delta=self.delta,
            time_bound=self.time_bound,
            approximation_error=self.approximation_error,
            error_budget=self.error_budget,
        )

    def with_delta(self, delta: float) -> 'SoundnessAnnotation':
        budget = self.error_budget or ErrorBudget()
        budget = ErrorBudget(
            delta=delta,
            epsilon=budget.epsilon,
            truncation=budget.truncation,
            discretization=budget.discretization,
        )
        return SoundnessAnnotation(
            level=SoundnessLevel.meet(self.level, SoundnessLevel.DELTA_SOUND),
            assumptions=self.assumptions + [f"dReal delta-satisfiability with delta={delta}"],
            delta=delta,
            time_bound=self.time_bound,
            approximation_error=self.approximation_error,
            error_budget=budget,
        )

    def with_time_bound(self, t: float) -> 'SoundnessAnnotation':
        return SoundnessAnnotation(
            level=SoundnessLevel.meet(self.level, SoundnessLevel.BOUNDED),
            assumptions=self.assumptions + [f"Bounded time horizon T={t}"],
            delta=self.delta,
            time_bound=t,
            approximation_error=self.approximation_error,
            error_budget=self.error_budget,
        )

    def with_error_budget(self, budget: ErrorBudget) -> 'SoundnessAnnotation':
        """Attach or update the error budget."""
        return SoundnessAnnotation(
            level=self.level,
            assumptions=self.assumptions,
            delta=self.delta,
            time_bound=self.time_bound,
            approximation_error=self.approximation_error,
            error_budget=budget,
        )
