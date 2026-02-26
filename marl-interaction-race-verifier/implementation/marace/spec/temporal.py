"""
Temporal logic over joint predicates.

Implements bounded temporal operators (Always, Eventually, Until, Next,
BoundedResponse) with both Boolean and quantitative (robustness)
semantics, online monitoring, and trace evaluation.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from marace.spec.predicates import Predicate


# ---------------------------------------------------------------------------
# Trace type alias
# ---------------------------------------------------------------------------

Trace = List[Dict[str, np.ndarray]]  # sequence of joint states


# ---------------------------------------------------------------------------
# Abstract temporal formula
# ---------------------------------------------------------------------------

class TemporalFormula(ABC):
    """Abstract base class for temporal logic formulas.

    Formulas are evaluated over finite *traces* — lists of joint states
    (each state is a dict mapping agent id to numpy array).  Both Boolean
    and quantitative (robustness) semantics are provided.

    Attributes:
        name: Human-readable label.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def evaluate(self, trace: Trace, t: int = 0) -> bool:
        """Evaluate the formula at position *t* of *trace*.

        Args:
            trace: Finite sequence of joint states.
            t: Starting time index.

        Returns:
            ``True`` iff the formula is satisfied.
        """

    @abstractmethod
    def robustness(self, trace: Trace, t: int = 0) -> float:
        """Quantitative robustness at position *t*.

        Positive ⇒ satisfied, negative ⇒ violated.
        """

    def __and__(self, other: "TemporalFormula") -> "Conjunction":
        return Conjunction(self, other)

    def __or__(self, other: "TemporalFormula") -> "Disjunction":
        return Disjunction(self, other)

    def __invert__(self) -> "Negation":
        return Negation(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"


# ---------------------------------------------------------------------------
# Predicate lift
# ---------------------------------------------------------------------------

class PredicateLift(TemporalFormula):
    """Lift a state predicate to a temporal formula (evaluated at time *t*)."""

    def __init__(self, predicate: Predicate) -> None:
        super().__init__(name=predicate.name)
        self.predicate = predicate

    def evaluate(self, trace: Trace, t: int = 0) -> bool:
        if t >= len(trace):
            return True  # vacuously true beyond trace
        return self.predicate.evaluate(trace[t])

    def robustness(self, trace: Trace, t: int = 0) -> float:
        if t >= len(trace):
            return float("inf")
        return self.predicate.robustness(trace[t])


# ---------------------------------------------------------------------------
# Boolean combinators
# ---------------------------------------------------------------------------

class Conjunction(TemporalFormula):
    def __init__(self, left: TemporalFormula, right: TemporalFormula) -> None:
        super().__init__(name=f"({left.name} ∧ {right.name})")
        self.left = left
        self.right = right

    def evaluate(self, trace: Trace, t: int = 0) -> bool:
        return self.left.evaluate(trace, t) and self.right.evaluate(trace, t)

    def robustness(self, trace: Trace, t: int = 0) -> float:
        return min(self.left.robustness(trace, t), self.right.robustness(trace, t))


class Disjunction(TemporalFormula):
    def __init__(self, left: TemporalFormula, right: TemporalFormula) -> None:
        super().__init__(name=f"({left.name} ∨ {right.name})")
        self.left = left
        self.right = right

    def evaluate(self, trace: Trace, t: int = 0) -> bool:
        return self.left.evaluate(trace, t) or self.right.evaluate(trace, t)

    def robustness(self, trace: Trace, t: int = 0) -> float:
        return max(self.left.robustness(trace, t), self.right.robustness(trace, t))


class Negation(TemporalFormula):
    def __init__(self, formula: TemporalFormula) -> None:
        super().__init__(name=f"¬{formula.name}")
        self.formula = formula

    def evaluate(self, trace: Trace, t: int = 0) -> bool:
        return not self.formula.evaluate(trace, t)

    def robustness(self, trace: Trace, t: int = 0) -> float:
        return -self.formula.robustness(trace, t)


# ---------------------------------------------------------------------------
# Temporal operators
# ---------------------------------------------------------------------------

class Always(TemporalFormula):
    """□[0,T] φ  —  *predicate* holds at every step in ``[t, t+horizon)``.

    If *predicate* is a plain ``Predicate`` it is automatically lifted.

    Attributes:
        predicate: Inner formula / predicate.
        horizon: Number of steps (``None`` = until end of trace).
    """

    def __init__(
        self,
        predicate: Any,
        horizon: Optional[int] = None,
        name: str = "",
    ) -> None:
        inner = _lift(predicate)
        super().__init__(name=name or f"□[0,{horizon}]({inner.name})")
        self.inner = inner
        self.horizon = horizon

    def evaluate(self, trace: Trace, t: int = 0) -> bool:
        end = len(trace) if self.horizon is None else min(t + self.horizon, len(trace))
        for k in range(t, end):
            if not self.inner.evaluate(trace, k):
                return False
        return True

    def robustness(self, trace: Trace, t: int = 0) -> float:
        end = len(trace) if self.horizon is None else min(t + self.horizon, len(trace))
        if t >= end:
            return float("inf")
        return min(self.inner.robustness(trace, k) for k in range(t, end))


class Eventually(TemporalFormula):
    """◇[0,T] φ  —  *predicate* holds at some step in ``[t, t+horizon)``.

    Attributes:
        predicate: Inner formula / predicate.
        horizon: Number of steps (``None`` = until end of trace).
    """

    def __init__(
        self,
        predicate: Any,
        horizon: Optional[int] = None,
        name: str = "",
    ) -> None:
        inner = _lift(predicate)
        super().__init__(name=name or f"◇[0,{horizon}]({inner.name})")
        self.inner = inner
        self.horizon = horizon

    def evaluate(self, trace: Trace, t: int = 0) -> bool:
        end = len(trace) if self.horizon is None else min(t + self.horizon, len(trace))
        for k in range(t, end):
            if self.inner.evaluate(trace, k):
                return True
        return False

    def robustness(self, trace: Trace, t: int = 0) -> float:
        end = len(trace) if self.horizon is None else min(t + self.horizon, len(trace))
        if t >= end:
            return -float("inf")
        return max(self.inner.robustness(trace, k) for k in range(t, end))


class Until(TemporalFormula):
    """φ₁ U[0,T] φ₂  —  *pred1* holds until *pred2* becomes true.

    Attributes:
        pred1: Formula that must hold until *pred2*.
        pred2: Formula that must eventually hold.
        horizon: Bounded horizon.
    """

    def __init__(
        self,
        pred1: Any,
        pred2: Any,
        horizon: Optional[int] = None,
        name: str = "",
    ) -> None:
        f1, f2 = _lift(pred1), _lift(pred2)
        super().__init__(name=name or f"({f1.name} U[0,{horizon}] {f2.name})")
        self.f1 = f1
        self.f2 = f2
        self.horizon = horizon

    def evaluate(self, trace: Trace, t: int = 0) -> bool:
        end = len(trace) if self.horizon is None else min(t + self.horizon, len(trace))
        for k in range(t, end):
            if self.f2.evaluate(trace, k):
                return True
            if not self.f1.evaluate(trace, k):
                return False
        return False

    def robustness(self, trace: Trace, t: int = 0) -> float:
        end = len(trace) if self.horizon is None else min(t + self.horizon, len(trace))
        best = -float("inf")
        for k in range(t, end):
            r2 = self.f2.robustness(trace, k)
            r1_min = min(
                self.f1.robustness(trace, j) for j in range(t, k)
            ) if k > t else float("inf")
            best = max(best, min(r1_min, r2))
        return best


class Next(TemporalFormula):
    """○ φ  —  *predicate* holds at the next step."""

    def __init__(self, predicate: Any, name: str = "") -> None:
        inner = _lift(predicate)
        super().__init__(name=name or f"○({inner.name})")
        self.inner = inner

    def evaluate(self, trace: Trace, t: int = 0) -> bool:
        return self.inner.evaluate(trace, t + 1)

    def robustness(self, trace: Trace, t: int = 0) -> float:
        return self.inner.robustness(trace, t + 1)


class BoundedResponse(TemporalFormula):
    """□(trigger → ◇[0,d] response).

    Whenever *trigger* is satisfied, *response* must be satisfied within
    *deadline* steps.

    Attributes:
        trigger: Trigger formula / predicate.
        response: Response formula / predicate.
        deadline: Maximum number of steps between trigger and response.
    """

    def __init__(
        self,
        trigger: Any,
        response: Any,
        deadline: int = 10,
        name: str = "",
    ) -> None:
        trig, resp = _lift(trigger), _lift(response)
        super().__init__(
            name=name or f"□({trig.name} → ◇[0,{deadline}] {resp.name})"
        )
        self.trigger = trig
        self.response = resp
        self.deadline = deadline

    def evaluate(self, trace: Trace, t: int = 0) -> bool:
        for k in range(t, len(trace)):
            if self.trigger.evaluate(trace, k):
                found = False
                for d in range(k, min(k + self.deadline + 1, len(trace))):
                    if self.response.evaluate(trace, d):
                        found = True
                        break
                if not found:
                    return False
        return True

    def robustness(self, trace: Trace, t: int = 0) -> float:
        rob = float("inf")
        for k in range(t, len(trace)):
            trig_rob = self.trigger.robustness(trace, k)
            if trig_rob >= 0:
                end = min(k + self.deadline + 1, len(trace))
                resp_rob = max(
                    self.response.robustness(trace, d)
                    for d in range(k, end)
                ) if end > k else -float("inf")
                # implication: max(-trig, resp)
                impl_rob = max(-trig_rob, resp_rob)
                rob = min(rob, impl_rob)
        return rob if rob != float("inf") else float("inf")


# ---------------------------------------------------------------------------
# Helper: lift Predicate → TemporalFormula
# ---------------------------------------------------------------------------

def _lift(p: Any) -> TemporalFormula:
    if isinstance(p, TemporalFormula):
        return p
    if isinstance(p, Predicate):
        return PredicateLift(p)
    raise TypeError(f"Cannot lift {type(p)} to TemporalFormula")


# ---------------------------------------------------------------------------
# Temporal formula evaluator
# ---------------------------------------------------------------------------

class TemporalFormulaEvaluator:
    """Batch evaluate temporal formulas over traces.

    Provides convenience methods for evaluating multiple formulas and
    collecting statistics.
    """

    def evaluate(
        self, formula: TemporalFormula, trace: Trace, t: int = 0
    ) -> bool:
        return formula.evaluate(trace, t)

    def robustness(
        self, formula: TemporalFormula, trace: Trace, t: int = 0
    ) -> float:
        return formula.robustness(trace, t)

    def evaluate_batch(
        self,
        formulas: List[TemporalFormula],
        trace: Trace,
    ) -> Dict[str, bool]:
        return {f.name: f.evaluate(trace) for f in formulas}

    def robustness_batch(
        self,
        formulas: List[TemporalFormula],
        trace: Trace,
    ) -> Dict[str, float]:
        return {f.name: f.robustness(trace) for f in formulas}

    def first_violation(
        self,
        formula: TemporalFormula,
        trace: Trace,
    ) -> Optional[int]:
        """Return the first time index at which *formula* is violated."""
        # For non-Always formulas this checks per-step evaluation
        for t in range(len(trace)):
            if not formula.evaluate(trace, t):
                return t
        return None


# ---------------------------------------------------------------------------
# Online monitor state
# ---------------------------------------------------------------------------

class MonitorState:
    """State for online monitoring of temporal formulas.

    Processes one state at a time and maintains sufficient history to
    evaluate bounded temporal formulas incrementally.

    Attributes:
        formula: Formula being monitored.
        horizon: Lookback buffer size.
    """

    def __init__(self, formula: TemporalFormula, horizon: int = 100) -> None:
        self.formula = formula
        self.horizon = horizon
        self._buffer: Trace = []
        self._violated: bool = False
        self._violation_time: Optional[int] = None
        self._step: int = 0

    def update(self, state: Dict[str, np.ndarray]) -> bool:
        """Process the next state.

        Returns:
            ``True`` if the formula is *currently* satisfied (over the
            available buffer).
        """
        self._buffer.append(state)
        if len(self._buffer) > self.horizon:
            self._buffer.pop(0)
        self._step += 1

        sat = self.formula.evaluate(self._buffer, 0)
        if not sat and not self._violated:
            self._violated = True
            self._violation_time = self._step
        return sat

    @property
    def violated(self) -> bool:
        return self._violated

    @property
    def violation_time(self) -> Optional[int]:
        return self._violation_time

    @property
    def current_robustness(self) -> float:
        if not self._buffer:
            return float("inf")
        return self.formula.robustness(self._buffer, 0)

    def reset(self) -> None:
        self._buffer.clear()
        self._violated = False
        self._violation_time = None
        self._step = 0

    def __repr__(self) -> str:
        return (
            f"MonitorState(formula={self.formula.name!r}, "
            f"step={self._step}, violated={self._violated})"
        )


# ---------------------------------------------------------------------------
# Robustness analysis
# ---------------------------------------------------------------------------

class Robustness:
    """Utilities for quantitative temporal-logic semantics."""

    @staticmethod
    def trace_robustness(formula: TemporalFormula, trace: Trace) -> float:
        """Overall robustness of *formula* over *trace*."""
        return formula.robustness(trace, 0)

    @staticmethod
    def robustness_profile(
        formula: TemporalFormula, trace: Trace
    ) -> List[float]:
        """Robustness at every time step."""
        return [formula.robustness(trace, t) for t in range(len(trace))]

    @staticmethod
    def min_robustness(
        formula: TemporalFormula, trace: Trace
    ) -> Tuple[int, float]:
        """Time step and value of minimum robustness."""
        profile = Robustness.robustness_profile(formula, trace)
        if not profile:
            return (0, float("inf"))
        idx = int(np.argmin(profile))
        return idx, profile[idx]

    @staticmethod
    def satisfaction_ratio(
        formula: TemporalFormula, trace: Trace
    ) -> float:
        """Fraction of time steps at which the formula is satisfied."""
        if not trace:
            return 1.0
        count = sum(1 for t in range(len(trace)) if formula.evaluate(trace, t))
        return count / len(trace)
