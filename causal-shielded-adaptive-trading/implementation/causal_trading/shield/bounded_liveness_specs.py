"""
Bounded liveness specifications for Causal-Shielded Adaptive Trading.

Instantiates concrete bounded liveness LTL specifications using the
existing BoundedLTL infrastructure.  The existing codebase uses only
trivial G(φ) safety properties; these specs add genuine bounded
liveness of the form:

    G(trigger → F[0, horizon](recovery))

This pattern ("if something bad happens, it must be corrected within
a bounded horizon") is strictly more expressive than pure safety and
exercises the full power of the bounded temporal logic checker.

Spec catalogue
--------------
DrawdownRecoverySpec   – "Recover from drawdown within H steps"
LossRecoverySpec       – "Reduce position after large loss within H steps"
PositionReductionSpec  – "Bring exposure below safe level within H steps"
RegimeTransitionSpec   – "Adapt strategy after regime change within H steps"

BoundedLivenessLibrary – factory for standard spec suites
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from causal_trading.shield.safety_specs import (
    LTLFormula,
    LTLOp,
    SafetySpecification,
    TrajectoryChecker,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Trajectory evaluation result
# -----------------------------------------------------------------------

@dataclass
class TrajectoryResult:
    """Result of evaluating a specification on a trajectory.

    Attributes
    ----------
    satisfied : bool
        Whether the trajectory satisfies the specification.
    violation_times : list of int
        Time steps where the trigger fired and recovery failed.
    trigger_count : int
        Number of times the trigger condition occurred.
    recovery_count : int
        Number of triggers that were successfully recovered from.
    robustness : float
        Quantitative robustness value (positive = satisfied).
    details : dict
        Additional per-time information.
    """
    satisfied: bool
    violation_times: List[int] = field(default_factory=list)
    trigger_count: int = 0
    recovery_count: int = 0
    robustness: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------
# Base bounded-liveness specification
# -----------------------------------------------------------------------

class BoundedLivenessSpec(SafetySpecification):
    """Abstract base for bounded liveness specifications.

    Encodes the pattern G(trigger → F[0, horizon](recovery)).
    Subclasses define the trigger and recovery predicates.

    Parameters
    ----------
    name : str
        Human-readable name.
    horizon : int
        Maximum number of steps within which recovery must occur.
    """

    def __init__(self, name: str, horizon: int) -> None:
        super().__init__(name)
        self.horizon = horizon

    # -- Template methods for subclasses --

    def trigger_predicate(self, state: Dict[str, float]) -> bool:
        """Return True when the "bad" condition fires."""
        raise NotImplementedError

    def recovery_predicate(self, state: Dict[str, float]) -> bool:
        """Return True when recovery is achieved."""
        raise NotImplementedError

    # -- SafetySpecification interface --

    def check(self, trajectory: List[Dict[str, float]]) -> bool:
        """Check if the trajectory satisfies the bounded liveness spec."""
        result = self.evaluate_trajectory(trajectory)
        return result.satisfied

    def is_satisfied(self, state: Dict[str, float]) -> bool:
        """Check instantaneous satisfaction (recovery predicate)."""
        return self.recovery_predicate(state)

    def to_ltl(self) -> LTLFormula:
        """Build G(trigger → F[0,H](recovery)) formula.

        Uses the closure-based LTLFormula from safety_specs.py.
        """
        trigger = LTLFormula(
            op=LTLOp.ATOM,
            atom=f"{self.name}_trigger",
            predicate=self.trigger_predicate,
        )
        recovery = LTLFormula(
            op=LTLOp.ATOM,
            atom=f"{self.name}_recovery",
            predicate=self.recovery_predicate,
        )
        eventually_recover = LTLFormula.eventually(recovery, bound=self.horizon)
        implication = LTLFormula.implies(trigger, eventually_recover)
        return LTLFormula.always(implication)

    def to_ltl_formula(self) -> str:
        """Return a parseable BoundedLTL formula string."""
        t = f"{self.name}_trigger"
        r = f"{self.name}_recovery"
        return f"G(({t}) -> F[0,{self.horizon}]({r}))"

    def get_constraints(self) -> List[Callable]:
        """State-level constraints: trigger implies eventual recovery."""
        return [self.recovery_predicate]

    def evaluate_trajectory(
        self,
        trajectory: List[Dict[str, float]],
    ) -> TrajectoryResult:
        """Evaluate the bounded liveness spec on a full trajectory.

        For every time step t where trigger fires, checks that recovery
        holds at some t' ∈ [t, t + horizon).
        """
        n = len(trajectory)
        violation_times: List[int] = []
        trigger_count = 0
        recovery_count = 0

        for t in range(n):
            if self.trigger_predicate(trajectory[t]):
                trigger_count += 1
                recovered = False
                end = min(t + self.horizon, n)
                for t2 in range(t, end):
                    if self.recovery_predicate(trajectory[t2]):
                        recovered = True
                        break
                if recovered:
                    recovery_count += 1
                else:
                    violation_times.append(t)

        satisfied = len(violation_times) == 0

        # Quantitative robustness via TrajectoryChecker
        robustness = 0.0
        if trajectory:
            checker = TrajectoryChecker(trajectory)
            robustness = checker.robustness(self.to_ltl())

        return TrajectoryResult(
            satisfied=satisfied,
            violation_times=violation_times,
            trigger_count=trigger_count,
            recovery_count=recovery_count,
            robustness=robustness,
            details={
                "horizon": self.horizon,
                "trajectory_length": n,
            },
        )


# -----------------------------------------------------------------------
# Concrete specifications
# -----------------------------------------------------------------------

class DrawdownRecoverySpec(BoundedLivenessSpec):
    """G(drawdown > threshold → F[0, H](drawdown < recovery_level))

    "If drawdown exceeds *threshold*, it must recover below
    *recovery_level* within *horizon* steps."

    Parameters
    ----------
    threshold : float
        Drawdown level that triggers the recovery obligation (e.g., 0.05).
    recovery_level : float
        Drawdown level that counts as recovered (e.g., 0.02).
    horizon : int
        Maximum steps for recovery.
    drawdown_key : str
        Key in the state dict for the current drawdown value.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        recovery_level: float = 0.02,
        horizon: int = 20,
        drawdown_key: str = "drawdown",
    ) -> None:
        name = f"DrawdownRecovery(thr={threshold},rec={recovery_level},H={horizon})"
        super().__init__(name, horizon)
        self.threshold = threshold
        self.recovery_level = recovery_level
        self.drawdown_key = drawdown_key

    def trigger_predicate(self, state: Dict[str, float]) -> bool:
        return state.get(self.drawdown_key, 0.0) > self.threshold

    def recovery_predicate(self, state: Dict[str, float]) -> bool:
        return state.get(self.drawdown_key, 0.0) < self.recovery_level

    def to_ltl_formula(self) -> str:
        return (
            f"G(({self.drawdown_key}_gt_{self.threshold}) "
            f"-> F[0,{self.horizon}]({self.drawdown_key}_lt_{self.recovery_level}))"
        )

    def get_safe_state_mask(self, n_states: int) -> np.ndarray:
        """States below recovery level are safe."""
        mask = np.ones(n_states, dtype=bool)
        for s in range(n_states):
            dd = s / max(n_states - 1, 1)
            if dd > self.threshold:
                mask[s] = False
        return mask


class LossRecoverySpec(BoundedLivenessSpec):
    """G(loss > threshold → F[0, H](recovered))

    "If unrealised loss exceeds *threshold*, position must be reduced
    within *horizon* steps so that loss is below *recovery_level*."

    Parameters
    ----------
    threshold : float
        Loss level triggering recovery (e.g., 0.03 for 3%).
    recovery_level : float
        Loss level considered recovered.
    horizon : int
        Steps allowed for recovery.
    loss_key : str
        State dict key for current loss.
    """

    def __init__(
        self,
        threshold: float = 0.03,
        recovery_level: float = 0.01,
        horizon: int = 5,
        loss_key: str = "loss",
    ) -> None:
        name = f"LossRecovery(thr={threshold},rec={recovery_level},H={horizon})"
        super().__init__(name, horizon)
        self.threshold = threshold
        self.recovery_level = recovery_level
        self.loss_key = loss_key

    def trigger_predicate(self, state: Dict[str, float]) -> bool:
        return state.get(self.loss_key, 0.0) > self.threshold

    def recovery_predicate(self, state: Dict[str, float]) -> bool:
        return state.get(self.loss_key, 0.0) < self.recovery_level

    def to_ltl_formula(self) -> str:
        return (
            f"G(({self.loss_key}_gt_{self.threshold}) "
            f"-> F[0,{self.horizon}]({self.loss_key}_lt_{self.recovery_level}))"
        )

    def get_safe_state_mask(self, n_states: int) -> np.ndarray:
        mask = np.ones(n_states, dtype=bool)
        thresh_idx = int(self.threshold * n_states)
        mask[thresh_idx:] = False
        return mask


class PositionReductionSpec(BoundedLivenessSpec):
    """G(exposure > limit → F[0, H](exposure < safe_level))

    "If total exposure exceeds *limit*, it must be brought below
    *safe_level* within *horizon* steps."

    Parameters
    ----------
    limit : float
        Exposure level triggering reduction.
    safe_level : float
        Exposure level considered safe.
    horizon : int
        Steps allowed.
    exposure_key : str
        State dict key for exposure.
    """

    def __init__(
        self,
        limit: float = 100.0,
        safe_level: float = 80.0,
        horizon: int = 10,
        exposure_key: str = "exposure",
    ) -> None:
        name = f"PositionReduction(lim={limit},safe={safe_level},H={horizon})"
        super().__init__(name, horizon)
        self.limit = limit
        self.safe_level = safe_level
        self.exposure_key = exposure_key

    def trigger_predicate(self, state: Dict[str, float]) -> bool:
        return state.get(self.exposure_key, 0.0) > self.limit

    def recovery_predicate(self, state: Dict[str, float]) -> bool:
        return state.get(self.exposure_key, 0.0) < self.safe_level

    def to_ltl_formula(self) -> str:
        return (
            f"G(({self.exposure_key}_gt_{self.limit}) "
            f"-> F[0,{self.horizon}]({self.exposure_key}_lt_{self.safe_level}))"
        )

    def get_safe_state_mask(self, n_states: int) -> np.ndarray:
        mask = np.ones(n_states, dtype=bool)
        threshold_idx = int(self.limit / (self.limit + 50) * n_states)
        mask[threshold_idx:] = False
        return mask


class RegimeTransitionSpec(BoundedLivenessSpec):
    """G(regime_change → F[0, adaptation_window](strategy_adapted))

    "After a regime change is detected, the trading strategy must
    adapt within *adaptation_window* steps."

    Parameters
    ----------
    adaptation_window : int
        Steps allowed for adaptation after regime change.
    regime_change_key : str
        State dict key indicating a regime change occurred (bool-like).
    adapted_key : str
        State dict key indicating strategy has adapted (bool-like).
    """

    def __init__(
        self,
        adaptation_window: int = 10,
        regime_change_key: str = "regime_change",
        adapted_key: str = "strategy_adapted",
    ) -> None:
        name = f"RegimeTransition(window={adaptation_window})"
        super().__init__(name, adaptation_window)
        self.regime_change_key = regime_change_key
        self.adapted_key = adapted_key

    def trigger_predicate(self, state: Dict[str, float]) -> bool:
        return bool(state.get(self.regime_change_key, 0.0))

    def recovery_predicate(self, state: Dict[str, float]) -> bool:
        return bool(state.get(self.adapted_key, 0.0))

    def to_ltl_formula(self) -> str:
        return (
            f"G(({self.regime_change_key}) "
            f"-> F[0,{self.horizon}]({self.adapted_key}))"
        )

    def get_safe_state_mask(self, n_states: int) -> np.ndarray:
        return np.ones(n_states, dtype=bool)


# -----------------------------------------------------------------------
# Bounded liveness library (factory)
# -----------------------------------------------------------------------

class BoundedLivenessLibrary:
    """Factory for standard bounded-liveness specification suites.

    Provides pre-configured suites at different risk tolerance levels
    and supports custom configuration via ``from_config``.
    """

    @staticmethod
    def conservative_suite() -> List[BoundedLivenessSpec]:
        """Tight bounds — strict recovery requirements."""
        return [
            DrawdownRecoverySpec(
                threshold=0.03, recovery_level=0.01, horizon=10,
            ),
            LossRecoverySpec(
                threshold=0.02, recovery_level=0.005, horizon=3,
            ),
            PositionReductionSpec(
                limit=80.0, safe_level=50.0, horizon=5,
            ),
            RegimeTransitionSpec(adaptation_window=5),
        ]

    @staticmethod
    def moderate_suite() -> List[BoundedLivenessSpec]:
        """Balanced bounds — reasonable recovery requirements."""
        return [
            DrawdownRecoverySpec(
                threshold=0.05, recovery_level=0.02, horizon=20,
            ),
            LossRecoverySpec(
                threshold=0.03, recovery_level=0.01, horizon=5,
            ),
            PositionReductionSpec(
                limit=100.0, safe_level=80.0, horizon=10,
            ),
            RegimeTransitionSpec(adaptation_window=10),
        ]

    @staticmethod
    def aggressive_suite() -> List[BoundedLivenessSpec]:
        """Loose bounds — relaxed recovery requirements."""
        return [
            DrawdownRecoverySpec(
                threshold=0.10, recovery_level=0.05, horizon=50,
            ),
            LossRecoverySpec(
                threshold=0.05, recovery_level=0.02, horizon=10,
            ),
            PositionReductionSpec(
                limit=150.0, safe_level=120.0, horizon=20,
            ),
            RegimeTransitionSpec(adaptation_window=20),
        ]

    @staticmethod
    def from_config(config: Dict[str, Any]) -> List[BoundedLivenessSpec]:
        """Create a specification suite from a configuration dict.

        Expected structure::

            {
                "drawdown_recovery": {
                    "threshold": 0.05, "recovery_level": 0.02, "horizon": 20
                },
                "loss_recovery": {
                    "threshold": 0.03, "recovery_level": 0.01, "horizon": 5
                },
                "position_reduction": {
                    "limit": 100, "safe_level": 80, "horizon": 10
                },
                "regime_transition": {
                    "adaptation_window": 10
                }
            }

        Any key may be omitted to skip that spec.
        """
        specs: List[BoundedLivenessSpec] = []

        if "drawdown_recovery" in config:
            cfg = config["drawdown_recovery"]
            specs.append(DrawdownRecoverySpec(**cfg))

        if "loss_recovery" in config:
            cfg = config["loss_recovery"]
            specs.append(LossRecoverySpec(**cfg))

        if "position_reduction" in config:
            cfg = config["position_reduction"]
            specs.append(PositionReductionSpec(**cfg))

        if "regime_transition" in config:
            cfg = config["regime_transition"]
            specs.append(RegimeTransitionSpec(**cfg))

        return specs

    @staticmethod
    def all_spec_names() -> List[str]:
        """Return canonical names for the four liveness specs."""
        return [
            "drawdown_recovery",
            "loss_recovery",
            "position_reduction",
            "regime_transition",
        ]
