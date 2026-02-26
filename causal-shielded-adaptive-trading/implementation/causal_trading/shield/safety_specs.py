"""
Temporal logic safety specifications for trading.

Defines safety specifications as bounded linear temporal logic (LTL) formulas
with concrete checking procedures against trajectories. Each spec encodes
a trading-domain constraint (drawdown, position limits, margin, etc.) and
can be composed via conjunction/disjunction.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import lfilter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LTL formula representation
# ---------------------------------------------------------------------------

class LTLOp(Enum):
    """Bounded LTL operators."""
    ALWAYS = "G"        # Globally (within horizon)
    EVENTUALLY = "F"    # Eventually (within horizon)
    UNTIL = "U"         # Until
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    ATOM = "ATOM"


@dataclass
class LTLFormula:
    """
    Bounded LTL formula representation.

    Supports atomic propositions, boolean connectives, and bounded
    temporal operators G[0,H], F[0,H], U[0,H].

    Parameters
    ----------
    op : LTLOp
        The operator.
    children : list
        Sub-formulas (for connectives/temporal ops).
    atom : str, optional
        Atomic proposition name (for ATOM op).
    bound : int, optional
        Time bound for temporal operators.
    predicate : callable, optional
        Predicate function for atomic propositions. Takes a state dict
        and returns bool.
    """
    op: LTLOp
    children: List["LTLFormula"] = field(default_factory=list)
    atom: Optional[str] = None
    bound: Optional[int] = None
    predicate: Optional[Callable[[Dict[str, float]], bool]] = None

    def __repr__(self) -> str:
        if self.op == LTLOp.ATOM:
            return f"Atom({self.atom})"
        elif self.op == LTLOp.NOT:
            return f"NOT({self.children[0]})"
        elif self.op == LTLOp.AND:
            return f"({self.children[0]} AND {self.children[1]})"
        elif self.op == LTLOp.OR:
            return f"({self.children[0]} OR {self.children[1]})"
        elif self.op == LTLOp.ALWAYS:
            b = f"[0,{self.bound}]" if self.bound else ""
            return f"G{b}({self.children[0]})"
        elif self.op == LTLOp.EVENTUALLY:
            b = f"[0,{self.bound}]" if self.bound else ""
            return f"F{b}({self.children[0]})"
        elif self.op == LTLOp.UNTIL:
            b = f"[0,{self.bound}]" if self.bound else ""
            return f"({self.children[0]} U{b} {self.children[1]})"
        elif self.op == LTLOp.IMPLIES:
            return f"({self.children[0]} -> {self.children[1]})"
        return f"LTL({self.op})"

    @staticmethod
    def atom(name: str, predicate: Callable[[Dict[str, float]], bool]) -> "LTLFormula":
        """Create an atomic proposition."""
        return LTLFormula(op=LTLOp.ATOM, atom=name, predicate=predicate)

    @staticmethod
    def always(child: "LTLFormula", bound: Optional[int] = None) -> "LTLFormula":
        """Create G[0,bound](child)."""
        return LTLFormula(op=LTLOp.ALWAYS, children=[child], bound=bound)

    @staticmethod
    def eventually(child: "LTLFormula", bound: Optional[int] = None) -> "LTLFormula":
        """Create F[0,bound](child)."""
        return LTLFormula(op=LTLOp.EVENTUALLY, children=[child], bound=bound)

    @staticmethod
    def until(left: "LTLFormula", right: "LTLFormula", bound: Optional[int] = None) -> "LTLFormula":
        """Create left U[0,bound] right."""
        return LTLFormula(op=LTLOp.UNTIL, children=[left, right], bound=bound)

    @staticmethod
    def and_(left: "LTLFormula", right: "LTLFormula") -> "LTLFormula":
        """Conjunction."""
        return LTLFormula(op=LTLOp.AND, children=[left, right])

    @staticmethod
    def or_(left: "LTLFormula", right: "LTLFormula") -> "LTLFormula":
        """Disjunction."""
        return LTLFormula(op=LTLOp.OR, children=[left, right])

    @staticmethod
    def not_(child: "LTLFormula") -> "LTLFormula":
        """Negation."""
        return LTLFormula(op=LTLOp.NOT, children=[child])

    @staticmethod
    def implies(left: "LTLFormula", right: "LTLFormula") -> "LTLFormula":
        """Implication."""
        return LTLFormula(op=LTLOp.IMPLIES, children=[left, right])


class TrajectoryChecker:
    """
    Check bounded LTL formulas against trajectories.

    A trajectory is a sequence of state dictionaries, where each state
    maps variable names to float values.

    Parameters
    ----------
    trajectory : list of dict
        Sequence of state observations.
    """

    def __init__(self, trajectory: List[Dict[str, float]]) -> None:
        self.trajectory = trajectory
        self.length = len(trajectory)

    def check(self, formula: LTLFormula, time: int = 0) -> bool:
        """
        Check if formula holds at the given time index.

        Parameters
        ----------
        formula : LTLFormula
            The formula to check.
        time : int
            Starting time index.

        Returns
        -------
        bool
            Whether the formula is satisfied.
        """
        if time >= self.length:
            return True  # vacuously true beyond trajectory

        op = formula.op

        if op == LTLOp.ATOM:
            if formula.predicate is None:
                return True
            return formula.predicate(self.trajectory[time])

        elif op == LTLOp.NOT:
            return not self.check(formula.children[0], time)

        elif op == LTLOp.AND:
            return (self.check(formula.children[0], time) and
                    self.check(formula.children[1], time))

        elif op == LTLOp.OR:
            return (self.check(formula.children[0], time) or
                    self.check(formula.children[1], time))

        elif op == LTLOp.IMPLIES:
            return (not self.check(formula.children[0], time) or
                    self.check(formula.children[1], time))

        elif op == LTLOp.ALWAYS:
            bound = formula.bound if formula.bound is not None else self.length - time
            end = min(time + bound, self.length)
            for t in range(time, end):
                if not self.check(formula.children[0], t):
                    return False
            return True

        elif op == LTLOp.EVENTUALLY:
            bound = formula.bound if formula.bound is not None else self.length - time
            end = min(time + bound, self.length)
            for t in range(time, end):
                if self.check(formula.children[0], t):
                    return True
            return False

        elif op == LTLOp.UNTIL:
            bound = formula.bound if formula.bound is not None else self.length - time
            end = min(time + bound, self.length)
            for t in range(time, end):
                if self.check(formula.children[1], t):
                    return True
                if not self.check(formula.children[0], t):
                    return False
            return False

        raise ValueError(f"Unknown LTL operator: {op}")

    def check_all_times(self, formula: LTLFormula) -> np.ndarray:
        """
        Check formula at every time step.

        Returns
        -------
        np.ndarray
            Boolean array of length len(trajectory).
        """
        results = np.zeros(self.length, dtype=bool)
        for t in range(self.length):
            results[t] = self.check(formula, t)
        return results

    def robustness(self, formula: LTLFormula, time: int = 0) -> float:
        """
        Compute quantitative robustness (Signal Temporal Logic style).

        Positive value means formula satisfied, negative means violated.
        Magnitude indicates margin of satisfaction/violation.

        Parameters
        ----------
        formula : LTLFormula
            The formula to evaluate.
        time : int
            Starting time.

        Returns
        -------
        float
            Robustness value.
        """
        if time >= self.length:
            return float('inf')

        op = formula.op

        if op == LTLOp.ATOM:
            if formula.predicate is None:
                return float('inf')
            return 1.0 if formula.predicate(self.trajectory[time]) else -1.0

        elif op == LTLOp.NOT:
            return -self.robustness(formula.children[0], time)

        elif op == LTLOp.AND:
            return min(
                self.robustness(formula.children[0], time),
                self.robustness(formula.children[1], time),
            )

        elif op == LTLOp.OR:
            return max(
                self.robustness(formula.children[0], time),
                self.robustness(formula.children[1], time),
            )

        elif op == LTLOp.IMPLIES:
            return max(
                -self.robustness(formula.children[0], time),
                self.robustness(formula.children[1], time),
            )

        elif op == LTLOp.ALWAYS:
            bound = formula.bound if formula.bound is not None else self.length - time
            end = min(time + bound, self.length)
            vals = [self.robustness(formula.children[0], t) for t in range(time, end)]
            return min(vals) if vals else float('inf')

        elif op == LTLOp.EVENTUALLY:
            bound = formula.bound if formula.bound is not None else self.length - time
            end = min(time + bound, self.length)
            vals = [self.robustness(formula.children[0], t) for t in range(time, end)]
            return max(vals) if vals else float('-inf')

        elif op == LTLOp.UNTIL:
            bound = formula.bound if formula.bound is not None else self.length - time
            end = min(time + bound, self.length)
            best = float('-inf')
            for t2 in range(time, end):
                rhs = self.robustness(formula.children[1], t2)
                lhs_vals = [
                    self.robustness(formula.children[0], t)
                    for t in range(time, t2)
                ]
                lhs_min = min(lhs_vals) if lhs_vals else float('inf')
                best = max(best, min(lhs_min, rhs))
            return best

        raise ValueError(f"Unknown op: {op}")


# ---------------------------------------------------------------------------
# Base safety specification
# ---------------------------------------------------------------------------

class SafetySpecification(ABC):
    """
    Abstract base class for safety specifications.

    Each specification defines:
    1. A check method that evaluates a trajectory against the spec.
    2. An LTL formula representation.
    3. Constraint functions for state-level checking.
    4. A safe state mask for discrete MDP shielding.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def check(self, trajectory: List[Dict[str, float]]) -> bool:
        """Check if a trajectory satisfies the specification."""
        ...

    @abstractmethod
    def to_ltl(self) -> LTLFormula:
        """Convert to bounded LTL formula."""
        ...

    @abstractmethod
    def get_constraints(self) -> List[Callable]:
        """Get list of constraint functions for state-level checking."""
        ...

    def get_safe_state_mask(self, n_states: int) -> np.ndarray:
        """
        Get boolean mask of safe states for MDP shielding.

        Default: delegates to get_constraints() applied to integer state indices.
        Subclasses should override for domain-specific state representations.
        """
        constraints = self.get_constraints()
        mask = np.ones(n_states, dtype=bool)
        for constraint_fn in constraints:
            for s in range(n_states):
                try:
                    if not constraint_fn(s):
                        mask[s] = False
                except (TypeError, ValueError):
                    pass
        return mask

    def check_quantitative(self, trajectory: List[Dict[str, float]]) -> float:
        """
        Quantitative satisfaction: how well is the spec satisfied?

        Returns a value in [-inf, inf]. Positive means satisfied.
        Subclasses should override for meaningful quantitative semantics.
        """
        checker = TrajectoryChecker(trajectory)
        return checker.robustness(self.to_ltl())


# ---------------------------------------------------------------------------
# Concrete specifications
# ---------------------------------------------------------------------------

class BoundedDrawdownSpec(SafetySpecification):
    """
    Bounded drawdown specification: drawdown <= D at all times within horizon H.

    Drawdown is defined as the peak-to-trough decline in portfolio value,
    measured as a fraction of the peak value:
        drawdown(t) = (peak(t) - value(t)) / peak(t)

    Parameters
    ----------
    max_drawdown : float
        Maximum allowed drawdown fraction (e.g. 0.10 = 10%).
    horizon : int
        Time horizon for the specification.
    value_key : str
        Key in state dict for portfolio value.
    """

    def __init__(
        self,
        max_drawdown: float = 0.10,
        horizon: int = 100,
        value_key: str = "portfolio_value",
    ) -> None:
        super().__init__(f"BoundedDrawdown(D={max_drawdown}, H={horizon})")
        self.max_drawdown = max_drawdown
        self.horizon = horizon
        self.value_key = value_key

    def check(self, trajectory: List[Dict[str, float]]) -> bool:
        """Check if drawdown stays within bound over the trajectory."""
        if not trajectory:
            return True

        values = np.array([s.get(self.value_key, 0.0) for s in trajectory])
        n = min(len(values), self.horizon)
        values = values[:n]

        if len(values) == 0:
            return True

        peak = values[0]
        for v in values:
            peak = max(peak, v)
            if peak > 0:
                dd = (peak - v) / peak
                if dd > self.max_drawdown:
                    return False
        return True

    def compute_drawdown_series(self, trajectory: List[Dict[str, float]]) -> np.ndarray:
        """Compute the drawdown at each time step."""
        values = np.array([s.get(self.value_key, 0.0) for s in trajectory])
        if len(values) == 0:
            return np.array([])

        peak = np.maximum.accumulate(values)
        # Avoid division by zero
        safe_peak = np.where(peak > 0, peak, 1.0)
        drawdown = (peak - values) / safe_peak
        return drawdown

    def to_ltl(self) -> LTLFormula:
        """Convert to G[0,H](drawdown <= D)."""
        D = self.max_drawdown
        key = self.value_key

        def predicate(state: Dict[str, float]) -> bool:
            dd = state.get("drawdown", 0.0)
            return dd <= D

        prop = LTLFormula(op=LTLOp.ATOM, atom=f"dd<={D}", predicate=predicate)
        return LTLFormula.always(prop, bound=self.horizon)

    def get_constraints(self) -> List[Callable]:
        """Get drawdown constraint as a function of state index."""
        D = self.max_drawdown

        def dd_constraint(state_index: int) -> bool:
            # In discretized state space, upper portion of states are safe
            # Assumes state_index encodes drawdown level
            return True  # Needs domain-specific state mapping

        return [dd_constraint]

    def get_safe_state_mask(self, n_states: int) -> np.ndarray:
        """
        Generate safe state mask assuming states encode drawdown levels.

        States are assumed to represent drawdown buckets:
        state i corresponds to drawdown ~ i / n_states.
        """
        mask = np.ones(n_states, dtype=bool)
        for s in range(n_states):
            dd_level = s / n_states
            if dd_level > self.max_drawdown:
                mask[s] = False
        return mask


class PositionLimitSpec(SafetySpecification):
    """
    Position limit specification: |position| <= L at all times.

    Ensures the absolute value of the trading position never exceeds
    a specified limit.

    Parameters
    ----------
    max_position : float
        Maximum absolute position size.
    position_key : str
        Key in state dict for position.
    """

    def __init__(
        self,
        max_position: float = 100.0,
        position_key: str = "position",
    ) -> None:
        super().__init__(f"PositionLimit(L={max_position})")
        self.max_position = max_position
        self.position_key = position_key

    def check(self, trajectory: List[Dict[str, float]]) -> bool:
        """Check if position stays within limits."""
        for state in trajectory:
            pos = state.get(self.position_key, 0.0)
            if abs(pos) > self.max_position:
                return False
        return True

    def max_violation(self, trajectory: List[Dict[str, float]]) -> float:
        """Compute maximum position violation over trajectory."""
        if not trajectory:
            return 0.0
        positions = np.array([s.get(self.position_key, 0.0) for s in trajectory])
        violations = np.maximum(0, np.abs(positions) - self.max_position)
        return float(np.max(violations))

    def to_ltl(self) -> LTLFormula:
        """Convert to G(|pos| <= L)."""
        L = self.max_position
        key = self.position_key

        def predicate(state: Dict[str, float]) -> bool:
            return abs(state.get(key, 0.0)) <= L

        prop = LTLFormula(op=LTLOp.ATOM, atom=f"|pos|<={L}", predicate=predicate)
        return LTLFormula.always(prop)

    def get_constraints(self) -> List[Callable]:
        L = self.max_position

        def pos_constraint(state_index: int) -> bool:
            return True

        return [pos_constraint]

    def get_safe_state_mask(self, n_states: int) -> np.ndarray:
        """States in the central portion represent safe positions."""
        mask = np.ones(n_states, dtype=bool)
        # Map state index to position: state n_states//2 is position 0
        center = n_states // 2
        max_safe_offset = int(self.max_position * n_states / (2 * self.max_position + 1))
        for s in range(n_states):
            offset = abs(s - center)
            if offset > max_safe_offset:
                mask[s] = False
        return mask


class MarginSpec(SafetySpecification):
    """
    Margin specification: margin >= M at all times.

    Ensures the margin (available collateral as fraction of position value)
    stays above a minimum threshold.

    Parameters
    ----------
    min_margin : float
        Minimum required margin ratio (e.g. 0.25 = 25%).
    margin_key : str
        Key in state dict for margin ratio.
    """

    def __init__(
        self,
        min_margin: float = 0.25,
        margin_key: str = "margin_ratio",
    ) -> None:
        super().__init__(f"Margin(M={min_margin})")
        self.min_margin = min_margin
        self.margin_key = margin_key

    def check(self, trajectory: List[Dict[str, float]]) -> bool:
        """Check margin requirement over trajectory."""
        for state in trajectory:
            margin = state.get(self.margin_key, 1.0)
            if margin < self.min_margin:
                return False
        return True

    def margin_headroom(self, trajectory: List[Dict[str, float]]) -> np.ndarray:
        """Compute margin headroom (margin - min_margin) at each step."""
        margins = np.array([s.get(self.margin_key, 1.0) for s in trajectory])
        return margins - self.min_margin

    def to_ltl(self) -> LTLFormula:
        """Convert to G(margin >= M)."""
        M = self.min_margin
        key = self.margin_key

        def predicate(state: Dict[str, float]) -> bool:
            return state.get(key, 1.0) >= M

        prop = LTLFormula(op=LTLOp.ATOM, atom=f"margin>={M}", predicate=predicate)
        return LTLFormula.always(prop)

    def get_constraints(self) -> List[Callable]:
        M = self.min_margin

        def margin_constraint(state_index: int) -> bool:
            return True

        return [margin_constraint]

    def get_safe_state_mask(self, n_states: int) -> np.ndarray:
        """Upper portion of state indices represent sufficient margin."""
        mask = np.ones(n_states, dtype=bool)
        threshold_idx = int(self.min_margin * n_states)
        mask[:threshold_idx] = False
        return mask


class MaxLossSpec(SafetySpecification):
    """
    Maximum cumulative loss specification within a rolling window.

    Ensures the cumulative loss over any window of W steps does not
    exceed a threshold C.

    Parameters
    ----------
    max_loss : float
        Maximum allowed cumulative loss in the window.
    window : int
        Rolling window size.
    pnl_key : str
        Key in state dict for per-step P&L.
    """

    def __init__(
        self,
        max_loss: float = 1000.0,
        window: int = 20,
        pnl_key: str = "pnl",
    ) -> None:
        super().__init__(f"MaxLoss(C={max_loss}, W={window})")
        self.max_loss = max_loss
        self.window = window
        self.pnl_key = pnl_key

    def check(self, trajectory: List[Dict[str, float]]) -> bool:
        """Check max cumulative loss in any window."""
        if len(trajectory) < 2:
            return True

        pnl = np.array([s.get(self.pnl_key, 0.0) for s in trajectory])
        losses = np.minimum(0, pnl)  # only negative values

        # Rolling sum of losses
        n = len(losses)
        for i in range(n - self.window + 1):
            window_loss = -np.sum(losses[i : i + self.window])
            if window_loss > self.max_loss:
                return False
        return True

    def compute_rolling_loss(self, trajectory: List[Dict[str, float]]) -> np.ndarray:
        """Compute rolling window cumulative loss."""
        pnl = np.array([s.get(self.pnl_key, 0.0) for s in trajectory])
        losses = -np.minimum(0, pnl)

        if len(losses) < self.window:
            return np.cumsum(losses)

        # Efficient rolling sum using cumsum
        cumsum = np.cumsum(losses)
        rolling = np.zeros(len(losses))
        rolling[:self.window] = cumsum[:self.window]
        rolling[self.window:] = cumsum[self.window:] - cumsum[:-self.window]
        return rolling

    def to_ltl(self) -> LTLFormula:
        """Convert to G(rolling_loss <= C)."""
        C = self.max_loss

        def predicate(state: Dict[str, float]) -> bool:
            return state.get("rolling_loss", 0.0) <= C

        prop = LTLFormula(
            op=LTLOp.ATOM,
            atom=f"rolling_loss<={C}",
            predicate=predicate,
        )
        return LTLFormula.always(prop)

    def get_constraints(self) -> List[Callable]:
        C = self.max_loss

        def loss_constraint(state_index: int) -> bool:
            return True

        return [loss_constraint]

    def get_safe_state_mask(self, n_states: int) -> np.ndarray:
        """Lower state indices are safe (represent lower loss levels)."""
        mask = np.ones(n_states, dtype=bool)
        threshold = int(n_states * 0.8)
        mask[threshold:] = False
        return mask


class TurnoverSpec(SafetySpecification):
    """
    Turnover specification: turnover <= T per period.

    Limits the rate at which positions are changed to control
    transaction costs and market impact.

    Parameters
    ----------
    max_turnover : float
        Maximum turnover per period (sum of absolute position changes).
    period : int
        Period length for measuring turnover.
    position_key : str
        Key in state dict for position.
    """

    def __init__(
        self,
        max_turnover: float = 50.0,
        period: int = 10,
        position_key: str = "position",
    ) -> None:
        super().__init__(f"Turnover(T={max_turnover}, P={period})")
        self.max_turnover = max_turnover
        self.period = period
        self.position_key = position_key

    def check(self, trajectory: List[Dict[str, float]]) -> bool:
        """Check if turnover stays within limits."""
        if len(trajectory) < 2:
            return True

        positions = np.array([s.get(self.position_key, 0.0) for s in trajectory])
        changes = np.abs(np.diff(positions))

        n = len(changes)
        for i in range(0, n - self.period + 1, self.period):
            period_turnover = np.sum(changes[i : i + self.period])
            if period_turnover > self.max_turnover:
                return False
        return True

    def compute_turnover_series(
        self, trajectory: List[Dict[str, float]]
    ) -> np.ndarray:
        """Compute turnover at each step using exponential moving average."""
        if len(trajectory) < 2:
            return np.array([0.0])

        positions = np.array([s.get(self.position_key, 0.0) for s in trajectory])
        changes = np.abs(np.diff(positions))

        # EMA of turnover
        alpha = 2.0 / (self.period + 1)
        ema = np.zeros(len(changes))
        ema[0] = changes[0]
        for i in range(1, len(changes)):
            ema[i] = alpha * changes[i] + (1 - alpha) * ema[i - 1]

        return ema * self.period  # Scale to period

    def to_ltl(self) -> LTLFormula:
        """Convert to G(turnover <= T)."""
        T = self.max_turnover

        def predicate(state: Dict[str, float]) -> bool:
            return state.get("turnover", 0.0) <= T

        prop = LTLFormula(
            op=LTLOp.ATOM,
            atom=f"turnover<={T}",
            predicate=predicate,
        )
        return LTLFormula.always(prop)

    def get_constraints(self) -> List[Callable]:
        return [lambda s: True]

    def get_safe_state_mask(self, n_states: int) -> np.ndarray:
        """All states are potentially safe; turnover is a trajectory property."""
        return np.ones(n_states, dtype=bool)


class CompositeSpec(SafetySpecification):
    """
    Composition of multiple safety specifications.

    Supports conjunction (all must hold) and disjunction (at least one holds).

    Parameters
    ----------
    specs : list of SafetySpecification
        Component specifications.
    mode : str
        'conjunction' (AND) or 'disjunction' (OR).
    """

    def __init__(
        self,
        specs: List[SafetySpecification],
        mode: str = "conjunction",
    ) -> None:
        names = [s.name for s in specs]
        op = " AND " if mode == "conjunction" else " OR "
        super().__init__(f"Composite({op.join(names)})")
        self.specs = specs
        self.mode = mode

    def check(self, trajectory: List[Dict[str, float]]) -> bool:
        """Check composite specification."""
        results = [spec.check(trajectory) for spec in self.specs]
        if self.mode == "conjunction":
            return all(results)
        else:
            return any(results)

    def check_detailed(
        self, trajectory: List[Dict[str, float]]
    ) -> Dict[str, bool]:
        """Check each spec individually and return per-spec results."""
        return {spec.name: spec.check(trajectory) for spec in self.specs}

    def find_violating_specs(
        self, trajectory: List[Dict[str, float]]
    ) -> List[str]:
        """Return names of specs that are violated."""
        return [
            spec.name for spec in self.specs
            if not spec.check(trajectory)
        ]

    def to_ltl(self) -> LTLFormula:
        """Convert to LTL conjunction/disjunction."""
        if not self.specs:
            return LTLFormula(op=LTLOp.ATOM, atom="true", predicate=lambda s: True)

        formulas = [spec.to_ltl() for spec in self.specs]
        result = formulas[0]
        for f in formulas[1:]:
            if self.mode == "conjunction":
                result = LTLFormula.and_(result, f)
            else:
                result = LTLFormula.or_(result, f)
        return result

    def get_constraints(self) -> List[Callable]:
        """Get all constraints from component specs."""
        constraints = []
        for spec in self.specs:
            constraints.extend(spec.get_constraints())
        return constraints

    def get_safe_state_mask(self, n_states: int) -> np.ndarray:
        """Combine safe state masks."""
        if not self.specs:
            return np.ones(n_states, dtype=bool)

        masks = [spec.get_safe_state_mask(n_states) for spec in self.specs]

        if self.mode == "conjunction":
            result = masks[0].copy()
            for m in masks[1:]:
                result &= m
        else:
            result = masks[0].copy()
            for m in masks[1:]:
                result |= m

        return result

    def most_restrictive(self, n_states: int) -> Tuple[str, float]:
        """
        Find the most restrictive component spec.

        Returns the spec name and its safe state fraction.
        """
        min_frac = 1.0
        min_name = ""
        for spec in self.specs:
            mask = spec.get_safe_state_mask(n_states)
            frac = np.mean(mask)
            if frac < min_frac:
                min_frac = frac
                min_name = spec.name
        return min_name, float(min_frac)
