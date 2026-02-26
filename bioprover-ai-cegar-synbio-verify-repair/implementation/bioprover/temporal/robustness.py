"""STL Quantitative Robustness Semantics.

Implements efficient bottom-up evaluation of STL robustness using dynamic
programming and O(n) sliding window min/max for temporal operators.
Supports robustness gradient computation and sensitivity analysis.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from bioprover.temporal.stl_ast import (
    Always,
    ComparisonOp,
    Eventually,
    Expression,
    Interval,
    Predicate,
    STLAnd,
    STLFormula,
    STLImplies,
    STLNot,
    STLOr,
    Until,
)


# ---------------------------------------------------------------------------
# Signal representation
# ---------------------------------------------------------------------------

class Signal:
    """A time series signal with linear interpolation.

    Attributes:
        times: Sorted array of time points.
        values: Corresponding signal values.
        name: Optional signal name.
    """

    def __init__(
        self,
        times: NDArray[np.float64],
        values: NDArray[np.float64],
        name: str = "",
    ) -> None:
        if len(times) != len(values):
            raise ValueError("times and values must have the same length")
        if len(times) == 0:
            raise ValueError("Signal must have at least one sample")
        order = np.argsort(times)
        self.times = np.asarray(times[order], dtype=np.float64)
        self.values = np.asarray(values[order], dtype=np.float64)
        self.name = name

    def at(self, t: float) -> float:
        """Evaluate signal at time t via linear interpolation."""
        return float(np.interp(t, self.times, self.values))

    def sample_at(self, query_times: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate signal at multiple time points."""
        return np.interp(query_times, self.times, self.values)

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])

    @property
    def t_start(self) -> float:
        return float(self.times[0])

    @property
    def t_end(self) -> float:
        return float(self.times[-1])

    def __len__(self) -> int:
        return len(self.times)

    @classmethod
    def from_function(
        cls,
        fn: Callable[[float], float],
        t_start: float,
        t_end: float,
        dt: float,
        name: str = "",
    ) -> Signal:
        """Create a signal by sampling a function at regular intervals."""
        times = np.arange(t_start, t_end + dt * 0.5, dt)
        values = np.array([fn(t) for t in times])
        return cls(times, values, name)


# ---------------------------------------------------------------------------
# Sliding window min/max (O(n) monotone deque)
# ---------------------------------------------------------------------------

def _sliding_window_min(
    values: NDArray[np.float64],
    times: NDArray[np.float64],
    window_lo: float,
    window_hi: float,
) -> NDArray[np.float64]:
    """Compute sliding window minimum: out[i] = min values[j] for
    times[j] in [times[i]+window_lo, times[i]+window_hi]."""
    n = len(values)
    result = np.full(n, np.inf)
    dq: Deque[int] = deque()
    j_start = 0
    j_end = 0

    for i in range(n):
        t_lo = times[i] + window_lo
        t_hi = times[i] + window_hi

        # Advance j_end to include all elements in [t_lo, t_hi]
        while j_end < n and times[j_end] <= t_hi:
            while dq and values[dq[-1]] >= values[j_end]:
                dq.pop()
            dq.append(j_end)
            j_end += 1

        # Remove elements that have fallen out of the window (times < t_lo)
        while dq and times[dq[0]] < t_lo:
            dq.popleft()

        if dq:
            result[i] = values[dq[0]]
        else:
            # No valid window elements; signal too short
            result[i] = np.inf

    return result


def _sliding_window_max(
    values: NDArray[np.float64],
    times: NDArray[np.float64],
    window_lo: float,
    window_hi: float,
) -> NDArray[np.float64]:
    """Compute sliding window maximum using negation trick."""
    return -_sliding_window_min(-values, times, window_lo, window_hi)


# ---------------------------------------------------------------------------
# Robustness computer
# ---------------------------------------------------------------------------

@dataclass
class RobustnessTrace:
    """Full robustness trace: robustness value at each time point."""
    times: NDArray[np.float64]
    values: NDArray[np.float64]

    @property
    def robustness_at_zero(self) -> float:
        """Robustness at time 0 (or first time point)."""
        return float(self.values[0])

    @property
    def min_robustness(self) -> float:
        return float(np.min(self.values))

    @property
    def satisfied(self) -> bool:
        return self.robustness_at_zero > 0

    @property
    def violated(self) -> bool:
        return self.robustness_at_zero < 0


class RobustnessComputer:
    """Efficient bottom-up STL robustness computation.

    Uses dynamic programming: computes robustness for each sub-formula
    at all time points, then combines using sliding window operations
    for temporal operators.
    """

    def __init__(self, signals: Dict[str, Signal]) -> None:
        self._signals = signals
        # Build a common time grid from all signals
        all_times: Set[float] = set()
        for sig in signals.values():
            all_times.update(sig.times.tolist())
        self._times = np.array(sorted(all_times), dtype=np.float64)
        # Pre-sample signals on common grid
        self._sampled: Dict[str, NDArray[np.float64]] = {}
        for name, sig in signals.items():
            self._sampled[name] = sig.sample_at(self._times)

    @property
    def times(self) -> NDArray[np.float64]:
        return self._times

    def compute(self, formula: STLFormula) -> RobustnessTrace:
        """Compute the full robustness trace for a formula."""
        values = self._eval(formula)
        return RobustnessTrace(self._times.copy(), values)

    def _eval(self, formula: STLFormula) -> NDArray[np.float64]:
        """Recursively evaluate robustness at all time points."""
        if isinstance(formula, Predicate):
            return self._eval_predicate(formula)
        if isinstance(formula, STLNot):
            return -self._eval(formula.child)
        if isinstance(formula, STLAnd):
            return np.minimum(self._eval(formula.left), self._eval(formula.right))
        if isinstance(formula, STLOr):
            return np.maximum(self._eval(formula.left), self._eval(formula.right))
        if isinstance(formula, STLImplies):
            return np.maximum(-self._eval(formula.antecedent),
                              self._eval(formula.consequent))
        if isinstance(formula, Always):
            child_vals = self._eval(formula.child)
            return _sliding_window_min(
                child_vals, self._times,
                formula.interval.lo, formula.interval.hi,
            )
        if isinstance(formula, Eventually):
            child_vals = self._eval(formula.child)
            return _sliding_window_max(
                child_vals, self._times,
                formula.interval.lo, formula.interval.hi,
            )
        if isinstance(formula, Until):
            return self._eval_until(formula)
        raise TypeError(f"Unknown formula type: {type(formula)}")

    def _eval_predicate(self, pred: Predicate) -> NDArray[np.float64]:
        """Evaluate a predicate at all time points."""
        expr = pred.expr
        if expr.variable is not None:
            if expr.variable not in self._sampled:
                raise KeyError(f"Signal '{expr.variable}' not found")
            sig_vals = self._sampled[expr.variable]
            vals = expr.scale * sig_vals + expr.offset
        elif expr.constant is not None:
            vals = np.full(len(self._times), expr.constant)
        else:
            raise ValueError("Expression has neither variable nor constant")

        if pred.op in (ComparisonOp.GT, ComparisonOp.GE):
            return vals - pred.threshold
        else:
            return pred.threshold - vals

    def _eval_until(self, formula: Until) -> NDArray[np.float64]:
        """Evaluate Until robustness: rho(phi1 U[a,b] phi2, t).

        For each time t, find the best t' in [t+a, t+b] such that phi2 holds at t'
        and phi1 holds on [t+a, t'). Uses O(n^2) baseline; an O(n log n) implementation
        would use segment trees.
        """
        left_vals = self._eval(formula.left)
        right_vals = self._eval(formula.right)
        n = len(self._times)
        result = np.full(n, -np.inf)

        lo_off = formula.interval.lo
        hi_off = formula.interval.hi

        for i in range(n):
            t = self._times[i]
            t_lo = t + lo_off
            t_hi = t + hi_off

            # Find indices in [t_lo, t_hi]
            j_start = np.searchsorted(self._times, t_lo, side="left")
            j_end = np.searchsorted(self._times, t_hi, side="right")

            best = -np.inf
            running_min_left = np.inf  # min of left_vals from j_start to current

            for j in range(j_start, j_end):
                rho_right = right_vals[j]
                # min of left over [j_start, j)
                candidate = min(rho_right, running_min_left)
                best = max(best, candidate)
                running_min_left = min(running_min_left, left_vals[j])

            result[i] = best

        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_robustness(
    formula: STLFormula,
    signals: Dict[str, Signal],
    t: float = 0.0,
) -> float:
    """Compute robustness of formula at time t given signals."""
    computer = RobustnessComputer(signals)
    trace = computer.compute(formula)
    return float(np.interp(t, trace.times, trace.values))


# ---------------------------------------------------------------------------
# Robustness gradient (finite differences)
# ---------------------------------------------------------------------------

def robustness_gradient(
    formula: STLFormula,
    signals: Dict[str, Signal],
    t: float = 0.0,
    epsilon: float = 1e-6,
) -> Dict[str, NDArray[np.float64]]:
    """Compute gradient of robustness w.r.t. each signal value via finite differences.

    Returns a dict mapping signal name to array of partial derivatives
    (one per time sample of that signal).
    """
    base_rho = compute_robustness(formula, signals, t)
    gradients: Dict[str, NDArray[np.float64]] = {}

    for name, sig in signals.items():
        grad = np.zeros(len(sig.values))
        for i in range(len(sig.values)):
            perturbed_values = sig.values.copy()
            perturbed_values[i] += epsilon
            perturbed_sig = Signal(sig.times.copy(), perturbed_values, sig.name)
            perturbed_signals = dict(signals)
            perturbed_signals[name] = perturbed_sig
            rho_pert = compute_robustness(formula, perturbed_signals, t)
            grad[i] = (rho_pert - base_rho) / epsilon
        gradients[name] = grad

    return gradients


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Result of sensitivity analysis for an STL formula."""
    atom_robustness: List[Tuple[Predicate, float]]
    critical_atom: Optional[Predicate]
    critical_robustness: float
    margin: float

    @property
    def is_robust(self) -> bool:
        return self.margin > 0


def sensitivity_analysis(
    formula: STLFormula,
    signals: Dict[str, Signal],
    t: float = 0.0,
) -> SensitivityResult:
    """Identify which atomic proposition is closest to switching truth value.

    Returns the atom with smallest absolute robustness (closest to violation/satisfaction
    boundary).
    """
    atoms = formula.atoms()
    atom_rho: List[Tuple[Predicate, float]] = []

    for atom in atoms:
        rho = compute_robustness(atom, signals, t)
        atom_rho.append((atom, rho))

    if not atom_rho:
        return SensitivityResult([], None, float("inf"), float("inf"))

    # Find critical atom: smallest |robustness|
    atom_rho.sort(key=lambda x: abs(x[1]))
    critical_atom, critical_rho = atom_rho[0]

    overall_rho = compute_robustness(formula, signals, t)

    return SensitivityResult(
        atom_robustness=atom_rho,
        critical_atom=critical_atom,
        critical_robustness=critical_rho,
        margin=abs(overall_rho),
    )


# ---------------------------------------------------------------------------
# Multi-signal (ensemble) robustness
# ---------------------------------------------------------------------------

@dataclass
class EnsembleRobustness:
    """Robustness statistics over an ensemble of trajectories."""
    individual: List[float]
    mean: float
    std: float
    min_val: float
    max_val: float
    satisfaction_ratio: float

    @property
    def all_satisfied(self) -> bool:
        return self.min_val > 0

    @property
    def all_violated(self) -> bool:
        return self.max_val < 0


def ensemble_robustness(
    formula: STLFormula,
    signal_sets: List[Dict[str, Signal]],
    t: float = 0.0,
) -> EnsembleRobustness:
    """Compute robustness over an ensemble of signal trajectories.

    Args:
        formula: STL formula to evaluate.
        signal_sets: List of signal dictionaries (one per trajectory).
        t: Time point for evaluation.

    Returns:
        Ensemble statistics including mean, std, satisfaction ratio.
    """
    rhos = [compute_robustness(formula, sigs, t) for sigs in signal_sets]
    arr = np.array(rhos)
    n_sat = int(np.sum(arr > 0))
    return EnsembleRobustness(
        individual=rhos,
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min_val=float(np.min(arr)),
        max_val=float(np.max(arr)),
        satisfaction_ratio=n_sat / len(rhos) if rhos else 0.0,
    )


# ---------------------------------------------------------------------------
# Robust satisfaction classification
# ---------------------------------------------------------------------------

class SatisfactionClass(Enum):
    """Classification of formula satisfaction based on robustness."""
    ROBUST_SAT = auto()
    MARGINAL_SAT = auto()
    MARGINAL_VIOL = auto()
    ROBUST_VIOL = auto()


def classify_satisfaction(
    formula: STLFormula,
    signals: Dict[str, Signal],
    t: float = 0.0,
    margin_threshold: float = 0.1,
) -> SatisfactionClass:
    """Classify satisfaction based on robustness magnitude."""
    rho = compute_robustness(formula, signals, t)
    if rho > margin_threshold:
        return SatisfactionClass.ROBUST_SAT
    elif rho > 0:
        return SatisfactionClass.MARGINAL_SAT
    elif rho > -margin_threshold:
        return SatisfactionClass.MARGINAL_VIOL
    else:
        return SatisfactionClass.ROBUST_VIOL
