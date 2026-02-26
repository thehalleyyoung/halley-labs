"""Interval-based Model Checking with Three-Valued Semantics.

Evaluates STL formulas over interval-valued signals where each time point
has [lo, hi] bounds (e.g., from reachability analysis flowpipes).
Uses three-valued logic: TRUE, FALSE, UNKNOWN with conservative evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from bioprover.temporal.stl_ast import (
    Always,
    ComparisonOp,
    Eventually,
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
# Three-valued logic
# ---------------------------------------------------------------------------

class ThreeValued(Enum):
    """Three-valued truth: TRUE, FALSE, UNKNOWN."""
    TRUE = auto()
    FALSE = auto()
    UNKNOWN = auto()

    def __and__(self, other: ThreeValued) -> ThreeValued:
        if self == ThreeValued.FALSE or other == ThreeValued.FALSE:
            return ThreeValued.FALSE
        if self == ThreeValued.TRUE and other == ThreeValued.TRUE:
            return ThreeValued.TRUE
        return ThreeValued.UNKNOWN

    def __or__(self, other: ThreeValued) -> ThreeValued:
        if self == ThreeValued.TRUE or other == ThreeValued.TRUE:
            return ThreeValued.TRUE
        if self == ThreeValued.FALSE and other == ThreeValued.FALSE:
            return ThreeValued.FALSE
        return ThreeValued.UNKNOWN

    def __invert__(self) -> ThreeValued:
        if self == ThreeValued.TRUE:
            return ThreeValued.FALSE
        if self == ThreeValued.FALSE:
            return ThreeValued.TRUE
        return ThreeValued.UNKNOWN

    @property
    def is_definite(self) -> bool:
        return self != ThreeValued.UNKNOWN


# ---------------------------------------------------------------------------
# Three-valued result with evidence
# ---------------------------------------------------------------------------

@dataclass
class ThreeValuedResult:
    """Result of three-valued model checking with supporting evidence."""
    value: ThreeValued
    formula: STLFormula
    time: float
    evidence: str = ""
    unknown_regions: List[Tuple[float, float]] = field(default_factory=list)
    sub_results: List[ThreeValuedResult] = field(default_factory=list)

    @property
    def is_definite(self) -> bool:
        return self.value.is_definite

    @property
    def needs_refinement(self) -> bool:
        return self.value == ThreeValued.UNKNOWN

    def refinement_guidance(self) -> List[Tuple[float, float]]:
        """Return time intervals that caused UNKNOWN results and need refinement."""
        regions: List[Tuple[float, float]] = list(self.unknown_regions)
        for sub in self.sub_results:
            if sub.value == ThreeValued.UNKNOWN:
                regions.extend(sub.refinement_guidance())
        # Merge overlapping intervals
        return _merge_intervals(regions)


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping intervals."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_ivs[0]]
    for lo, hi in sorted_ivs[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


# ---------------------------------------------------------------------------
# Interval signal
# ---------------------------------------------------------------------------

class IntervalSignal:
    """A signal where each time point has an interval [lo, hi] of possible values.

    Represents uncertainty bounds from reachability analysis or measurement error.
    """

    def __init__(
        self,
        times: NDArray[np.float64],
        lo_values: NDArray[np.float64],
        hi_values: NDArray[np.float64],
        name: str = "",
    ) -> None:
        if not (len(times) == len(lo_values) == len(hi_values)):
            raise ValueError("times, lo_values, and hi_values must have equal length")
        order = np.argsort(times)
        self.times = np.asarray(times[order], dtype=np.float64)
        self.lo_values = np.asarray(lo_values[order], dtype=np.float64)
        self.hi_values = np.asarray(hi_values[order], dtype=np.float64)
        self.name = name

    def lo_at(self, t: float) -> float:
        return float(np.interp(t, self.times, self.lo_values))

    def hi_at(self, t: float) -> float:
        return float(np.interp(t, self.times, self.hi_values))

    def interval_at(self, t: float) -> Tuple[float, float]:
        return (self.lo_at(t), self.hi_at(t))

    @property
    def t_start(self) -> float:
        return float(self.times[0])

    @property
    def t_end(self) -> float:
        return float(self.times[-1])

    def width_at(self, t: float) -> float:
        lo, hi = self.interval_at(t)
        return hi - lo

    def max_width(self) -> float:
        return float(np.max(self.hi_values - self.lo_values))

    @classmethod
    def from_point_signal(
        cls,
        times: NDArray[np.float64],
        values: NDArray[np.float64],
        uncertainty: float,
        name: str = "",
    ) -> IntervalSignal:
        """Create interval signal from point signal with uniform uncertainty."""
        return cls(times, values - uncertainty, values + uncertainty, name)


# ---------------------------------------------------------------------------
# Interval Model Checker
# ---------------------------------------------------------------------------

class IntervalModelChecker:
    """Three-valued STL model checker over interval signals.

    Evaluates STL formulas conservatively:
    - TRUE only if formula holds for ALL possible signal realizations
    - FALSE only if formula fails for ALL possible signal realizations
    - UNKNOWN if result depends on actual signal within the intervals
    """

    def __init__(self, signals: Dict[str, IntervalSignal]) -> None:
        self._signals = signals
        all_times: Set[float] = set()
        for sig in signals.values():
            all_times.update(sig.times.tolist())
        self._times = np.array(sorted(all_times), dtype=np.float64)

    def check(self, formula: STLFormula, t: float = 0.0) -> ThreeValuedResult:
        """Check formula at time t with three-valued semantics."""
        return self._eval(formula, t)

    def check_trace(self, formula: STLFormula) -> List[ThreeValuedResult]:
        """Check formula at all time points."""
        return [self._eval(formula, float(t)) for t in self._times]

    def _eval(self, formula: STLFormula, t: float) -> ThreeValuedResult:
        if isinstance(formula, Predicate):
            return self._eval_predicate(formula, t)
        if isinstance(formula, STLNot):
            return self._eval_not(formula, t)
        if isinstance(formula, STLAnd):
            return self._eval_and(formula, t)
        if isinstance(formula, STLOr):
            return self._eval_or(formula, t)
        if isinstance(formula, STLImplies):
            return self._eval_implies(formula, t)
        if isinstance(formula, Always):
            return self._eval_always(formula, t)
        if isinstance(formula, Eventually):
            return self._eval_eventually(formula, t)
        if isinstance(formula, Until):
            return self._eval_until(formula, t)
        raise TypeError(f"Unknown formula type: {type(formula)}")

    def _eval_predicate(self, pred: Predicate, t: float) -> ThreeValuedResult:
        """Evaluate predicate over interval signal."""
        expr = pred.expr
        if expr.variable is None or expr.variable not in self._signals:
            raise KeyError(f"Signal '{expr.variable}' not found in interval signals")

        sig = self._signals[expr.variable]
        sig_lo = expr.scale * sig.lo_at(t) + expr.offset
        sig_hi = expr.scale * sig.hi_at(t) + expr.offset

        # Handle negative scale (flips lo/hi)
        if expr.scale < 0:
            sig_lo, sig_hi = sig_hi, sig_lo

        threshold = pred.threshold

        if pred.op in (ComparisonOp.GT, ComparisonOp.GE):
            if sig_lo > threshold:
                return ThreeValuedResult(ThreeValued.TRUE, pred, t,
                                         f"lo={sig_lo:.4f} > {threshold}")
            if sig_hi < threshold:
                return ThreeValuedResult(ThreeValued.FALSE, pred, t,
                                         f"hi={sig_hi:.4f} < {threshold}")
            if sig_hi == threshold and pred.op == ComparisonOp.GT:
                return ThreeValuedResult(ThreeValued.FALSE, pred, t,
                                         f"hi={sig_hi:.4f} = {threshold} (strict)")
        else:  # LT, LE
            if sig_hi < threshold:
                return ThreeValuedResult(ThreeValued.TRUE, pred, t,
                                         f"hi={sig_hi:.4f} < {threshold}")
            if sig_lo > threshold:
                return ThreeValuedResult(ThreeValued.FALSE, pred, t,
                                         f"lo={sig_lo:.4f} > {threshold}")
            if sig_lo == threshold and pred.op == ComparisonOp.LT:
                return ThreeValuedResult(ThreeValued.FALSE, pred, t,
                                         f"lo={sig_lo:.4f} = {threshold} (strict)")

        return ThreeValuedResult(
            ThreeValued.UNKNOWN, pred, t,
            f"interval [{sig_lo:.4f}, {sig_hi:.4f}] straddles {threshold}",
            unknown_regions=[(t, t)],
        )

    def _eval_not(self, formula: STLNot, t: float) -> ThreeValuedResult:
        sub = self._eval(formula.child, t)
        return ThreeValuedResult(
            ~sub.value, formula, t,
            f"NOT({sub.evidence})",
            sub.unknown_regions,
            [sub],
        )

    def _eval_and(self, formula: STLAnd, t: float) -> ThreeValuedResult:
        left = self._eval(formula.left, t)
        right = self._eval(formula.right, t)
        val = left.value & right.value
        unknown = left.unknown_regions + right.unknown_regions
        return ThreeValuedResult(val, formula, t,
                                 f"AND({left.evidence}, {right.evidence})",
                                 unknown, [left, right])

    def _eval_or(self, formula: STLOr, t: float) -> ThreeValuedResult:
        left = self._eval(formula.left, t)
        right = self._eval(formula.right, t)
        val = left.value | right.value
        unknown = left.unknown_regions + right.unknown_regions
        return ThreeValuedResult(val, formula, t,
                                 f"OR({left.evidence}, {right.evidence})",
                                 unknown, [left, right])

    def _eval_implies(self, formula: STLImplies, t: float) -> ThreeValuedResult:
        left = self._eval(formula.antecedent, t)
        right = self._eval(formula.consequent, t)
        val = (~left.value) | right.value
        unknown = left.unknown_regions + right.unknown_regions
        return ThreeValuedResult(val, formula, t,
                                 f"IMPLIES({left.evidence}, {right.evidence})",
                                 unknown, [left, right])

    def _eval_always(self, formula: Always, t: float) -> ThreeValuedResult:
        """G[a,b](phi): TRUE iff phi TRUE at all points in [t+a, t+b],
        FALSE if phi FALSE at any point, UNKNOWN otherwise."""
        lo = t + formula.interval.lo
        hi = t + formula.interval.hi
        relevant_times = self._times_in_range(lo, hi)
        if len(relevant_times) == 0:
            relevant_times = [lo]

        sub_results: List[ThreeValuedResult] = []
        overall = ThreeValued.TRUE
        unknown_regions: List[Tuple[float, float]] = []

        for tp in relevant_times:
            sub = self._eval(formula.child, float(tp))
            sub_results.append(sub)
            if sub.value == ThreeValued.FALSE:
                return ThreeValuedResult(
                    ThreeValued.FALSE, formula, t,
                    f"G: child FALSE at t={tp:.4f}",
                    sub.unknown_regions, sub_results,
                )
            if sub.value == ThreeValued.UNKNOWN:
                overall = ThreeValued.UNKNOWN
                unknown_regions.extend(sub.unknown_regions)

        return ThreeValuedResult(
            overall, formula, t,
            f"G[{formula.interval}]: {'all TRUE' if overall == ThreeValued.TRUE else 'some UNKNOWN'}",
            unknown_regions, sub_results,
        )

    def _eval_eventually(self, formula: Eventually, t: float) -> ThreeValuedResult:
        """F[a,b](phi): TRUE if phi TRUE at some point in [t+a, t+b],
        FALSE if phi FALSE at all points, UNKNOWN otherwise."""
        lo = t + formula.interval.lo
        hi = t + formula.interval.hi
        relevant_times = self._times_in_range(lo, hi)
        if len(relevant_times) == 0:
            relevant_times = [lo]

        sub_results: List[ThreeValuedResult] = []
        overall = ThreeValued.FALSE
        unknown_regions: List[Tuple[float, float]] = []

        for tp in relevant_times:
            sub = self._eval(formula.child, float(tp))
            sub_results.append(sub)
            if sub.value == ThreeValued.TRUE:
                return ThreeValuedResult(
                    ThreeValued.TRUE, formula, t,
                    f"F: child TRUE at t={tp:.4f}",
                    sub.unknown_regions, sub_results,
                )
            if sub.value == ThreeValued.UNKNOWN:
                overall = ThreeValued.UNKNOWN
                unknown_regions.extend(sub.unknown_regions)

        return ThreeValuedResult(
            overall, formula, t,
            f"F[{formula.interval}]: {'all FALSE' if overall == ThreeValued.FALSE else 'some UNKNOWN'}",
            unknown_regions, sub_results,
        )

    def _eval_until(self, formula: Until, t: float) -> ThreeValuedResult:
        """phi1 U[a,b] phi2: conservative evaluation."""
        lo = t + formula.interval.lo
        hi = t + formula.interval.hi
        relevant_times = self._times_in_range(lo, hi)
        if len(relevant_times) == 0:
            return ThreeValuedResult(ThreeValued.FALSE, formula, t,
                                     "U: no time points in interval")

        # Check if there exists a witness: phi2 TRUE at some t' and phi1 TRUE on [lo, t')
        has_unknown = False
        unknown_regions: List[Tuple[float, float]] = []
        sub_results: List[ThreeValuedResult] = []

        for i, tp in enumerate(relevant_times):
            r2 = self._eval(formula.right, float(tp))
            sub_results.append(r2)

            if r2.value == ThreeValued.TRUE:
                # Check phi1 on [lo, tp)
                all_left_true = True
                for tp2 in relevant_times[:i]:
                    r1 = self._eval(formula.left, float(tp2))
                    sub_results.append(r1)
                    if r1.value == ThreeValued.FALSE:
                        all_left_true = False
                        break
                    if r1.value == ThreeValued.UNKNOWN:
                        has_unknown = True
                        unknown_regions.extend(r1.unknown_regions)
                if all_left_true and not has_unknown:
                    return ThreeValuedResult(
                        ThreeValued.TRUE, formula, t,
                        f"U: witness at t={tp:.4f}",
                        unknown_regions, sub_results,
                    )
            elif r2.value == ThreeValued.UNKNOWN:
                has_unknown = True
                unknown_regions.extend(r2.unknown_regions)

        # Check if definitely FALSE: no possible witness
        all_right_false = all(
            self._eval(formula.right, float(tp)).value == ThreeValued.FALSE
            for tp in relevant_times
        )
        if all_right_false:
            return ThreeValuedResult(ThreeValued.FALSE, formula, t,
                                     "U: phi2 FALSE everywhere in interval",
                                     [], sub_results)

        return ThreeValuedResult(
            ThreeValued.UNKNOWN, formula, t,
            "U: inconclusive",
            unknown_regions, sub_results,
        )

    def _times_in_range(self, lo: float, hi: float) -> List[float]:
        """Return time points in [lo, hi]."""
        mask = (self._times >= lo) & (self._times <= hi)
        return self._times[mask].tolist()


# ---------------------------------------------------------------------------
# UNKNOWN region characterization
# ---------------------------------------------------------------------------

@dataclass
class UnknownCharacterization:
    """Characterization of UNKNOWN regions from interval model checking."""
    total_unknown_time: float
    unknown_fraction: float
    unknown_intervals: List[Tuple[float, float]]
    contributing_signals: Dict[str, List[Tuple[float, float]]]

    @property
    def is_tight(self) -> bool:
        """True if unknown fraction is small (< 10%)."""
        return self.unknown_fraction < 0.1


def characterize_unknown(
    result: ThreeValuedResult,
    signals: Dict[str, IntervalSignal],
    total_time: float,
) -> UnknownCharacterization:
    """Analyze UNKNOWN regions: which signals and time intervals cause uncertainty."""
    unknown_ivs = result.refinement_guidance()
    total_unknown = sum(hi - lo for lo, hi in unknown_ivs) if unknown_ivs else 0.0

    # Identify which signals contribute to uncertainty at each UNKNOWN region
    contributing: Dict[str, List[Tuple[float, float]]] = {}
    for sig_name, sig in signals.items():
        sig_regions: List[Tuple[float, float]] = []
        for lo, hi in unknown_ivs:
            # Check if signal has significant width in this region
            t_mid = (lo + hi) / 2.0
            if sig.width_at(t_mid) > 1e-10:
                sig_regions.append((lo, hi))
        if sig_regions:
            contributing[sig_name] = sig_regions

    return UnknownCharacterization(
        total_unknown_time=total_unknown,
        unknown_fraction=total_unknown / total_time if total_time > 0 else 0.0,
        unknown_intervals=unknown_ivs,
        contributing_signals=contributing,
    )
