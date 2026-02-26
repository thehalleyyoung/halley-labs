"""
Compression ratio analysis for CoaCert.

Computes, predicts, and visualises compression ratios across
benchmarks, symmetry types, and parameter sizes.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .metrics import PipelineMetrics, StateSpaceMetrics


class SymmetryType(Enum):
    """Symmetry families that drive quotient compression."""
    NONE = "none"
    CYCLIC = "cyclic"
    FULL = "full"
    ROTATION = "rotation"
    REFLECTION = "reflection"
    DIHEDRAL = "dihedral"
    PRODUCT = "product"
    UNKNOWN = "unknown"


@dataclass
class CompressionPoint:
    """A single compression measurement at a given parameter value."""
    parameter: int
    original_states: int
    quotient_states: int
    original_transitions: int
    quotient_transitions: int
    witness_size_bytes: int = 0
    symmetry: SymmetryType = SymmetryType.UNKNOWN

    @property
    def state_ratio(self) -> float:
        if self.original_states == 0:
            return 1.0
        return self.quotient_states / self.original_states

    @property
    def transition_ratio(self) -> float:
        if self.original_transitions == 0:
            return 1.0
        return self.quotient_transitions / self.original_transitions

    @property
    def witness_ratio(self) -> float:
        """Ratio of witness size to original state count (bytes/state)."""
        if self.original_states == 0:
            return 0.0
        return self.witness_size_bytes / self.original_states

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter,
            "original_states": self.original_states,
            "quotient_states": self.quotient_states,
            "original_transitions": self.original_transitions,
            "quotient_transitions": self.quotient_transitions,
            "witness_size_bytes": self.witness_size_bytes,
            "symmetry": self.symmetry.value,
            "state_ratio": self.state_ratio,
            "transition_ratio": self.transition_ratio,
        }


@dataclass
class CompressionQuality:
    """Scoring of compression quality relative to theoretical bounds."""
    achieved_state_ratio: float
    achieved_transition_ratio: float
    theoretical_state_ratio: float
    theoretical_transition_ratio: float
    score: float  # 0..1, higher is better

    def to_dict(self) -> Dict[str, Any]:
        return {
            "achieved_state_ratio": self.achieved_state_ratio,
            "achieved_transition_ratio": self.achieved_transition_ratio,
            "theoretical_state_ratio": self.theoretical_state_ratio,
            "theoretical_transition_ratio": self.theoretical_transition_ratio,
            "score": self.score,
        }


def _theoretical_bound(symmetry: SymmetryType, parameter: int) -> Tuple[float, float]:
    """Return (state_ratio, transition_ratio) theoretical lower bounds.

    These are the best achievable ratios under exact quotient by the
    given symmetry group.  For full symmetry on n processes,
    |S/~| ≈ |S| / n!; for cyclic, |S/~| ≈ |S| / n, etc.
    """
    n = max(parameter, 1)
    if symmetry == SymmetryType.FULL:
        factorial = math.factorial(n)
        return (1.0 / factorial, 1.0 / factorial)
    if symmetry == SymmetryType.CYCLIC:
        return (1.0 / n, 1.0 / n)
    if symmetry == SymmetryType.DIHEDRAL:
        return (1.0 / (2 * n), 1.0 / (2 * n))
    if symmetry == SymmetryType.ROTATION:
        return (1.0 / n, 1.0 / n)
    if symmetry == SymmetryType.REFLECTION:
        return (0.5, 0.5)
    if symmetry == SymmetryType.PRODUCT:
        return (1.0 / (n * n), 1.0 / (n * n))
    return (1.0, 1.0)


class CompressionAnalyzer:
    """Analyse compression across benchmarks and parameters.

    Usage::

        ca = CompressionAnalyzer()
        ca.add_point(CompressionPoint(...))
        ca.add_from_metrics(pipeline_metrics)
        summary = ca.summary_by_symmetry()
    """

    def __init__(self) -> None:
        self._points: List[CompressionPoint] = []

    @property
    def points(self) -> List[CompressionPoint]:
        return list(self._points)

    def add_point(self, pt: CompressionPoint) -> None:
        self._points.append(pt)

    def add_from_metrics(
        self,
        m: PipelineMetrics,
        parameter: int = 0,
        symmetry: SymmetryType = SymmetryType.UNKNOWN,
    ) -> None:
        ss = m.state_space
        self._points.append(CompressionPoint(
            parameter=parameter,
            original_states=ss.original_states,
            quotient_states=ss.quotient_states,
            original_transitions=ss.original_transitions,
            quotient_transitions=ss.quotient_transitions,
            witness_size_bytes=m.witness.witness_size_bytes,
            symmetry=symmetry,
        ))

    # -- analysis by symmetry type -------------------------------------------

    def points_by_symmetry(self) -> Dict[SymmetryType, List[CompressionPoint]]:
        result: Dict[SymmetryType, List[CompressionPoint]] = {}
        for pt in self._points:
            result.setdefault(pt.symmetry, []).append(pt)
        return result

    def summary_by_symmetry(self) -> Dict[str, Dict[str, float]]:
        """Mean/median/min/max state and transition ratios per symmetry type."""
        out: Dict[str, Dict[str, float]] = {}
        for sym, pts in self.points_by_symmetry().items():
            sr = [p.state_ratio for p in pts]
            tr = [p.transition_ratio for p in pts]
            out[sym.value] = {
                "count": len(pts),
                "state_ratio_mean": statistics.mean(sr),
                "state_ratio_median": statistics.median(sr),
                "state_ratio_min": min(sr),
                "state_ratio_max": max(sr),
                "transition_ratio_mean": statistics.mean(tr),
                "transition_ratio_median": statistics.median(tr),
                "transition_ratio_min": min(tr),
                "transition_ratio_max": max(tr),
            }
        return out

    # -- analysis across parameter sizes -------------------------------------

    def points_sorted_by_parameter(self) -> List[CompressionPoint]:
        return sorted(self._points, key=lambda p: p.parameter)

    def ratio_vs_parameter(self) -> List[Tuple[int, float, float]]:
        """Return [(parameter, state_ratio, transition_ratio)] sorted by parameter."""
        return [
            (p.parameter, p.state_ratio, p.transition_ratio)
            for p in self.points_sorted_by_parameter()
        ]

    # -- extrapolation -------------------------------------------------------

    def _fit_log_linear(
        self, xs: List[float], ys: List[float]
    ) -> Tuple[float, float]:
        """Fit y = a * x^b via log-linear regression. Returns (a, b)."""
        if len(xs) < 2:
            return (1.0, 0.0)
        log_xs = [math.log(max(x, 1e-12)) for x in xs]
        log_ys = [math.log(max(y, 1e-12)) for y in ys]
        n = len(log_xs)
        sum_x = sum(log_xs)
        sum_y = sum(log_ys)
        sum_xx = sum(x * x for x in log_xs)
        sum_xy = sum(x * y for x, y in zip(log_xs, log_ys))
        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-15:
            return (1.0, 0.0)
        b = (n * sum_xy - sum_x * sum_y) / denom
        a_log = (sum_y - b * sum_x) / n
        return (math.exp(a_log), b)

    def predict_state_ratio(self, target_parameter: int) -> float:
        """Extrapolate state compression ratio for a larger parameter value."""
        pts = self.points_sorted_by_parameter()
        if not pts:
            return 1.0
        if len(pts) == 1:
            return pts[0].state_ratio
        xs = [float(p.parameter) for p in pts]
        ys = [p.state_ratio for p in pts]
        a, b = self._fit_log_linear(xs, ys)
        predicted = a * (target_parameter ** b)
        return max(0.0, min(1.0, predicted))

    def predict_transition_ratio(self, target_parameter: int) -> float:
        pts = self.points_sorted_by_parameter()
        if not pts:
            return 1.0
        if len(pts) == 1:
            return pts[0].transition_ratio
        xs = [float(p.parameter) for p in pts]
        ys = [p.transition_ratio for p in pts]
        a, b = self._fit_log_linear(xs, ys)
        predicted = a * (target_parameter ** b)
        return max(0.0, min(1.0, predicted))

    # -- quality scoring -----------------------------------------------------

    def quality_score(self, pt: CompressionPoint) -> CompressionQuality:
        """Score the compression quality of a single point against theory."""
        th_sr, th_tr = _theoretical_bound(pt.symmetry, pt.parameter)
        # Score: how close the achieved ratio is to the theoretical minimum.
        # score = theoretical / achieved (clamped to [0, 1]).
        s_score = th_sr / max(pt.state_ratio, 1e-12) if pt.state_ratio > 0 else 0.0
        t_score = th_tr / max(pt.transition_ratio, 1e-12) if pt.transition_ratio > 0 else 0.0
        combined = min(1.0, (s_score + t_score) / 2.0)
        return CompressionQuality(
            achieved_state_ratio=pt.state_ratio,
            achieved_transition_ratio=pt.transition_ratio,
            theoretical_state_ratio=th_sr,
            theoretical_transition_ratio=th_tr,
            score=combined,
        )

    def all_quality_scores(self) -> List[CompressionQuality]:
        return [self.quality_score(p) for p in self._points]

    # -- comparison with theoretical bounds ----------------------------------

    def gap_from_theory(self) -> List[Dict[str, Any]]:
        """For each point, show the gap between achieved and theoretical."""
        results: List[Dict[str, Any]] = []
        for pt in self._points:
            th_sr, th_tr = _theoretical_bound(pt.symmetry, pt.parameter)
            results.append({
                "parameter": pt.parameter,
                "symmetry": pt.symmetry.value,
                "state_ratio": pt.state_ratio,
                "theoretical_state_ratio": th_sr,
                "state_gap": pt.state_ratio - th_sr,
                "transition_ratio": pt.transition_ratio,
                "theoretical_transition_ratio": th_tr,
                "transition_gap": pt.transition_ratio - th_tr,
            })
        return results

    # -- visualisation data --------------------------------------------------

    def plot_data_state_ratio(self) -> Dict[str, Any]:
        """Return data suitable for a state-ratio vs parameter chart."""
        by_sym = self.points_by_symmetry()
        series: Dict[str, Dict[str, List]] = {}
        for sym, pts in by_sym.items():
            xs = [p.parameter for p in sorted(pts, key=lambda p: p.parameter)]
            ys = [p.state_ratio for p in sorted(pts, key=lambda p: p.parameter)]
            series[sym.value] = {"x": xs, "y": ys}
        return {"title": "State Compression Ratio vs Parameter", "series": series}

    def plot_data_transition_ratio(self) -> Dict[str, Any]:
        by_sym = self.points_by_symmetry()
        series: Dict[str, Dict[str, List]] = {}
        for sym, pts in by_sym.items():
            xs = [p.parameter for p in sorted(pts, key=lambda p: p.parameter)]
            ys = [p.transition_ratio for p in sorted(pts, key=lambda p: p.parameter)]
            series[sym.value] = {"x": xs, "y": ys}
        return {"title": "Transition Compression Ratio vs Parameter", "series": series}

    def plot_data_witness_overhead(self) -> Dict[str, Any]:
        """Witness bytes-per-state vs parameter."""
        pts = self.points_sorted_by_parameter()
        return {
            "title": "Witness Overhead (bytes/state) vs Parameter",
            "x": [p.parameter for p in pts],
            "y": [p.witness_ratio for p in pts],
        }

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self._points],
            "summary_by_symmetry": self.summary_by_symmetry(),
            "quality_scores": [q.to_dict() for q in self.all_quality_scores()],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_points(cls, points: Sequence[CompressionPoint]) -> "CompressionAnalyzer":
        ca = cls()
        for p in points:
            ca.add_point(p)
        return ca

    @classmethod
    def from_metrics_list(
        cls,
        metrics: Sequence[PipelineMetrics],
        parameters: Optional[Sequence[int]] = None,
        symmetry: SymmetryType = SymmetryType.UNKNOWN,
    ) -> "CompressionAnalyzer":
        ca = cls()
        for i, m in enumerate(metrics):
            param = parameters[i] if parameters and i < len(parameters) else i
            ca.add_from_metrics(m, parameter=param, symmetry=symmetry)
        return ca

    # -- comparison across two analyzer instances ----------------------------

    @staticmethod
    def compare(
        a: "CompressionAnalyzer", b: "CompressionAnalyzer"
    ) -> Dict[str, Any]:
        """Compare two analyzers (e.g. two different compression strategies)."""
        a_sr = [p.state_ratio for p in a.points]
        b_sr = [p.state_ratio for p in b.points]
        a_tr = [p.transition_ratio for p in a.points]
        b_tr = [p.transition_ratio for p in b.points]
        result: Dict[str, Any] = {
            "a_count": len(a.points),
            "b_count": len(b.points),
        }
        if a_sr:
            result["a_state_ratio_mean"] = statistics.mean(a_sr)
        if b_sr:
            result["b_state_ratio_mean"] = statistics.mean(b_sr)
        if a_tr:
            result["a_transition_ratio_mean"] = statistics.mean(a_tr)
        if b_tr:
            result["b_transition_ratio_mean"] = statistics.mean(b_tr)
        if a_sr and b_sr:
            result["state_ratio_improvement"] = (
                statistics.mean(a_sr) - statistics.mean(b_sr)
            )
        if a_tr and b_tr:
            result["transition_ratio_improvement"] = (
                statistics.mean(a_tr) - statistics.mean(b_tr)
            )
        return result
