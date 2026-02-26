"""Evaluation metrics for the MARACE (Multi-Agent Race Condition Verifier) system.

Provides metric collection, detection quality assessment, scalability analysis,
and formatting utilities for race-condition verification pipelines.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# MetricCollector
# ---------------------------------------------------------------------------

class MetricCollector:
    """Collect and aggregate named floating-point metrics from analysis runs.

    Each metric name maps to a time-ordered list of recorded values, enabling
    downstream statistical queries (mean, std, percentiles) and serialisation.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, list[float]] = defaultdict(list)

    # -- recording ----------------------------------------------------------

    def record(self, name: str, value: float) -> None:
        """Append *value* under the metric *name*.

        Args:
            name: Identifier for the metric (e.g. ``"detection_recall"``).
            value: Numeric observation to record.
        """
        self._metrics[name].append(float(value))

    # -- retrieval ----------------------------------------------------------

    def get(self, name: str) -> list[float]:
        """Return all recorded values for *name*, or an empty list."""
        return list(self._metrics.get(name, []))

    def mean(self, name: str) -> float:
        """Arithmetic mean of recorded values for *name*.

        Returns:
            The mean, or ``nan`` if no values have been recorded.
        """
        vals = self._metrics.get(name, [])
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    def std(self, name: str) -> float:
        """Population standard deviation for *name*.

        Returns:
            The std-dev, or ``nan`` if no values have been recorded.
        """
        vals = self._metrics.get(name, [])
        if not vals:
            return float("nan")
        return float(np.std(vals))

    def percentile(self, name: str, p: float) -> float:
        """Return the *p*-th percentile (0–100) for *name*.

        Args:
            name: Metric identifier.
            p: Percentile in ``[0, 100]``.

        Returns:
            The percentile value, or ``nan`` if no values exist.
        """
        vals = self._metrics.get(name, [])
        if not vals:
            return float("nan")
        return float(np.percentile(vals, p))

    def all_names(self) -> list[str]:
        """Return a sorted list of all recorded metric names."""
        return sorted(self._metrics.keys())

    def reset(self) -> None:
        """Discard all recorded metrics."""
        self._metrics.clear()

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, list[float]]:
        """Serialise the collector to a plain dictionary."""
        return {k: list(v) for k, v in self._metrics.items()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> "MetricCollector":
        """Reconstruct a :class:`MetricCollector` from a dictionary.

        Args:
            data: Mapping of metric names to lists of floats, as produced
                by :meth:`to_dict`.
        """
        collector = cls()
        for name, values in data.items():
            for v in values:
                collector.record(name, v)
        return collector

    def __repr__(self) -> str:  # pragma: no cover
        n_metrics = len(self._metrics)
        n_values = sum(len(v) for v in self._metrics.values())
        return f"MetricCollector(metrics={n_metrics}, values={n_values})"


# ---------------------------------------------------------------------------
# DetectionRecall
# ---------------------------------------------------------------------------

class DetectionRecall:
    """Fraction of *planted* race conditions that were successfully detected.

    A planted race is considered detected when at least one reported race
    achieves a match score ≥ the configurable *match_threshold*.
    """

    def compute(
        self,
        planted_races: list[dict],
        detected_races: list[dict],
        match_threshold: float = 0.5,
    ) -> float:
        """Compute recall = |matched planted| / |planted|.

        Args:
            planted_races: Ground-truth races, each with keys
                ``"agents_involved"`` (list[str]) and ``"region"``
                (dict with ``"start"``/``"end"`` integer bounds).
            detected_races: Races reported by the analyser (same schema).
            match_threshold: Minimum overlap score to count as a match.

        Returns:
            Recall in ``[0, 1]``, or ``1.0`` when there are no planted races.
        """
        if not planted_races:
            return 1.0

        matched = 0
        for planted in planted_races:
            best = max(
                (self._match_race(planted, det) for det in detected_races),
                default=0.0,
            )
            if best >= match_threshold:
                matched += 1

        return matched / len(planted_races)

    @staticmethod
    def _match_race(planted: dict, detected: dict) -> float:
        """Score how well *detected* matches *planted* (0 = no match, 1 = perfect).

        The score is the mean of two components:

        1. **Agent overlap** – Jaccard similarity of ``agents_involved`` sets.
        2. **Region overlap** – ratio of intersection length to union length of
           the ``[start, end]`` intervals in the ``"region"`` sub-dicts.
        """
        # -- agent overlap ---------------------------------------------------
        p_agents = set(planted.get("agents_involved", []))
        d_agents = set(detected.get("agents_involved", []))
        if p_agents or d_agents:
            agent_score = len(p_agents & d_agents) / len(p_agents | d_agents)
        else:
            agent_score = 1.0

        # -- region overlap --------------------------------------------------
        p_region = planted.get("region", {})
        d_region = detected.get("region", {})

        p_start = p_region.get("start", 0)
        p_end = p_region.get("end", 0)
        d_start = d_region.get("start", 0)
        d_end = d_region.get("end", 0)

        inter_start = max(p_start, d_start)
        inter_end = min(p_end, d_end)
        intersection = max(0, inter_end - inter_start)

        union_start = min(p_start, d_start)
        union_end = max(p_end, d_end)
        union = max(1, union_end - union_start)  # avoid division by zero

        region_score = intersection / union

        return (agent_score + region_score) / 2.0

    def detailed_results(
        self,
        planted_races: list[dict],
        detected_races: list[dict],
        match_threshold: float = 0.5,
    ) -> dict:
        """Per-race matching details.

        Returns:
            A dict with ``"per_race"`` (list of per-planted-race info),
            ``"recall"``, and ``"num_matched"``.
        """
        per_race: list[dict] = []
        num_matched = 0

        for idx, planted in enumerate(planted_races):
            scores = [self._match_race(planted, det) for det in detected_races]
            best_score = max(scores) if scores else 0.0
            best_idx = int(np.argmax(scores)) if scores else -1
            matched = best_score >= match_threshold
            if matched:
                num_matched += 1
            per_race.append({
                "planted_index": idx,
                "best_score": best_score,
                "best_detected_index": best_idx,
                "matched": matched,
            })

        recall = num_matched / len(planted_races) if planted_races else 1.0
        return {
            "per_race": per_race,
            "recall": recall,
            "num_matched": num_matched,
        }


# ---------------------------------------------------------------------------
# FalsePositiveRate
# ---------------------------------------------------------------------------

class FalsePositiveRate:
    """Fraction of *detected* races that do not correspond to any planted race."""

    def compute(
        self,
        detected_races: list[dict],
        planted_races: list[dict],
        match_threshold: float = 0.5,
    ) -> float:
        """Compute FPR = |unmatched detected| / |detected|.

        Args:
            detected_races: Races reported by the analyser.
            planted_races: Ground-truth races.
            match_threshold: Minimum overlap score to count as a match.

        Returns:
            False-positive rate in ``[0, 1]``, or ``0.0`` when nothing is
            detected.
        """
        if not detected_races:
            return 0.0

        recall_helper = DetectionRecall()
        unmatched = 0

        for detected in detected_races:
            best = max(
                (recall_helper._match_race(planted, detected) for planted in planted_races),
                default=0.0,
            )
            if best < match_threshold:
                unmatched += 1

        return unmatched / len(detected_races)


# ---------------------------------------------------------------------------
# SoundCoverage
# ---------------------------------------------------------------------------

class SoundCoverage:
    """Fraction of the state-schedule space that has been certified race-free.

    Regions are axis-aligned hyper-rectangles described by per-dimension
    ``"low"`` / ``"high"`` bounds.
    """

    def compute(
        self,
        verified_regions: list[dict],
        total_bounds: dict,
    ) -> float:
        """Compute coverage = Σ vol(region_i) / vol(total_bounds).

        Overlapping regions are *not* deduplicated; the result can therefore
        exceed ``1.0`` in degenerate inputs (clamped to ``[0, 1]``).

        Args:
            verified_regions: Each dict maps dimension names to
                ``{"low": float, "high": float}``.
            total_bounds: Same schema, defining the full space.

        Returns:
            Coverage fraction in ``[0, 1]``.
        """
        total_vol = self._compute_volume(total_bounds)
        if total_vol <= 0:
            return 0.0

        verified_vol = sum(self._compute_volume(r) for r in verified_regions)
        return min(verified_vol / total_vol, 1.0)

    @staticmethod
    def _compute_volume(region: dict) -> float:
        """Compute the hyper-rectangular volume of *region*.

        Args:
            region: Mapping of dimension names to ``{"low": …, "high": …}``.

        Returns:
            Product of per-dimension extents, or ``0.0`` if any extent is
            non-positive or the region is empty.
        """
        if not region:
            return 0.0

        volume = 1.0
        for dim_name, bounds in region.items():
            low = bounds.get("low", 0.0)
            high = bounds.get("high", 0.0)
            extent = high - low
            if extent <= 0:
                return 0.0
            volume *= extent

        return volume


# ---------------------------------------------------------------------------
# TimeToDetection
# ---------------------------------------------------------------------------

class TimeToDetection:
    """Wall-clock time (seconds) from analysis start to first race detection."""

    def compute(self, detection_log: list[dict]) -> float:
        """Return seconds to the *first* detection.

        Args:
            detection_log: Entries with at least a ``"timestamp"`` key
                (float, seconds since analysis start).

        Returns:
            Time in seconds, or ``inf`` if the log is empty.
        """
        if not detection_log:
            return float("inf")

        timestamps = [
            entry["timestamp"]
            for entry in detection_log
            if "timestamp" in entry
        ]
        if not timestamps:
            return float("inf")
        return float(min(timestamps))

    def compute_all(self, detection_log: list[dict]) -> list[float]:
        """Return a sorted list of timestamps for every detection.

        Args:
            detection_log: Same schema as :meth:`compute`.

        Returns:
            Ascending-sorted list of detection times (seconds).
        """
        timestamps = sorted(
            entry["timestamp"]
            for entry in detection_log
            if "timestamp" in entry
        )
        return [float(t) for t in timestamps]


# ---------------------------------------------------------------------------
# ProbabilityBoundAccuracy
# ---------------------------------------------------------------------------

class ProbabilityBoundAccuracy:
    """Quality of an estimated race probability relative to the true value.

    A *sound* estimate must be ≥ the true probability (conservative bound).
    """

    def compute(self, estimated: float, true_prob: float) -> float:
        """Accuracy ratio ``estimated / true_prob``.

        Returns:
            The ratio (≥ 1 when sound), or ``inf`` when *true_prob* is zero
            and *estimated* is positive, or ``1.0`` when both are zero.
        """
        if true_prob == 0.0:
            return 1.0 if estimated == 0.0 else float("inf")
        return estimated / true_prob

    def is_sound(self, estimated: float, true_prob: float) -> bool:
        """Return ``True`` iff the estimate is a conservative upper bound."""
        return estimated >= true_prob

    def compute_batch(
        self,
        pairs: list[tuple[float, float]],
    ) -> dict:
        """Aggregate accuracy over a batch of ``(estimated, true_prob)`` pairs.

        Returns:
            ``{"mean_ratio": …, "fraction_sound": …, "num_pairs": …}``
        """
        if not pairs:
            return {"mean_ratio": float("nan"), "fraction_sound": float("nan"), "num_pairs": 0}

        ratios: list[float] = []
        sound_count = 0
        for est, true_p in pairs:
            r = self.compute(est, true_p)
            if math.isfinite(r):
                ratios.append(r)
            if self.is_sound(est, true_p):
                sound_count += 1

        mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
        return {
            "mean_ratio": mean_ratio,
            "fraction_sound": sound_count / len(pairs),
            "num_pairs": len(pairs),
        }


# ---------------------------------------------------------------------------
# ScalabilityMetric
# ---------------------------------------------------------------------------

class ScalabilityMetric:
    """Verification time as a function of the number of agents.

    Fits a power-law model ``t = a · n^b`` via log-space linear regression and
    exposes prediction for unseen agent counts.
    """

    def __init__(self) -> None:
        self._coefficient: float = 0.0
        self._exponent: float = 0.0
        self._r_squared: float = 0.0

    def compute(self, timing_data: list[dict]) -> dict:
        """Fit a power-law to *timing_data* and return fit statistics.

        Args:
            timing_data: Each entry must contain ``"num_agents"`` (int) and
                ``"time"`` (float, seconds).

        Returns:
            ``{"coefficient": a, "exponent": b, "r_squared": R²}``

        Raises:
            ValueError: If fewer than two data points are provided.
        """
        if len(timing_data) < 2:
            raise ValueError("Need at least 2 data points for power-law fit")

        ns = [int(d["num_agents"]) for d in timing_data]
        times = [float(d["time"]) for d in timing_data]

        coeff, exp, r2 = self._fit_power_law(ns, times)
        self._coefficient = coeff
        self._exponent = exp
        self._r_squared = r2

        return {"coefficient": coeff, "exponent": exp, "r_squared": r2}

    @staticmethod
    def _fit_power_law(
        ns: list[int],
        times: list[float],
    ) -> tuple[float, float, float]:
        """Fit ``t = a · n^b`` via least-squares in log-log space.

        Args:
            ns: Agent counts (must be > 0).
            times: Corresponding verification times (must be > 0).

        Returns:
            ``(coefficient, exponent, r_squared)``
        """
        log_n = np.log(np.array(ns, dtype=np.float64))
        log_t = np.log(np.array(times, dtype=np.float64))

        # Linear regression: log_t = log_a + b * log_n
        coeffs = np.polyfit(log_n, log_t, 1)
        exponent = float(coeffs[0])
        coefficient = float(np.exp(coeffs[1]))

        # R² (coefficient of determination)
        predicted = coeffs[0] * log_n + coeffs[1]
        ss_res = float(np.sum((log_t - predicted) ** 2))
        ss_tot = float(np.sum((log_t - np.mean(log_t)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        return coefficient, exponent, r_squared

    def predict(self, n: int) -> float:
        """Predict verification time for *n* agents using the fitted model.

        Must call :meth:`compute` before using this method.

        Args:
            n: Number of agents.

        Returns:
            Predicted time in seconds.
        """
        return self._coefficient * (n ** self._exponent)


# ---------------------------------------------------------------------------
# MetricsAggregator
# ---------------------------------------------------------------------------

class MetricsAggregator:
    """Aggregate :class:`MetricCollector` instances across multiple runs."""

    def __init__(self) -> None:
        self._runs: dict[str, MetricCollector] = {}

    def add_run(self, run_id: str, collector: MetricCollector) -> None:
        """Register a collector under *run_id*."""
        self._runs[run_id] = collector

    def aggregate(self) -> dict[str, dict[str, float]]:
        """Compute summary statistics per metric across all runs.

        For each metric name present in *any* run, the per-run means are
        gathered and then summarised with ``mean``, ``std``, ``min``, and
        ``max``.

        Returns:
            ``{metric_name: {"mean": …, "std": …, "min": …, "max": …}}``
        """
        all_names: set[str] = set()
        for collector in self._runs.values():
            all_names.update(collector.all_names())

        result: dict[str, dict[str, float]] = {}
        for name in sorted(all_names):
            per_run_means = [
                c.mean(name)
                for c in self._runs.values()
                if name in c._metrics and c._metrics[name]
            ]
            if not per_run_means:
                result[name] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
            else:
                arr = np.array(per_run_means)
                result[name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
        return result

    def summary_table(self) -> str:
        """Render a human-readable ASCII table of aggregated metrics.

        Returns:
            Formatted table string.
        """
        agg = self.aggregate()
        if not agg:
            return "(no metrics recorded)"

        header = f"{'Metric':<30s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}"
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for name, stats in agg.items():
            lines.append(
                f"{name:<30s} "
                f"{stats['mean']:>10.4f} "
                f"{stats['std']:>10.4f} "
                f"{stats['min']:>10.4f} "
                f"{stats['max']:>10.4f}"
            )
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MetricsFormatter
# ---------------------------------------------------------------------------

class MetricsFormatter:
    """Format metric dictionaries for display and export.

    All methods are static and operate on plain ``dict`` objects (typically
    produced by :meth:`MetricsAggregator.aggregate`).
    """

    @staticmethod
    def to_text(metrics: dict) -> str:
        """Render *metrics* as a human-readable multi-line string."""
        lines: list[str] = []
        for name, value in sorted(metrics.items()):
            if isinstance(value, dict):
                parts = ", ".join(f"{k}={v:.4f}" for k, v in value.items())
                lines.append(f"{name}: {parts}")
            elif isinstance(value, (int, float)):
                lines.append(f"{name}: {value:.4f}")
            else:
                lines.append(f"{name}: {value}")
        return "\n".join(lines)

    @staticmethod
    def to_json(metrics: dict) -> str:
        """Serialise *metrics* to a pretty-printed JSON string."""
        return json.dumps(metrics, indent=2, default=str)

    @staticmethod
    def to_csv(metrics: dict) -> str:
        """Serialise *metrics* to CSV.

        For flat ``{name: value}`` dicts a two-column CSV is produced.  For
        nested ``{name: {stat: value}}`` dicts (aggregated output), columns
        are ``metric, <stat_1>, <stat_2>, …``.
        """
        if not metrics:
            return ""

        first_val = next(iter(metrics.values()))

        if isinstance(first_val, dict):
            stat_keys = sorted(first_val.keys())
            header = "metric," + ",".join(stat_keys)
            rows = [header]
            for name in sorted(metrics):
                vals = ",".join(str(metrics[name].get(k, "")) for k in stat_keys)
                rows.append(f"{name},{vals}")
            return "\n".join(rows)

        # flat dict
        rows = ["metric,value"]
        for name in sorted(metrics):
            rows.append(f"{name},{metrics[name]}")
        return "\n".join(rows)

    @staticmethod
    def to_latex_table(metrics: dict, caption: str = "") -> str:
        """Render *metrics* as a LaTeX ``tabular`` inside a ``table`` float.

        Args:
            metrics: Metric dictionary (flat or nested).
            caption: Optional table caption.

        Returns:
            Complete LaTeX ``table`` environment string.
        """
        lines: list[str] = []
        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")

        first_val = next(iter(metrics.values()), None)

        if isinstance(first_val, dict):
            stat_keys = sorted(first_val.keys())
            col_spec = "l" + "r" * len(stat_keys)
            lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
            lines.append("\\hline")
            header = "Metric & " + " & ".join(k.capitalize() for k in stat_keys) + " \\\\"
            lines.append(header)
            lines.append("\\hline")
            for name in sorted(metrics):
                vals = " & ".join(f"{metrics[name].get(k, 0):.4f}" for k in stat_keys)
                # Escape underscores in metric names for LaTeX
                safe_name = name.replace("_", "\\_")
                lines.append(f"{safe_name} & {vals} \\\\")
        else:
            lines.append("\\begin{tabular}{lr}")
            lines.append("\\hline")
            lines.append("Metric & Value \\\\")
            lines.append("\\hline")
            for name in sorted(metrics):
                safe_name = name.replace("_", "\\_")
                val = metrics[name]
                if isinstance(val, float):
                    lines.append(f"{safe_name} & {val:.4f} \\\\")
                else:
                    lines.append(f"{safe_name} & {val} \\\\")

        lines.append("\\hline")
        lines.append("\\end{tabular}")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        lines.append("\\end{table}")
        return "\n".join(lines)
