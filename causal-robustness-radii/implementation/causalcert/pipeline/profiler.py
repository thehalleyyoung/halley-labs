"""
Built-in profiler for CausalCert pipeline stages.

Provides a context-manager profiler that tracks wall-clock time, memory
usage, and counts per pipeline stage (CI testing, search, estimation).
Generates profiling reports, identifies bottlenecks, fits scalability
curves, and supports run-to-run comparison.
"""

from __future__ import annotations

import logging
import os
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage record
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StageRecord:
    """Profiling record for a single pipeline stage.

    Attributes
    ----------
    name : str
        Stage name (e.g. ``"ci_testing"``, ``"search"``, ``"estimation"``).
    wall_time_s : float
        Elapsed wall-clock time in seconds.
    peak_memory_bytes : int
        Peak memory usage during this stage.
    start_memory_bytes : int
        Memory at stage start.
    end_memory_bytes : int
        Memory at stage end.
    count : int
        Number of operations performed in this stage.
    metadata : dict[str, Any]
        Free-form metadata (e.g. solver name, CI test count).
    """

    name: str
    wall_time_s: float = 0.0
    peak_memory_bytes: int = 0
    start_memory_bytes: int = 0
    end_memory_bytes: int = 0
    count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def memory_delta_bytes(self) -> int:
        """Net memory change (end - start)."""
        return self.end_memory_bytes - self.start_memory_bytes

    @property
    def peak_memory_mb(self) -> float:
        """Peak memory in MiB."""
        return self.peak_memory_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunRecord:
    """Complete profiling record for one CausalCert pipeline run.

    Attributes
    ----------
    run_id : str
        Unique identifier for the run.
    n_nodes : int
        Number of DAG nodes.
    n_edges : int
        Number of DAG edges.
    n_samples : int
        Number of data samples.
    stages : list[StageRecord]
        Per-stage profiling records.
    total_wall_time_s : float
        Total run time.
    total_peak_memory_bytes : int
        Peak memory across all stages.
    timestamp : float
        Unix timestamp when the run completed.
    """

    run_id: str = ""
    n_nodes: int = 0
    n_edges: int = 0
    n_samples: int = 0
    stages: list[StageRecord] = field(default_factory=list)
    total_wall_time_s: float = 0.0
    total_peak_memory_bytes: int = 0
    timestamp: float = 0.0

    def stage_by_name(self, name: str) -> StageRecord | None:
        """Look up a stage by name."""
        for s in self.stages:
            if s.name == name:
                return s
        return None

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    def bottleneck(self) -> StageRecord | None:
        """Return the stage that consumed the most wall-clock time."""
        if not self.stages:
            return None
        return max(self.stages, key=lambda s: s.wall_time_s)

    def memory_bottleneck(self) -> StageRecord | None:
        """Return the stage with the highest peak memory."""
        if not self.stages:
            return None
        return max(self.stages, key=lambda s: s.peak_memory_bytes)


# ---------------------------------------------------------------------------
# Scalability fitting
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScalabilityCurve:
    """Fitted scalability curve from multiple runs.

    Attributes
    ----------
    variable : str
        Independent variable name (``"n_nodes"``, ``"n_samples"``, etc.).
    stage : str
        Pipeline stage name.
    x_values : NDArray[np.float64]
        Observed independent variable values.
    y_values : NDArray[np.float64]
        Observed times in seconds.
    poly_coeffs : NDArray[np.float64]
        Polynomial fit coefficients (highest degree first).
    degree : int
        Degree of the fitted polynomial.
    r_squared : float
        Coefficient of determination.
    """

    variable: str
    stage: str
    x_values: NDArray[np.float64]
    y_values: NDArray[np.float64]
    poly_coeffs: NDArray[np.float64]
    degree: int
    r_squared: float

    def predict(self, x: float) -> float:
        """Predict time for a given *x* value."""
        return float(np.polyval(self.poly_coeffs, x))


def fit_scalability(
    runs: Sequence[RunRecord],
    variable: str,
    stage: str,
    max_degree: int = 3,
) -> ScalabilityCurve | None:
    """Fit a polynomial scalability curve from multiple runs.

    Parameters
    ----------
    runs : Sequence[RunRecord]
        Profiling records from multiple runs with varying parameters.
    variable : str
        Which attribute to use as the independent variable
        (``"n_nodes"``, ``"n_edges"``, or ``"n_samples"``).
    stage : str
        Name of the pipeline stage to model.
    max_degree : int
        Maximum polynomial degree to try.

    Returns
    -------
    ScalabilityCurve | None
        Fitted curve, or ``None`` if insufficient data.
    """
    xs: list[float] = []
    ys: list[float] = []

    for run in runs:
        x_val = getattr(run, variable, None)
        if x_val is None:
            continue
        sr = run.stage_by_name(stage)
        if sr is None:
            continue
        xs.append(float(x_val))
        ys.append(sr.wall_time_s)

    if len(xs) < 2:
        return None

    x_arr = np.array(xs, dtype=np.float64)
    y_arr = np.array(ys, dtype=np.float64)

    # Fit polynomial of increasing degree, select by best R²
    best_r2 = -np.inf
    best_coeffs: NDArray[np.float64] = np.array([0.0])
    best_deg = 1

    y_mean = np.mean(y_arr)
    ss_tot = np.sum((y_arr - y_mean) ** 2)
    if ss_tot < 1e-15:
        ss_tot = 1.0

    for deg in range(1, min(max_degree + 1, len(xs))):
        coeffs = np.polyfit(x_arr, y_arr, deg)
        y_pred = np.polyval(coeffs, x_arr)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        r2 = 1.0 - ss_res / ss_tot

        if r2 > best_r2:
            best_r2 = r2
            best_coeffs = coeffs
            best_deg = deg

    return ScalabilityCurve(
        variable=variable,
        stage=stage,
        x_values=x_arr,
        y_values=y_arr,
        poly_coeffs=best_coeffs,
        degree=best_deg,
        r_squared=float(best_r2),
    )


# ---------------------------------------------------------------------------
# CausalCertProfiler
# ---------------------------------------------------------------------------


class CausalCertProfiler:
    """Context-manager profiler for CausalCert pipeline stages.

    Usage
    -----
    >>> profiler = CausalCertProfiler(run_id="run-001", n_nodes=20, n_edges=30)
    >>> with profiler.stage("ci_testing"):
    ...     # perform CI tests
    ...     pass
    >>> with profiler.stage("search"):
    ...     # run solver
    ...     pass
    >>> report = profiler.report()
    >>> print(report)
    """

    def __init__(
        self,
        run_id: str = "",
        n_nodes: int = 0,
        n_edges: int = 0,
        n_samples: int = 0,
        track_memory: bool = True,
    ) -> None:
        self._run_id = run_id or f"run-{int(time.time())}"
        self._n_nodes = n_nodes
        self._n_edges = n_edges
        self._n_samples = n_samples
        self._track_memory = track_memory

        self._stages: list[StageRecord] = []
        self._active_stage: str | None = None
        self._run_start: float = time.monotonic()

        self._tracemalloc_was_tracing = tracemalloc.is_tracing()
        if self._track_memory and not self._tracemalloc_was_tracing:
            tracemalloc.start()

    @contextmanager
    def stage(
        self,
        name: str,
        count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[StageRecord]:
        """Profile a pipeline stage.

        Parameters
        ----------
        name : str
            Stage name (e.g. ``"ci_testing"``).
        count : int
            Number of operations in this stage.
        metadata : dict[str, Any] | None
            Additional metadata.

        Yields
        ------
        StageRecord
            The stage record (can be mutated to add count/metadata).
        """
        rec = StageRecord(
            name=name,
            count=count,
            metadata=metadata or {},
        )
        self._active_stage = name

        # Memory snapshot
        if self._track_memory and tracemalloc.is_tracing():
            snapshot_start = tracemalloc.take_snapshot()
            mem_start = sum(s.size for s in snapshot_start.statistics("filename"))
            rec.start_memory_bytes = mem_start
        else:
            mem_start = 0

        t_start = time.monotonic()
        peak_during = 0

        try:
            yield rec
        finally:
            t_end = time.monotonic()
            rec.wall_time_s = t_end - t_start

            if self._track_memory and tracemalloc.is_tracing():
                snapshot_end = tracemalloc.take_snapshot()
                mem_end = sum(s.size for s in snapshot_end.statistics("filename"))
                rec.end_memory_bytes = mem_end
                _, peak = tracemalloc.get_traced_memory()
                rec.peak_memory_bytes = peak
            else:
                rec.end_memory_bytes = 0
                rec.peak_memory_bytes = 0

            self._stages.append(rec)
            self._active_stage = None

            logger.debug(
                "Stage '%s': %.3fs, peak_mem=%.1f MiB, count=%d",
                name, rec.wall_time_s, rec.peak_memory_mb, rec.count,
            )

    def record(self) -> RunRecord:
        """Finalise and return the :class:`RunRecord`.

        Returns
        -------
        RunRecord
        """
        total_wall = time.monotonic() - self._run_start
        peak_mem = max((s.peak_memory_bytes for s in self._stages), default=0)

        return RunRecord(
            run_id=self._run_id,
            n_nodes=self._n_nodes,
            n_edges=self._n_edges,
            n_samples=self._n_samples,
            stages=list(self._stages),
            total_wall_time_s=total_wall,
            total_peak_memory_bytes=peak_mem,
            timestamp=time.time(),
        )

    def report(self, width: int = 80) -> str:
        """Generate a human-readable profiling report.

        Parameters
        ----------
        width : int
            Character width of the report.

        Returns
        -------
        str
        """
        rec = self.record()
        lines: list[str] = []
        sep = "=" * width

        lines.append(sep)
        lines.append(f"CausalCert Profiling Report — {rec.run_id}")
        lines.append(sep)
        lines.append(
            f"DAG: {rec.n_nodes} nodes, {rec.n_edges} edges  |  "
            f"Data: {rec.n_samples} samples"
        )
        lines.append(f"Total wall-clock: {rec.total_wall_time_s:.3f}s")
        lines.append(f"Peak memory: {rec.total_peak_memory_bytes / 1024 / 1024:.1f} MiB")
        lines.append("")

        # Stage table
        lines.append(f"{'Stage':<20} {'Time (s)':>10} {'%':>6} {'Peak MiB':>10} {'Count':>8}")
        lines.append("-" * width)

        total_t = max(sum(s.wall_time_s for s in rec.stages), 1e-9)
        for s in rec.stages:
            pct = 100.0 * s.wall_time_s / total_t
            lines.append(
                f"{s.name:<20} {s.wall_time_s:>10.3f} {pct:>5.1f}% "
                f"{s.peak_memory_mb:>10.1f} {s.count:>8d}"
            )
        lines.append("-" * width)

        # Bottleneck
        bn = rec.bottleneck()
        if bn:
            lines.append(f"⚠ Time bottleneck: '{bn.name}' ({bn.wall_time_s:.3f}s)")
        mem_bn = rec.memory_bottleneck()
        if mem_bn:
            lines.append(
                f"⚠ Memory bottleneck: '{mem_bn.name}' ({mem_bn.peak_memory_mb:.1f} MiB)"
            )

        lines.append(sep)
        return "\n".join(lines)

    def close(self) -> None:
        """Stop tracemalloc if we started it."""
        if self._track_memory and not self._tracemalloc_was_tracing:
            try:
                tracemalloc.stop()
            except RuntimeError:
                pass

    def __enter__(self) -> CausalCertProfiler:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"CausalCertProfiler(run_id='{self._run_id}', "
            f"stages={len(self._stages)})"
        )


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Comparison between two profiling runs.

    Attributes
    ----------
    run_a : str
        Run ID of the first run.
    run_b : str
        Run ID of the second run.
    time_speedup : float
        Speedup factor (a_time / b_time); >1 means *b* is faster.
    memory_ratio : float
        Memory ratio (a_peak / b_peak); >1 means *b* uses less memory.
    stage_comparisons : dict[str, dict[str, float]]
        Per-stage time and memory comparisons.
    """

    run_a: str
    run_b: str
    time_speedup: float
    memory_ratio: float
    stage_comparisons: dict[str, dict[str, float]]


def compare_runs(a: RunRecord, b: RunRecord) -> ComparisonResult:
    """Compare two profiling runs.

    Parameters
    ----------
    a, b : RunRecord
        The two runs to compare.

    Returns
    -------
    ComparisonResult
    """
    time_speedup = a.total_wall_time_s / max(b.total_wall_time_s, 1e-9)

    mem_a = max(a.total_peak_memory_bytes, 1)
    mem_b = max(b.total_peak_memory_bytes, 1)
    memory_ratio = mem_a / mem_b

    stage_cmp: dict[str, dict[str, float]] = {}
    all_names = set(a.stage_names) | set(b.stage_names)
    for name in sorted(all_names):
        sa = a.stage_by_name(name)
        sb = b.stage_by_name(name)
        t_a = sa.wall_time_s if sa else 0.0
        t_b = sb.wall_time_s if sb else 0.0
        m_a = sa.peak_memory_bytes if sa else 0
        m_b = sb.peak_memory_bytes if sb else 0
        stage_cmp[name] = {
            "time_a_s": t_a,
            "time_b_s": t_b,
            "time_speedup": t_a / max(t_b, 1e-9),
            "memory_a_bytes": float(m_a),
            "memory_b_bytes": float(m_b),
            "memory_ratio": m_a / max(m_b, 1),
        }

    return ComparisonResult(
        run_a=a.run_id,
        run_b=b.run_id,
        time_speedup=time_speedup,
        memory_ratio=memory_ratio,
        stage_comparisons=stage_cmp,
    )


def format_comparison(cmp: ComparisonResult, width: int = 80) -> str:
    """Format a comparison result as a human-readable string.

    Parameters
    ----------
    cmp : ComparisonResult
        Comparison to format.
    width : int
        Line width.

    Returns
    -------
    str
    """
    lines: list[str] = []
    sep = "=" * width
    lines.append(sep)
    lines.append(f"Run Comparison: {cmp.run_a} vs {cmp.run_b}")
    lines.append(sep)
    lines.append(f"Overall speedup: {cmp.time_speedup:.2f}x")
    lines.append(f"Memory ratio: {cmp.memory_ratio:.2f}x")
    lines.append("")

    lines.append(
        f"{'Stage':<20} {'A (s)':>8} {'B (s)':>8} {'Speedup':>8} "
        f"{'A MiB':>8} {'B MiB':>8}"
    )
    lines.append("-" * width)

    for name, sc in cmp.stage_comparisons.items():
        lines.append(
            f"{name:<20} "
            f"{sc['time_a_s']:>8.3f} {sc['time_b_s']:>8.3f} "
            f"{sc['time_speedup']:>7.2f}x "
            f"{sc['memory_a_bytes'] / 1024 / 1024:>8.1f} "
            f"{sc['memory_b_bytes'] / 1024 / 1024:>8.1f}"
        )

    lines.append(sep)
    return "\n".join(lines)
