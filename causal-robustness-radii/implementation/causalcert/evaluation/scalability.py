"""
Scalability profiling for CausalCert components.

Measures wall-clock time and memory usage of the solver, CI tester,
and fragility scorer across a range of DAG sizes and densities.
"""

from __future__ import annotations

import gc
import logging
import os
import resource
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from causalcert.types import AdjacencyMatrix

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProfileResult:
    """Single profiling measurement.

    Attributes
    ----------
    n_nodes : int
        Number of DAG nodes.
    n_edges : int
        Number of DAG edges.
    component : str
        Component name (e.g. ``"solver"``, ``"ci_test"``, ``"fragility"``).
    time_s : float
        Wall-clock time in seconds.
    peak_memory_mb : float
        Peak memory usage in MB.
    extra : dict[str, Any]
        Additional metrics.
    """

    n_nodes: int
    n_edges: int
    component: str
    time_s: float
    peak_memory_mb: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Random DAG generation utilities
# ---------------------------------------------------------------------------


def _random_dag(
    n_nodes: int,
    density: float,
    rng: np.random.RandomState,
) -> AdjacencyMatrix:
    """Generate a random DAG (Erdos-Renyi with topological ordering).

    Parameters
    ----------
    n_nodes : int
    density : float
        Expected fraction of possible edges present.
    rng : np.random.RandomState

    Returns
    -------
    AdjacencyMatrix
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < density:
                adj[i, j] = 1
    return adj


# ---------------------------------------------------------------------------
# Memory measurement helpers
# ---------------------------------------------------------------------------


def _measure_peak_memory(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Execute *fn* and return ``(result, peak_memory_mb)``."""
    gc.collect()
    tracemalloc.start()
    try:
        result = fn()
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return result, peak / (1024 * 1024)


def _timed_call(fn: Callable[[], Any]) -> tuple[Any, float, float]:
    """Execute *fn* and return ``(result, elapsed_s, peak_mb)``."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        result = fn()
    finally:
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Standard component wrappers
# ---------------------------------------------------------------------------


def profile_pipeline(
    adj: AdjacencyMatrix,
    treatment: int = 0,
    outcome: int | None = None,
    n_samples: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Profile the full CausalCert pipeline on a DAG.

    Returns timing breakdown by stage.
    """
    n = adj.shape[0]
    if outcome is None:
        outcome = n - 1

    from causalcert.evaluation.dgp import generate_linear_gaussian
    rng = np.random.RandomState(seed)
    df, _, _ = generate_linear_gaussian(adj, n_samples=n_samples, treatment=treatment, rng=rng)

    timings: dict[str, float] = {}

    # CI testing
    from causalcert.dag.dsep import DSeparationOracle
    from causalcert.dag.graph import CausalDAG

    dag = CausalDAG(adj)
    oracle = DSeparationOracle(dag)

    t0 = time.perf_counter()
    _ = oracle.all_ci_implications()
    timings["dsep_implications"] = time.perf_counter() - t0

    # Fragility scoring
    try:
        from causalcert.fragility.scorer import compute_fragility_scores
        t0 = time.perf_counter()
        _ = compute_fragility_scores(dag, treatment, outcome)
        timings["fragility_scoring"] = time.perf_counter() - t0
    except Exception:
        timings["fragility_scoring"] = -1.0

    # Adjustment set
    try:
        from causalcert.estimation.adjustment import find_optimal_adjustment_set
        t0 = time.perf_counter()
        _ = find_optimal_adjustment_set(dag, treatment, outcome)
        timings["adjustment_set"] = time.perf_counter() - t0
    except Exception:
        timings["adjustment_set"] = -1.0

    return timings


# ---------------------------------------------------------------------------
# Main profiler class
# ---------------------------------------------------------------------------


class ScalabilityProfiler:
    """Scalability profiling harness.

    Parameters
    ----------
    n_nodes_list : list[int]
        DAG sizes to test.
    density_list : list[float]
        Edge densities to test.
    n_repeats : int
        Repetitions per configuration.
    seed : int
        Random seed.
    timeout_s : float
        Per-call timeout in seconds (skip if exceeded).
    """

    def __init__(
        self,
        n_nodes_list: list[int] | None = None,
        density_list: list[float] | None = None,
        n_repeats: int = 3,
        seed: int = 42,
        timeout_s: float = 300.0,
    ) -> None:
        self.n_nodes_list = n_nodes_list or [10, 20, 50, 100, 200]
        self.density_list = density_list or [0.1, 0.2, 0.3]
        self.n_repeats = n_repeats
        self.seed = seed
        self.timeout_s = timeout_s

    def profile(
        self,
        component_fn: Callable[..., Any],
        component_name: str,
    ) -> list[ProfileResult]:
        """Profile a component across all DAG sizes and densities.

        Parameters
        ----------
        component_fn : Callable
            Function to profile.  Signature: ``fn(adj, **kwargs)``.
        component_name : str
            Name for reporting.

        Returns
        -------
        list[ProfileResult]
        """
        results: list[ProfileResult] = []
        rng = np.random.RandomState(self.seed)

        for n_nodes in self.n_nodes_list:
            for density in self.density_list:
                times: list[float] = []
                mems: list[float] = []
                extras: list[dict[str, Any]] = []

                for rep in range(self.n_repeats):
                    adj = _random_dag(n_nodes, density, rng)
                    n_edges = int(adj.sum())

                    try:
                        result, elapsed, peak_mb = _timed_call(
                            lambda a=adj: component_fn(a)
                        )
                        if elapsed > self.timeout_s:
                            logger.warning(
                                "%s timed out for n=%d d=%.2f (%.1fs)",
                                component_name, n_nodes, density, elapsed,
                            )

                        times.append(elapsed)
                        mems.append(peak_mb)

                        extra: dict[str, Any] = {"repeat": rep, "density": density}
                        if isinstance(result, dict):
                            extra.update(result)
                        extras.append(extra)

                    except Exception as exc:
                        logger.warning(
                            "%s failed for n=%d d=%.2f: %s",
                            component_name, n_nodes, density, exc,
                        )
                        times.append(float("nan"))
                        mems.append(float("nan"))
                        extras.append({"repeat": rep, "density": density, "error": str(exc)})

                for i, (t, m, ex) in enumerate(zip(times, mems, extras)):
                    adj_ex = _random_dag(n_nodes, density, rng) if i > 0 else adj
                    results.append(ProfileResult(
                        n_nodes=n_nodes,
                        n_edges=int(adj.sum()),
                        component=component_name,
                        time_s=t,
                        peak_memory_mb=m,
                        extra=ex,
                    ))

        return results

    def profile_multiple(
        self,
        components: dict[str, Callable[..., Any]],
    ) -> list[ProfileResult]:
        """Profile multiple components.

        Parameters
        ----------
        components : dict[str, Callable]
            Mapping from component name to function.

        Returns
        -------
        list[ProfileResult]
        """
        all_results: list[ProfileResult] = []
        for name, fn in components.items():
            logger.info("Profiling component: %s", name)
            all_results.extend(self.profile(fn, name))
        return all_results

    def profile_full_pipeline(
        self,
        n_samples: int = 500,
    ) -> list[ProfileResult]:
        """Profile the full pipeline across DAG sizes.

        Parameters
        ----------
        n_samples : int
            Samples per instance.

        Returns
        -------
        list[ProfileResult]
        """
        results: list[ProfileResult] = []
        rng = np.random.RandomState(self.seed)

        for n_nodes in self.n_nodes_list:
            for density in self.density_list:
                for rep in range(self.n_repeats):
                    adj = _random_dag(n_nodes, density, rng)
                    n_edges = int(adj.sum())

                    try:
                        _, elapsed, peak_mb = _timed_call(
                            lambda a=adj: profile_pipeline(
                                a, treatment=0, outcome=n_nodes - 1,
                                n_samples=n_samples, seed=self.seed + rep,
                            )
                        )
                        timings = profile_pipeline(
                            adj, treatment=0, outcome=n_nodes - 1,
                            n_samples=n_samples, seed=self.seed + rep,
                        )
                        results.append(ProfileResult(
                            n_nodes=n_nodes,
                            n_edges=n_edges,
                            component="full_pipeline",
                            time_s=elapsed,
                            peak_memory_mb=peak_mb,
                            extra={
                                "repeat": rep,
                                "density": density,
                                **timings,
                            },
                        ))
                    except Exception as exc:
                        logger.warning(
                            "Full pipeline failed n=%d d=%.2f: %s",
                            n_nodes, density, exc,
                        )
                        results.append(ProfileResult(
                            n_nodes=n_nodes,
                            n_edges=n_edges,
                            component="full_pipeline",
                            time_s=float("nan"),
                            peak_memory_mb=float("nan"),
                            extra={"repeat": rep, "density": density, "error": str(exc)},
                        ))

        return results

    def summary_table(self, results: list[ProfileResult]) -> pd.DataFrame:
        """Create a summary DataFrame from profiling results.

        Parameters
        ----------
        results : list[ProfileResult]

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for r in results:
            row = {
                "n_nodes": r.n_nodes,
                "n_edges": r.n_edges,
                "component": r.component,
                "time_s": round(r.time_s, 4) if not np.isnan(r.time_s) else None,
                "peak_memory_mb": round(r.peak_memory_mb, 2) if not np.isnan(r.peak_memory_mb) else None,
            }
            row.update(r.extra)
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def aggregate_summary(self, results: list[ProfileResult]) -> pd.DataFrame:
        """Create an aggregated summary by (n_nodes, component).

        Parameters
        ----------
        results : list[ProfileResult]

        Returns
        -------
        pd.DataFrame
            Columns: ``n_nodes``, ``component``, ``mean_time``, ``std_time``,
            ``mean_memory``, ``std_memory``, ``n_runs``.
        """
        df = self.summary_table(results)
        if df.empty:
            return pd.DataFrame()

        agg = df.groupby(["n_nodes", "component"]).agg(
            mean_time=("time_s", "mean"),
            std_time=("time_s", "std"),
            median_time=("time_s", "median"),
            max_time=("time_s", "max"),
            mean_memory=("peak_memory_mb", "mean"),
            std_memory=("peak_memory_mb", "std"),
            n_runs=("time_s", "count"),
        ).reset_index()

        # Round
        for col in ["mean_time", "std_time", "median_time", "max_time"]:
            agg[col] = agg[col].round(4)
        for col in ["mean_memory", "std_memory"]:
            agg[col] = agg[col].round(2)

        return agg

    def scaling_exponent(self, results: list[ProfileResult]) -> pd.DataFrame:
        """Estimate scaling exponent (time ~ n^alpha) per component.

        Fits log(time) = alpha * log(n) + c by ordinary least squares.

        Parameters
        ----------
        results : list[ProfileResult]

        Returns
        -------
        pd.DataFrame
            Columns: ``component``, ``alpha``, ``r_squared``.
        """
        df = self.summary_table(results)
        if df.empty:
            return pd.DataFrame(columns=["component", "alpha", "r_squared"])

        rows = []
        for component, group in df.groupby("component"):
            valid = group.dropna(subset=["time_s"])
            valid = valid[valid["time_s"] > 0]
            if len(valid) < 2:
                rows.append({"component": component, "alpha": float("nan"), "r_squared": 0.0})
                continue

            log_n = np.log(valid["n_nodes"].values.astype(float))
            log_t = np.log(valid["time_s"].values.astype(float))

            # OLS
            A = np.vstack([log_n, np.ones(len(log_n))]).T
            coeff, residuals, rank, sv = np.linalg.lstsq(A, log_t, rcond=None)
            alpha = coeff[0]

            # R^2
            ss_res = np.sum((log_t - A @ coeff) ** 2)
            ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            rows.append({
                "component": component,
                "alpha": round(float(alpha), 3),
                "r_squared": round(float(r2), 3),
            })

        return pd.DataFrame(rows)
