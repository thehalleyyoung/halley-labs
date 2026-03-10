"""
Comprehensive benchmark suite for CausalCert evaluation.

Provides a standardised evaluation protocol with configurable benchmark
sizes (small, medium, large), wall-clock timing with warmup, paired
permutation tests for method comparison, and result serialization.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from causalcert.types import (
    AdjacencyMatrix,
    EditType,
    NodeId,
    RobustnessRadius,
    StructuralEdit,
)
from causalcert.evaluation.dgp import DGPInstance, SyntheticDGP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BenchmarkConfig:
    """Configuration for a single benchmark tier.

    Attributes
    ----------
    name : str
        Human-readable tier name (e.g. ``"small"``).
    n_nodes_range : tuple[int, int]
        ``(min, max)`` number of nodes.
    n_instances : int
        Number of random DGP instances.
    n_samples : int
        Observations per instance.
    density_range : tuple[float, float]
        ``(min, max)`` edge density.
    radius_range : tuple[int, int]
        ``(min, max)`` true robustness radius.
    max_k : int
        Maximum edit distance for the solver.
    warmup_runs : int
        Number of warmup iterations (discarded from timing).
    seed : int
        Random seed for reproducibility.
    """

    name: str = "small"
    n_nodes_range: tuple[int, int] = (4, 10)
    n_instances: int = 50
    n_samples: int = 500
    density_range: tuple[float, float] = (0.15, 0.35)
    radius_range: tuple[int, int] = (1, 3)
    max_k: int = 5
    warmup_runs: int = 2
    seed: int = 42


# Pre-defined tier configurations
SMALL_CONFIG = BenchmarkConfig(
    name="small",
    n_nodes_range=(4, 10),
    n_instances=50,
    n_samples=500,
    density_range=(0.15, 0.35),
    radius_range=(1, 3),
    max_k=5,
    warmup_runs=2,
)

MEDIUM_CONFIG = BenchmarkConfig(
    name="medium",
    n_nodes_range=(10, 30),
    n_instances=30,
    n_samples=1000,
    density_range=(0.10, 0.25),
    radius_range=(1, 5),
    max_k=8,
    warmup_runs=1,
)

LARGE_CONFIG = BenchmarkConfig(
    name="large",
    n_nodes_range=(30, 100),
    n_instances=10,
    n_samples=2000,
    density_range=(0.05, 0.15),
    radius_range=(1, 8),
    max_k=12,
    warmup_runs=1,
)


# ---------------------------------------------------------------------------
# Method specification
# ---------------------------------------------------------------------------

MethodFn = Callable[[AdjacencyMatrix, pd.DataFrame, int, int, int], RobustnessRadius]
"""Signature: (adj, data, treatment, outcome, max_k) -> RobustnessRadius."""


@dataclass(slots=True)
class MethodSpec:
    """Specification for a method under evaluation.

    Attributes
    ----------
    name : str
        Method name (e.g. ``"causalcert_ilp"``).
    fn : MethodFn
        Callable implementing the method.
    is_baseline : bool
        ``True`` for baseline methods.
    """

    name: str
    fn: MethodFn
    is_baseline: bool = False


# ---------------------------------------------------------------------------
# Per-instance result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InstanceResult:
    """Result of running a method on a single DGP instance.

    Attributes
    ----------
    instance_id : int
        Index of the instance.
    method : str
        Method name.
    true_radius : int
        Ground-truth radius.
    estimated_lb : int
        Estimated lower bound.
    estimated_ub : int
        Estimated upper bound.
    exact : bool
        ``True`` if LB == UB == true_radius.
    covered : bool
        ``True`` if true_radius in [LB, UB].
    wall_time_s : float
        Wall-clock time in seconds (excluding warmup).
    error : str
        Error message if the method failed, else ``""``.
    """

    instance_id: int
    method: str
    true_radius: int
    estimated_lb: int = 0
    estimated_ub: int = 0
    exact: bool = False
    covered: bool = False
    wall_time_s: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AggregateResult:
    """Aggregate statistics for a method on a benchmark tier.

    Attributes
    ----------
    method : str
    tier : str
    n_instances : int
    coverage : float
    exact_match : float
    mae : float
    mean_width : float
    mean_time_s : float
    median_time_s : float
    p95_time_s : float
    total_time_s : float
    n_errors : int
    """

    method: str = ""
    tier: str = ""
    n_instances: int = 0
    coverage: float = 0.0
    exact_match: float = 0.0
    mae: float = 0.0
    mean_width: float = 0.0
    mean_time_s: float = 0.0
    median_time_s: float = 0.0
    p95_time_s: float = 0.0
    total_time_s: float = 0.0
    n_errors: int = 0


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _timed_call(
    fn: MethodFn,
    adj: AdjacencyMatrix,
    data: pd.DataFrame,
    treatment: int,
    outcome: int,
    max_k: int,
) -> tuple[RobustnessRadius | None, float, str]:
    """Run *fn* and return (result, elapsed_seconds, error_message)."""
    start = time.perf_counter()
    try:
        result = fn(adj, data, treatment, outcome, max_k)
        elapsed = time.perf_counter() - start
        return result, elapsed, ""
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return None, elapsed, str(exc)


def _warmup(
    fn: MethodFn,
    instance: DGPInstance,
    max_k: int,
    n_warmup: int,
) -> None:
    """Run *fn* a few times to warm up JIT / caches."""
    for _ in range(n_warmup):
        try:
            fn(instance.adj, instance.data, instance.treatment,
               instance.outcome, max_k)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Paired permutation test
# ---------------------------------------------------------------------------


def paired_permutation_test(
    metric_a: Sequence[float],
    metric_b: Sequence[float],
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Two-sided paired permutation test for H₀: mean(A) = mean(B).

    Parameters
    ----------
    metric_a, metric_b : Sequence[float]
        Paired metric values.
    n_permutations : int
    seed : int

    Returns
    -------
    dict[str, float]
        ``observed_diff``, ``p_value``, ``ci_lower``, ``ci_upper``.
    """
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return {"observed_diff": 0.0, "p_value": 1.0,
                "ci_lower": 0.0, "ci_upper": 0.0}

    diffs = a - b
    observed = float(np.mean(diffs))
    rng = np.random.RandomState(seed)

    perm_means = np.empty(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_means[i] = float(np.mean(diffs * signs))

    p_value = float((np.sum(np.abs(perm_means) >= abs(observed)) + 1)
                     / (n_permutations + 1))

    ci_lower = float(np.percentile(perm_means, 2.5))
    ci_upper = float(np.percentile(perm_means, 97.5))

    return {
        "observed_diff": observed,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Standardised benchmark evaluation protocol.

    Parameters
    ----------
    configs : Sequence[BenchmarkConfig] | None
        Benchmark tier configurations.  Defaults to small/medium/large.
    methods : Sequence[MethodSpec] | None
        Methods to evaluate.  Must be set before :meth:`run`.
    """

    def __init__(
        self,
        configs: Sequence[BenchmarkConfig] | None = None,
        methods: Sequence[MethodSpec] | None = None,
    ) -> None:
        self.configs: list[BenchmarkConfig] = list(
            configs or [SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG]
        )
        self.methods: list[MethodSpec] = list(methods or [])
        self.instance_results: list[InstanceResult] = []
        self.aggregate_results: list[AggregateResult] = []
        self.comparison_tests: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Instance generation
    # ------------------------------------------------------------------

    def generate_instances(
        self, config: BenchmarkConfig
    ) -> list[DGPInstance]:
        """Generate synthetic DGP instances for a benchmark tier.

        Parameters
        ----------
        config : BenchmarkConfig

        Returns
        -------
        list[DGPInstance]
        """
        rng = np.random.RandomState(config.seed)
        dgp = SyntheticDGP(seed=config.seed)
        instances: list[DGPInstance] = []

        for i in range(config.n_instances):
            n_nodes = rng.randint(config.n_nodes_range[0],
                                  config.n_nodes_range[1] + 1)
            density = rng.uniform(*config.density_range)
            radius = rng.randint(config.radius_range[0],
                                 config.radius_range[1] + 1)
            inst = dgp.generate(
                n_nodes=n_nodes,
                n_samples=config.n_samples,
                density=density,
                true_radius=radius,
                seed=config.seed + i,
            )
            instances.append(inst)
        return instances

    # ------------------------------------------------------------------
    # Run benchmark
    # ------------------------------------------------------------------

    def run(
        self,
        configs: Sequence[BenchmarkConfig] | None = None,
    ) -> list[AggregateResult]:
        """Execute the full benchmark suite.

        Parameters
        ----------
        configs : Sequence[BenchmarkConfig] | None
            Override tier configs; defaults to ``self.configs``.

        Returns
        -------
        list[AggregateResult]
        """
        configs = configs or self.configs
        self.instance_results = []
        self.aggregate_results = []

        for config in configs:
            logger.info("Benchmark tier: %s", config.name)
            instances = self.generate_instances(config)

            for method_spec in self.methods:
                logger.info("  Method: %s", method_spec.name)
                # Warmup
                if instances and config.warmup_runs > 0:
                    _warmup(method_spec.fn, instances[0],
                            config.max_k, config.warmup_runs)

                method_results: list[InstanceResult] = []
                for idx, inst in enumerate(instances):
                    result, elapsed, error = _timed_call(
                        method_spec.fn, inst.adj, inst.data,
                        inst.treatment, inst.outcome, config.max_k,
                    )
                    ir = InstanceResult(
                        instance_id=idx,
                        method=method_spec.name,
                        true_radius=inst.true_radius,
                    )
                    if result is not None:
                        ir.estimated_lb = result.lower_bound
                        ir.estimated_ub = result.upper_bound
                        ir.exact = (result.lower_bound == result.upper_bound
                                    == inst.true_radius)
                        ir.covered = (result.lower_bound <= inst.true_radius
                                      <= result.upper_bound)
                    ir.wall_time_s = elapsed
                    ir.error = error
                    method_results.append(ir)
                    self.instance_results.append(ir)

                agg = self._aggregate(method_results, method_spec.name,
                                      config.name)
                self.aggregate_results.append(agg)

        self._run_comparisons()
        return self.aggregate_results

    def _aggregate(
        self,
        results: list[InstanceResult],
        method: str,
        tier: str,
    ) -> AggregateResult:
        """Compute aggregate statistics from per-instance results."""
        n = len(results)
        if n == 0:
            return AggregateResult(method=method, tier=tier)

        valid = [r for r in results if not r.error]
        n_valid = len(valid) or 1

        coverage = sum(r.covered for r in valid) / n_valid
        exact_match = sum(r.exact for r in valid) / n_valid
        mae = float(np.mean([abs(r.estimated_ub - r.true_radius)
                             for r in valid])) if valid else 0.0
        widths = [r.estimated_ub - r.estimated_lb for r in valid]
        times = [r.wall_time_s for r in valid]

        return AggregateResult(
            method=method,
            tier=tier,
            n_instances=n,
            coverage=coverage,
            exact_match=exact_match,
            mae=mae,
            mean_width=float(np.mean(widths)) if widths else 0.0,
            mean_time_s=float(np.mean(times)) if times else 0.0,
            median_time_s=float(np.median(times)) if times else 0.0,
            p95_time_s=float(np.percentile(times, 95)) if times else 0.0,
            total_time_s=sum(times),
            n_errors=n - len(valid),
        )

    def _run_comparisons(self) -> None:
        """Run paired permutation tests between all method pairs."""
        self.comparison_tests = {}
        methods = list({r.method for r in self.instance_results})
        if len(methods) < 2:
            return

        # Group results by (tier, instance_id) for pairing
        by_method_tier: dict[tuple[str, str], list[InstanceResult]] = defaultdict(list)
        for r in self.instance_results:
            by_method_tier[(r.method, r.tier)].append(r)

        tiers = list({r.tier for r in self.instance_results})
        for tier in tiers:
            for i, m_a in enumerate(methods):
                for m_b in methods[i + 1:]:
                    res_a = by_method_tier.get((m_a, tier), [])
                    res_b = by_method_tier.get((m_b, tier), [])
                    n_common = min(len(res_a), len(res_b))
                    if n_common < 5:
                        continue

                    mae_a = [abs(r.estimated_ub - r.true_radius)
                             for r in res_a[:n_common]]
                    mae_b = [abs(r.estimated_ub - r.true_radius)
                             for r in res_b[:n_common]]
                    test_result = paired_permutation_test(mae_a, mae_b)
                    key = f"{tier}/{m_a}_vs_{m_b}"
                    self.comparison_tests[key] = test_result

    # ------------------------------------------------------------------
    # Baseline methods
    # ------------------------------------------------------------------

    @staticmethod
    def brute_force_baseline(
        adj: AdjacencyMatrix,
        data: pd.DataFrame,
        treatment: int,
        outcome: int,
        max_k: int,
    ) -> RobustnessRadius:
        """Brute-force baseline: enumerate all DAGs up to edit distance max_k.

        Finds the minimum distance at which the treatment-outcome path
        changes.  Suitable only for small graphs.

        Parameters
        ----------
        adj, data, treatment, outcome, max_k : ...

        Returns
        -------
        RobustnessRadius
        """
        from causalcert.dag.edit import apply_edit, all_single_edits
        from causalcert.dag.validation import is_dag
        from collections import deque

        adj = np.asarray(adj, dtype=np.int8)
        n = adj.shape[0]
        has_path = _has_directed_path(adj, treatment, outcome)

        seen: set[bytes] = {adj.tobytes()}
        queue: deque[tuple[np.ndarray, int]] = deque([(adj.copy(), 0)])

        while queue:
            cur, depth = queue.popleft()
            if depth >= max_k:
                continue
            for edit in all_single_edits(cur):
                new_adj = apply_edit(cur, edit)
                key = new_adj.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                cur_path = _has_directed_path(new_adj, treatment, outcome)
                if cur_path != has_path:
                    return RobustnessRadius(
                        lower_bound=depth + 1,
                        upper_bound=depth + 1,
                        witness_edits=(edit,),
                        certified=True,
                    )
                if depth + 1 < max_k:
                    queue.append((new_adj, depth + 1))

        return RobustnessRadius(
            lower_bound=max_k,
            upper_bound=max_k,
            certified=False,
        )

    @staticmethod
    def random_baseline(
        adj: AdjacencyMatrix,
        data: pd.DataFrame,
        treatment: int,
        outcome: int,
        max_k: int,
    ) -> RobustnessRadius:
        """Random baseline: returns a random radius in [1, max_k].

        Parameters
        ----------
        adj, data, treatment, outcome, max_k : ...

        Returns
        -------
        RobustnessRadius
        """
        rng = np.random.RandomState(42)
        r = rng.randint(1, max_k + 1)
        return RobustnessRadius(lower_bound=r, upper_bound=r, certified=False)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize results to JSON.

        Parameters
        ----------
        path : str | Path | None
            If provided, write to file.

        Returns
        -------
        str
            JSON string.
        """
        payload: dict[str, Any] = {
            "aggregate": [_dataclass_to_dict(a) for a in self.aggregate_results],
            "instances": [_dataclass_to_dict(r) for r in self.instance_results],
            "comparisons": self.comparison_tests,
        }
        text = json.dumps(payload, indent=2, default=str)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_csv(self, path: str | Path | None = None) -> str:
        """Serialize aggregate results to CSV.

        Parameters
        ----------
        path : str | Path | None

        Returns
        -------
        str
            CSV string.
        """
        if not self.aggregate_results:
            return ""
        fields = list(_dataclass_to_dict(self.aggregate_results[0]).keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        for agg in self.aggregate_results:
            writer.writerow(_dataclass_to_dict(agg))
        text = buf.getvalue()
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_dataframe(self) -> pd.DataFrame:
        """Return aggregate results as a DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        rows = [_dataclass_to_dict(a) for a in self.aggregate_results]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return a human-readable summary string.

        Returns
        -------
        str
        """
        lines = ["=== Benchmark Summary ==="]
        for agg in self.aggregate_results:
            lines.append(
                f"  {agg.tier}/{agg.method}: "
                f"coverage={agg.coverage:.2%}, "
                f"exact={agg.exact_match:.2%}, "
                f"MAE={agg.mae:.2f}, "
                f"time={agg.mean_time_s:.3f}s"
            )
        if self.comparison_tests:
            lines.append("--- Comparisons ---")
            for key, val in self.comparison_tests.items():
                lines.append(
                    f"  {key}: diff={val['observed_diff']:.4f}, "
                    f"p={val['p_value']:.4f}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_directed_path(adj: np.ndarray, src: int, dst: int) -> bool:
    """BFS check for a directed path from *src* to *dst*."""
    from collections import deque

    n = adj.shape[0]
    visited: set[int] = {src}
    q: deque[int] = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            return True
        for v in np.nonzero(adj[u])[0]:
            v = int(v)
            if v not in visited:
                visited.add(v)
                q.append(v)
    return src == dst


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a plain dict."""
    return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
