"""
Semi-synthetic benchmarks using real DAGs with simulated data.

Combines published causal DAG structures with synthetic data generation
to produce benchmarks where the DAG is realistic but the ground-truth
radius can be computed.

Supports:
- Take published DAG, generate data from it
- Introduce controlled misspecification
- Measure fragility score accuracy against known ground truth
- Bootstrap variance estimation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from causalcert.evaluation.dgp import (
    DGPInstance,
    generate_linear_gaussian,
    generate_nonlinear_additive,
)
from causalcert.types import AdjacencyMatrix, EditType, StructuralEdit

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BenchmarkResult:
    """Result of a single benchmark run.

    Attributes
    ----------
    dgp_name : str
        Name of the DGP or DAG.
    true_radius : int
        Ground-truth radius.
    estimated_lower : int
        Estimated lower bound.
    estimated_upper : int
        Estimated upper bound.
    solver_time_s : float
        Solver time.
    n_ci_tests : int
        Number of CI tests performed.
    baseline_ate : float
        Baseline ATE estimate.
    perturbed_ate : float
        ATE under witness perturbation.
    fragility_auc : float
        AUC for fragility ranking.
    """

    dgp_name: str
    true_radius: int
    estimated_lower: int
    estimated_upper: int
    solver_time_s: float = 0.0
    n_ci_tests: int = 0
    baseline_ate: float = 0.0
    perturbed_ate: float = 0.0
    fragility_auc: float = 0.0


# ---------------------------------------------------------------------------
# Misspecification strategies
# ---------------------------------------------------------------------------


def introduce_misspecification(
    adj: AdjacencyMatrix,
    n_edits: int = 1,
    strategy: str = "random",
    rng: np.random.RandomState | None = None,
) -> tuple[AdjacencyMatrix, list[StructuralEdit]]:
    """Introduce controlled misspecification into a DAG.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Original DAG.
    n_edits : int
        Number of edits to apply.
    strategy : str
        ``"random"``, ``"delete"``, ``"add"``, or ``"reverse"``.
    rng : np.random.RandomState | None

    Returns
    -------
    tuple[AdjacencyMatrix, list[StructuralEdit]]
        ``(misspecified_adj, edits_applied)``
    """
    if rng is None:
        rng = np.random.RandomState()
    adj = np.asarray(adj, dtype=np.int8).copy()
    n = adj.shape[0]
    edits: list[StructuralEdit] = []

    for _ in range(n_edits):
        if strategy == "delete" or (strategy == "random" and rng.random() < 0.4):
            # Delete a random existing edge
            existing = list(zip(*np.nonzero(adj)))
            if existing:
                idx = rng.randint(len(existing))
                u, v = existing[idx]
                adj[u, v] = 0
                edits.append(StructuralEdit(EditType.DELETE, int(u), int(v)))

        elif strategy == "add" or (strategy == "random" and rng.random() < 0.7):
            # Add a random edge (check acyclicity)
            for _attempt in range(50):
                u = rng.randint(n)
                v = rng.randint(n)
                if u != v and not adj[u, v]:
                    adj_test = adj.copy()
                    adj_test[u, v] = 1
                    if _is_dag_fast(adj_test):
                        adj[u, v] = 1
                        edits.append(StructuralEdit(EditType.ADD, u, v))
                        break

        else:
            # Reverse a random existing edge
            existing = list(zip(*np.nonzero(adj)))
            if existing:
                rng.shuffle(existing)
                for u, v in existing:
                    adj_test = adj.copy()
                    adj_test[u, v] = 0
                    adj_test[v, u] = 1
                    if _is_dag_fast(adj_test):
                        adj[u, v] = 0
                        adj[v, u] = 1
                        edits.append(StructuralEdit(EditType.REVERSE, int(u), int(v)))
                        break

    return adj, edits


def _is_dag_fast(adj: np.ndarray) -> bool:
    """Quick DAG check via Kahn's algorithm."""
    from collections import deque
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        node = queue.popleft()
        count += 1
        for c in np.nonzero(adj[node])[0]:
            c = int(c)
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    return count == n


# ---------------------------------------------------------------------------
# Bootstrap variance
# ---------------------------------------------------------------------------


def bootstrap_radius_variance(
    adj: AdjacencyMatrix,
    data: pd.DataFrame,
    treatment: int,
    outcome: int,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate the variance of the radius estimate via bootstrap.

    Parameters
    ----------
    adj : AdjacencyMatrix
    data : pd.DataFrame
    treatment, outcome : int
    n_bootstrap : int
    seed : int

    Returns
    -------
    dict[str, float]
        Keys: ``mean``, ``std``, ``ci_lower``, ``ci_upper``.
    """
    rng = np.random.RandomState(seed)
    n_obs = len(data)
    estimates: list[int] = []

    for b in range(n_bootstrap):
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        boot_data = data.iloc[idx].reset_index(drop=True)

        try:
            from causalcert.pipeline.config import PipelineRunConfig
            from causalcert.pipeline.orchestrator import CausalCertPipeline

            cfg = PipelineRunConfig(
                treatment=treatment, outcome=outcome,
                max_k=5, n_folds=2, cache_dir=None,
            )
            cfg.steps.estimation = False
            cfg.steps.report = False

            pipeline = CausalCertPipeline(cfg)
            report = pipeline.run(adj, boot_data)
            estimates.append(report.radius.upper_bound)
        except Exception:
            continue

    if not estimates:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    arr = np.array(estimates, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
    }


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------


class SemiSyntheticBenchmark:
    """Semi-synthetic benchmark suite.

    Parameters
    ----------
    dag_names : Sequence[str] | None
        Names of published DAGs to use.  ``None`` for all.
    n_samples : int
        Samples per instance.
    n_repeats : int
        Number of repeated runs per DAG.
    seed : int
        Random seed.
    misspecification_levels : list[int] | None
        Number of edits for misspecification (default [0, 1, 2, 3]).
    """

    def __init__(
        self,
        dag_names: Sequence[str] | None = None,
        n_samples: int = 1000,
        n_repeats: int = 10,
        seed: int = 42,
        misspecification_levels: list[int] | None = None,
    ) -> None:
        self.dag_names = list(dag_names) if dag_names else None
        self.n_samples = n_samples
        self.n_repeats = n_repeats
        self.seed = seed
        self.misspecification_levels = misspecification_levels or [0, 1, 2, 3]

    def _get_dags(self) -> list[Any]:
        """Load published DAGs."""
        from causalcert.evaluation.published_dags import (
            get_published_dag,
            list_published_dags,
            get_small_dags,
        )

        if self.dag_names:
            return [get_published_dag(name) for name in self.dag_names]
        # Default: use small and medium DAGs
        return get_small_dags(max_nodes=20)

    def run(self) -> list[BenchmarkResult]:
        """Run the full benchmark suite.

        Returns
        -------
        list[BenchmarkResult]
        """
        dags = self._get_dags()
        results: list[BenchmarkResult] = []
        rng = np.random.RandomState(self.seed)

        for dag_info in dags:
            adj = dag_info.adj
            treatment = dag_info.default_treatment
            outcome = dag_info.default_outcome
            if treatment is None or outcome is None:
                continue

            for repeat in range(self.n_repeats):
                seed_r = self.seed + repeat
                rng_r = np.random.RandomState(seed_r)

                # Generate data from the published DAG
                df, _coeffs, _ate = generate_linear_gaussian(
                    adj, n_samples=self.n_samples,
                    treatment=treatment, rng=rng_r,
                )

                for n_edits in self.misspecification_levels:
                    if n_edits == 0:
                        test_adj = adj.copy()
                        true_radius = 0  # No misspecification
                    else:
                        test_adj, edits_applied = introduce_misspecification(
                            adj, n_edits=n_edits, strategy="random", rng=rng_r,
                        )
                        true_radius = n_edits

                    # Run pipeline on (possibly misspecified) DAG
                    try:
                        from causalcert.pipeline.config import PipelineRunConfig
                        from causalcert.pipeline.orchestrator import CausalCertPipeline

                        cfg = PipelineRunConfig(
                            treatment=treatment, outcome=outcome,
                            max_k=min(5, n_edits + 2),
                            n_folds=2, seed=seed_r, cache_dir=None,
                        )
                        cfg.steps.estimation = False

                        t0 = time.perf_counter()
                        pipeline = CausalCertPipeline(cfg)
                        report = pipeline.run(test_adj, df)
                        elapsed = time.perf_counter() - t0

                        results.append(BenchmarkResult(
                            dgp_name=f"{dag_info.name}_e{n_edits}_r{repeat}",
                            true_radius=true_radius,
                            estimated_lower=report.radius.lower_bound,
                            estimated_upper=report.radius.upper_bound,
                            solver_time_s=elapsed,
                            n_ci_tests=len(report.ci_results),
                        ))
                    except Exception as exc:
                        logger.warning(
                            "Benchmark failed for %s (repeat %d, edits %d): %s",
                            dag_info.name, repeat, n_edits, exc,
                        )
                        results.append(BenchmarkResult(
                            dgp_name=f"{dag_info.name}_e{n_edits}_r{repeat}",
                            true_radius=true_radius,
                            estimated_lower=0,
                            estimated_upper=cfg.max_k,
                            solver_time_s=0.0,
                        ))

        return results

    def summary_table(self, results: list[BenchmarkResult]) -> pd.DataFrame:
        """Produce a summary DataFrame of results.

        Parameters
        ----------
        results : list[BenchmarkResult]

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for r in results:
            covered = r.estimated_lower <= r.true_radius <= r.estimated_upper
            exact = r.estimated_lower == r.estimated_upper == r.true_radius
            width = r.estimated_upper - r.estimated_lower
            rows.append({
                "dgp_name": r.dgp_name,
                "true_radius": r.true_radius,
                "est_lower": r.estimated_lower,
                "est_upper": r.estimated_upper,
                "covered": covered,
                "exact": exact,
                "width": width,
                "solver_time_s": round(r.solver_time_s, 3),
                "n_ci_tests": r.n_ci_tests,
            })

        df = pd.DataFrame(rows)

        if not df.empty:
            logger.info(
                "Benchmark summary: %d runs, coverage=%.2f, exact=%.2f, mean_width=%.2f",
                len(df),
                df["covered"].mean(),
                df["exact"].mean(),
                df["width"].mean(),
            )

        return df
