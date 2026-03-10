"""
Batch analysis for CausalCert.

Run CausalCert audits over multiple treatment-outcome pairs, multiple DAGs,
or combinations thereof.  Supports parallel execution and result aggregation.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import pandas as pd

from causalcert.types import (
    AdjacencyMatrix,
    AuditReport,
    CITestMethod,
    FragilityScore,
    NodeId,
    PipelineConfig,
    RobustnessRadius,
    SolverStrategy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch item specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BatchItem:
    """Specification for a single analysis in a batch."""
    dag: AdjacencyMatrix
    treatment: NodeId
    outcome: NodeId
    data: pd.DataFrame | None = None
    label: str = ""
    dag_name: str = ""
    node_names: tuple[str, ...] | None = None


@dataclass(slots=True)
class BatchResult:
    """Result of a single batch item."""
    item: BatchItem
    audit_report: AuditReport | None = None
    error: str | None = None
    run_time_s: float = 0.0
    success: bool = True


@dataclass(slots=True)
class BatchSummary:
    """Aggregate summary of a batch run."""
    n_total: int
    n_success: int
    n_failed: int
    total_time_s: float
    mean_radius_lb: float
    mean_radius_ub: float
    min_radius: int
    max_radius: int
    results: list[BatchResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Single-item runner
# ---------------------------------------------------------------------------


def _run_single_item(
    item: BatchItem,
    config: PipelineConfig,
) -> BatchResult:
    """Run a single batch item and return the result."""
    start = time.monotonic()
    try:
        from causalcert.pipeline.api import CausalCertAnalysis

        analysis = CausalCertAnalysis(
            dag=item.dag,
            data=item.data,
            treatment=item.treatment,
            outcome=item.outcome,
        )
        analysis.set_alpha(config.alpha)
        analysis.set_ci_method(config.ci_method)
        analysis.set_solver(config.solver_strategy)
        analysis.set_max_k(config.max_k)
        analysis.set_seed(config.seed)

        result = analysis.run()
        elapsed = time.monotonic() - start

        return BatchResult(
            item=item,
            audit_report=result.audit_report,
            run_time_s=elapsed,
            success=True,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.warning("Batch item '%s' failed: %s", item.label, exc)
        return BatchResult(
            item=item,
            error=str(exc),
            run_time_s=elapsed,
            success=False,
        )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


class BatchRunner:
    """Run CausalCert on multiple analyses with optional parallelism.

    Parameters
    ----------
    config : PipelineConfig | None
        Default configuration for all items. If ``None``, uses defaults.
    n_jobs : int
        Number of parallel workers. 1 = sequential, >1 = thread pool.
    progress_callback : Callable | None
        Called with ``(completed, total)`` after each item.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        n_jobs: int = 1,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._n_jobs = n_jobs
        self._progress = progress_callback
        self._items: list[BatchItem] = []

    def add_item(self, item: BatchItem) -> BatchRunner:
        """Add a single batch item. Returns self for chaining."""
        self._items.append(item)
        return self

    def add_items(self, items: Sequence[BatchItem]) -> BatchRunner:
        """Add multiple batch items. Returns self for chaining."""
        self._items.extend(items)
        return self

    def add_pair(
        self,
        dag: AdjacencyMatrix,
        treatment: NodeId,
        outcome: NodeId,
        data: pd.DataFrame | None = None,
        label: str = "",
    ) -> BatchRunner:
        """Convenience method to add a treatment-outcome pair."""
        self._items.append(BatchItem(
            dag=dag, treatment=treatment, outcome=outcome,
            data=data, label=label or f"t{treatment}_o{outcome}",
        ))
        return self

    def clear(self) -> None:
        """Remove all items."""
        self._items.clear()

    @property
    def n_items(self) -> int:
        return len(self._items)

    def run(self) -> BatchSummary:
        """Execute all batch items and return a summary.

        If ``n_jobs > 1``, runs items in a thread pool.
        """
        start = time.monotonic()
        results: list[BatchResult] = []

        if self._n_jobs <= 1:
            # Sequential execution
            for idx, item in enumerate(self._items):
                res = _run_single_item(item, self._config)
                results.append(res)
                if self._progress:
                    self._progress(idx + 1, len(self._items))
        else:
            # Parallel execution using threads
            with ThreadPoolExecutor(max_workers=self._n_jobs) as pool:
                futures = {
                    pool.submit(_run_single_item, item, self._config): idx
                    for idx, item in enumerate(self._items)
                }
                completed = 0
                for future in as_completed(futures):
                    results.append(future.result())
                    completed += 1
                    if self._progress:
                        self._progress(completed, len(self._items))

            # Sort by original order
            results.sort(
                key=lambda r: next(
                    i for i, item in enumerate(self._items) if item is r.item
                )
            )

        total_time = time.monotonic() - start
        return self._summarize(results, total_time)

    def _summarize(
        self, results: list[BatchResult], total_time: float
    ) -> BatchSummary:
        """Build a BatchSummary from individual results."""
        n_success = sum(1 for r in results if r.success)
        n_failed = len(results) - n_success

        radii_lb = [
            r.audit_report.radius.lower_bound
            for r in results if r.success and r.audit_report
        ]
        radii_ub = [
            r.audit_report.radius.upper_bound
            for r in results if r.success and r.audit_report
        ]

        return BatchSummary(
            n_total=len(results),
            n_success=n_success,
            n_failed=n_failed,
            total_time_s=total_time,
            mean_radius_lb=float(np.mean(radii_lb)) if radii_lb else 0.0,
            mean_radius_ub=float(np.mean(radii_ub)) if radii_ub else 0.0,
            min_radius=int(min(radii_lb)) if radii_lb else 0,
            max_radius=int(max(radii_ub)) if radii_ub else 0,
            results=results,
        )


# ---------------------------------------------------------------------------
# Convenience: all-pairs analysis
# ---------------------------------------------------------------------------


def all_pairs_analysis(
    dag: AdjacencyMatrix,
    data: pd.DataFrame | None = None,
    config: PipelineConfig | None = None,
    exclude_self: bool = True,
    n_jobs: int = 1,
) -> BatchSummary:
    """Run CausalCert for all treatment-outcome pairs in a DAG.

    Parameters
    ----------
    dag : AdjacencyMatrix
        DAG adjacency matrix.
    data : pd.DataFrame | None
        Optional observational data.
    config : PipelineConfig | None
        Pipeline configuration.
    exclude_self : bool
        If True, exclude self-pairs (treatment == outcome).
    n_jobs : int
        Number of parallel workers.

    Returns
    -------
    BatchSummary
    """
    n = dag.shape[0]
    runner = BatchRunner(config=config, n_jobs=n_jobs)
    for t in range(n):
        for o in range(n):
            if exclude_self and t == o:
                continue
            runner.add_pair(dag, t, o, data, label=f"{t}->{o}")
    return runner.run()


def multi_dag_analysis(
    dags: list[AdjacencyMatrix],
    treatment: NodeId,
    outcome: NodeId,
    data: pd.DataFrame | None = None,
    dag_labels: list[str] | None = None,
    config: PipelineConfig | None = None,
    n_jobs: int = 1,
) -> BatchSummary:
    """Run CausalCert on the same query across multiple DAGs.

    Useful for comparing robustness across different structural assumptions.
    """
    if dag_labels is None:
        dag_labels = [f"dag_{i}" for i in range(len(dags))]

    runner = BatchRunner(config=config, n_jobs=n_jobs)
    for dag, label in zip(dags, dag_labels):
        runner.add_item(BatchItem(
            dag=dag, treatment=treatment, outcome=outcome,
            data=data, label=label, dag_name=label,
        ))
    return runner.run()


# ---------------------------------------------------------------------------
# Result export
# ---------------------------------------------------------------------------


def batch_results_to_dataframe(summary: BatchSummary) -> pd.DataFrame:
    """Convert batch results to a pandas DataFrame."""
    rows = []
    for res in summary.results:
        row: dict[str, Any] = {
            "label": res.item.label,
            "dag_name": res.item.dag_name,
            "treatment": res.item.treatment,
            "outcome": res.item.outcome,
            "success": res.success,
            "run_time_s": res.run_time_s,
        }
        if res.success and res.audit_report:
            report = res.audit_report
            row.update({
                "n_nodes": report.n_nodes,
                "n_edges": report.n_edges,
                "radius_lb": report.radius.lower_bound,
                "radius_ub": report.radius.upper_bound,
                "certified": report.radius.certified,
                "gap": report.radius.gap,
                "solver_time_s": report.radius.solver_time_s,
                "n_fragile": len(report.fragility_ranking),
            })
            if report.baseline_estimate:
                row["ate"] = report.baseline_estimate.ate
                row["ate_se"] = report.baseline_estimate.se
        else:
            row["error"] = res.error
        rows.append(row)
    return pd.DataFrame(rows)


def export_batch_csv(
    summary: BatchSummary,
    path: str | Path,
) -> None:
    """Export batch results to a CSV file."""
    df = batch_results_to_dataframe(summary)
    df.to_csv(path, index=False)
    logger.info("Batch results exported to %s", path)


def batch_summary_report(summary: BatchSummary) -> str:
    """Generate a human-readable batch summary."""
    lines = [
        "=== CausalCert Batch Summary ===",
        f"Total items:    {summary.n_total}",
        f"Successful:     {summary.n_success}",
        f"Failed:         {summary.n_failed}",
        f"Total time:     {summary.total_time_s:.2f}s",
        f"Mean radius LB: {summary.mean_radius_lb:.2f}",
        f"Mean radius UB: {summary.mean_radius_ub:.2f}",
        f"Min radius:     {summary.min_radius}",
        f"Max radius:     {summary.max_radius}",
    ]
    if summary.n_failed > 0:
        lines.append("\nFailed items:")
        for res in summary.results:
            if not res.success:
                lines.append(f"  - {res.item.label}: {res.error}")
    return "\n".join(lines)
