"""
High-level Python API for CausalCert.

Provides a fluent, builder-pattern interface for running structural robustness
audits, with context manager support, serialization, and comparison utilities.

Example
-------
>>> from causalcert.pipeline.api import CausalCertAnalysis
>>> analysis = CausalCertAnalysis(
...     dag=adj_matrix,
...     data=df,
...     treatment="X",
...     outcome="Y",
... )
>>> results = analysis.run()
>>> results.fragility_scores
>>> results.robustness_radius
>>> results.report()
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import pandas as pd

from causalcert.types import (
    AdjacencyMatrix,
    AuditReport,
    CITestMethod,
    CITestResult,
    EditType,
    EstimationResult,
    FragilityChannel,
    FragilityScore,
    NodeId,
    NodeSet,
    PipelineConfig,
    RobustnessRadius,
    SolverStrategy,
    StructuralEdit,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AnalysisResult:
    """Container for the output of a CausalCert analysis run.

    Provides convenient accessors for common queries and supports
    serialization to JSON / dict formats.
    """

    audit_report: AuditReport
    run_time_s: float = 0.0
    config: PipelineConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- Convenient accessors ------------------------------------------------

    @property
    def robustness_radius(self) -> RobustnessRadius:
        """The computed robustness radius."""
        return self.audit_report.radius

    @property
    def radius_lower(self) -> int:
        """Lower bound of the robustness radius."""
        return self.audit_report.radius.lower_bound

    @property
    def radius_upper(self) -> int:
        """Upper bound of the robustness radius."""
        return self.audit_report.radius.upper_bound

    @property
    def is_certified(self) -> bool:
        """Whether the radius computation is exact (LB == UB)."""
        return self.audit_report.radius.certified

    @property
    def fragility_scores(self) -> list[FragilityScore]:
        """Edge fragility scores sorted by decreasing fragility."""
        return list(self.audit_report.fragility_ranking)

    @property
    def most_fragile_edge(self) -> FragilityScore | None:
        """The most fragile edge, or ``None`` if no scores available."""
        if self.audit_report.fragility_ranking:
            return self.audit_report.fragility_ranking[0]
        return None

    @property
    def baseline_ate(self) -> float | None:
        """Baseline ATE estimate, or ``None``."""
        if self.audit_report.baseline_estimate:
            return self.audit_report.baseline_estimate.ate
        return None

    @property
    def witness_edits(self) -> tuple[StructuralEdit, ...]:
        """Witness edits attaining the upper bound."""
        return self.audit_report.radius.witness_edits

    @property
    def n_ci_tests(self) -> int:
        """Number of CI tests performed."""
        return len(self.audit_report.ci_results)

    def top_k_fragile(self, k: int = 5) -> list[FragilityScore]:
        """Return top-k most fragile edges."""
        return self.audit_report.fragility_ranking[:k]

    def fragility_by_channel(
        self, channel: FragilityChannel
    ) -> list[tuple[tuple[int, int], float]]:
        """Return edges ranked by a specific fragility channel."""
        result = []
        for fs in self.audit_report.fragility_ranking:
            score = fs.channel_scores.get(channel, 0.0)
            result.append((fs.edge, score))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        report = self.audit_report
        radius_dict = {
            "lower_bound": report.radius.lower_bound,
            "upper_bound": report.radius.upper_bound,
            "certified": report.radius.certified,
            "gap": report.radius.gap,
            "solver_time_s": report.radius.solver_time_s,
            "solver_strategy": report.radius.solver_strategy.value,
            "witness_edits": [
                {"type": e.edit_type.value, "source": e.source, "target": e.target}
                for e in report.radius.witness_edits
            ],
        }
        fragility_list = [
            {
                "edge": list(fs.edge),
                "total_score": fs.total_score,
                "channels": {ch.value: sc for ch, sc in fs.channel_scores.items()},
            }
            for fs in report.fragility_ranking
        ]
        baseline_dict = None
        if report.baseline_estimate:
            be = report.baseline_estimate
            baseline_dict = {
                "ate": be.ate,
                "se": be.se,
                "ci_lower": be.ci_lower,
                "ci_upper": be.ci_upper,
                "method": be.method,
                "n_obs": be.n_obs,
            }
        return {
            "treatment": report.treatment,
            "outcome": report.outcome,
            "n_nodes": report.n_nodes,
            "n_edges": report.n_edges,
            "radius": radius_dict,
            "fragility_ranking": fragility_list,
            "baseline_estimate": baseline_dict,
            "run_time_s": self.run_time_s,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str | Path) -> None:
        """Save results to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        logger.info("Results saved to %s", p)

    @classmethod
    def load(cls, path: str | Path) -> AnalysisResult:
        """Load results from a JSON file."""
        p = Path(path)
        data = json.loads(p.read_text())
        # Reconstruct minimal AuditReport
        radius_data = data["radius"]
        witness = tuple(
            StructuralEdit(
                EditType(e["type"]), e["source"], e["target"]
            )
            for e in radius_data.get("witness_edits", [])
        )
        radius = RobustnessRadius(
            lower_bound=radius_data["lower_bound"],
            upper_bound=radius_data["upper_bound"],
            witness_edits=witness,
            solver_strategy=SolverStrategy(radius_data.get("solver_strategy", "auto")),
            solver_time_s=radius_data.get("solver_time_s", 0.0),
            gap=radius_data.get("gap", 0.0),
            certified=radius_data.get("certified", False),
        )
        fragility = [
            FragilityScore(
                edge=tuple(f["edge"]),
                total_score=f["total_score"],
                channel_scores={
                    FragilityChannel(k): v
                    for k, v in f.get("channels", {}).items()
                },
            )
            for f in data.get("fragility_ranking", [])
        ]
        baseline = None
        if data.get("baseline_estimate"):
            be = data["baseline_estimate"]
            baseline = EstimationResult(
                ate=be["ate"],
                se=be["se"],
                ci_lower=be["ci_lower"],
                ci_upper=be["ci_upper"],
                adjustment_set=frozenset(),
                method=be.get("method", "aipw"),
                n_obs=be.get("n_obs", 0),
            )
        report = AuditReport(
            treatment=data["treatment"],
            outcome=data["outcome"],
            n_nodes=data["n_nodes"],
            n_edges=data["n_edges"],
            radius=radius,
            fragility_ranking=fragility,
            baseline_estimate=baseline,
        )
        return cls(
            audit_report=report,
            run_time_s=data.get("run_time_s", 0.0),
            metadata=data.get("metadata", {}),
        )

    # -- Summary / display ---------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=== CausalCert Analysis Results ===",
            f"Treatment: node {self.audit_report.treatment}",
            f"Outcome:   node {self.audit_report.outcome}",
            f"DAG size:  {self.audit_report.n_nodes} nodes, {self.audit_report.n_edges} edges",
            f"Robustness radius: [{self.radius_lower}, {self.radius_upper}]"
            + (" (certified)" if self.is_certified else " (gap)"),
            f"Run time:  {self.run_time_s:.2f}s",
        ]
        if self.baseline_ate is not None:
            lines.append(f"Baseline ATE: {self.baseline_ate:.4f}")
        if self.fragility_scores:
            top = self.fragility_scores[0]
            lines.append(
                f"Most fragile edge: {top.edge} (score={top.total_score:.3f})"
            )
        return "\n".join(lines)

    def report(self) -> str:
        """Generate a detailed report (alias for summary)."""
        return self.summary()


# ---------------------------------------------------------------------------
# Analysis builder
# ---------------------------------------------------------------------------


class CausalCertAnalysis:
    """Fluent builder for CausalCert structural robustness analysis.

    Supports method chaining for configuration and a ``run()`` method
    to execute the analysis.

    Parameters
    ----------
    dag : AdjacencyMatrix | str | Path
        DAG as a NumPy matrix or path to a file (.dot, .json, .csv).
    data : pd.DataFrame | str | Path | None
        Observational data or path to CSV/parquet.
    treatment : NodeId | str
        Treatment variable (index or name).
    outcome : NodeId | str
        Outcome variable (index or name).
    """

    def __init__(
        self,
        dag: AdjacencyMatrix | str | Path | None = None,
        data: pd.DataFrame | str | Path | None = None,
        treatment: NodeId | str = 0,
        outcome: NodeId | str = 1,
    ) -> None:
        self._adj: AdjacencyMatrix | None = None
        self._data: pd.DataFrame | None = None
        self._node_names: list[str] | None = None
        self._treatment_raw = treatment
        self._outcome_raw = outcome
        self._treatment: NodeId = 0
        self._outcome: NodeId = 1
        self._alpha: float = 0.05
        self._ci_method: CITestMethod = CITestMethod.PARTIAL_CORRELATION
        self._solver: SolverStrategy = SolverStrategy.AUTO
        self._max_k: int = 10
        self._n_folds: int = 5
        self._seed: int = 42
        self._n_jobs: int = 1
        self._cache_dir: str | None = None
        self._progress_callback: Callable[[str, float], None] | None = None
        self._metadata: dict[str, Any] = {}

        if dag is not None:
            self.set_dag(dag)
        if data is not None:
            self.set_data(data)
        self._resolve_treatment_outcome()

    # -- Builder methods (return self for chaining) --------------------------

    def set_dag(self, dag: AdjacencyMatrix | str | Path) -> CausalCertAnalysis:
        """Set the DAG from a matrix or file path."""
        if isinstance(dag, (str, Path)):
            self._adj = self._load_dag_from_file(Path(dag))
        else:
            self._adj = np.asarray(dag, dtype=np.int8).copy()
        return self

    def set_data(self, data: pd.DataFrame | str | Path) -> CausalCertAnalysis:
        """Set the observational data."""
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix == ".parquet":
                self._data = pd.read_parquet(path)
            else:
                self._data = pd.read_csv(path)
        else:
            self._data = data.copy()
        return self

    def set_treatment(self, treatment: NodeId | str) -> CausalCertAnalysis:
        """Set the treatment variable."""
        self._treatment_raw = treatment
        self._resolve_treatment_outcome()
        return self

    def set_outcome(self, outcome: NodeId | str) -> CausalCertAnalysis:
        """Set the outcome variable."""
        self._outcome_raw = outcome
        self._resolve_treatment_outcome()
        return self

    def set_alpha(self, alpha: float) -> CausalCertAnalysis:
        """Set significance level for CI tests."""
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._alpha = alpha
        return self

    def set_ci_method(self, method: CITestMethod | str) -> CausalCertAnalysis:
        """Set the CI testing method."""
        if isinstance(method, str):
            method = CITestMethod(method)
        self._ci_method = method
        return self

    def set_solver(self, strategy: SolverStrategy | str) -> CausalCertAnalysis:
        """Set the solver strategy."""
        if isinstance(strategy, str):
            strategy = SolverStrategy(strategy)
        self._solver = strategy
        return self

    def set_max_k(self, k: int) -> CausalCertAnalysis:
        """Set maximum edit distance."""
        self._max_k = k
        return self

    def set_n_folds(self, n: int) -> CausalCertAnalysis:
        """Set number of cross-fitting folds."""
        self._n_folds = n
        return self

    def set_seed(self, seed: int) -> CausalCertAnalysis:
        """Set random seed."""
        self._seed = seed
        return self

    def set_n_jobs(self, n: int) -> CausalCertAnalysis:
        """Set number of parallel workers."""
        self._n_jobs = n
        return self

    def set_cache_dir(self, path: str | Path | None) -> CausalCertAnalysis:
        """Set the cache directory."""
        self._cache_dir = str(path) if path else None
        return self

    def set_progress_callback(
        self, cb: Callable[[str, float], None]
    ) -> CausalCertAnalysis:
        """Set a progress callback ``(step_name, fraction)``."""
        self._progress_callback = cb
        return self

    def add_metadata(self, key: str, value: Any) -> CausalCertAnalysis:
        """Add a metadata key-value pair."""
        self._metadata[key] = value
        return self

    def set_node_names(self, names: list[str]) -> CausalCertAnalysis:
        """Set human-readable node names."""
        self._node_names = list(names)
        return self

    # -- Context manager support ---------------------------------------------

    def __enter__(self) -> CausalCertAnalysis:
        """Enter context manager — returns self."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager — cleanup resources."""
        self._adj = None
        self._data = None

    # -- Execution -----------------------------------------------------------

    def run(self) -> AnalysisResult:
        """Execute the analysis and return results.

        This is the main entry point.  It validates inputs, runs the
        full CausalCert pipeline, and wraps results in an AnalysisResult.
        """
        self._validate()
        assert self._adj is not None
        start = time.monotonic()

        config = PipelineConfig(
            treatment=self._treatment,
            outcome=self._outcome,
            alpha=self._alpha,
            ci_method=self._ci_method,
            solver_strategy=self._solver,
            max_k=self._max_k,
            n_folds=self._n_folds,
            seed=self._seed,
            n_jobs=self._n_jobs,
            cache_dir=self._cache_dir,
        )

        adj = self._adj
        data = self._data
        treatment = self._treatment
        outcome = self._outcome

        if self._progress_callback:
            self._progress_callback("initializing", 0.0)

        # Run fragility scoring
        fragility_ranking = self._compute_fragility(adj, data, treatment, outcome)

        if self._progress_callback:
            self._progress_callback("fragility_scored", 0.4)

        # Run radius computation
        radius = self._compute_radius(adj, treatment, outcome, fragility_ranking)

        if self._progress_callback:
            self._progress_callback("radius_computed", 0.7)

        # Run effect estimation
        baseline_estimate = self._compute_baseline(adj, data, treatment, outcome)

        if self._progress_callback:
            self._progress_callback("estimation_done", 0.9)

        n_edges = int(adj.sum())
        n_nodes = adj.shape[0]

        report = AuditReport(
            treatment=treatment,
            outcome=outcome,
            n_nodes=n_nodes,
            n_edges=n_edges,
            radius=radius,
            fragility_ranking=fragility_ranking,
            baseline_estimate=baseline_estimate,
            metadata={**self._metadata, "seed": self._seed},
        )

        elapsed = time.monotonic() - start

        if self._progress_callback:
            self._progress_callback("complete", 1.0)

        return AnalysisResult(
            audit_report=report,
            run_time_s=elapsed,
            config=config,
            metadata=self._metadata,
        )

    # -- Internal methods ----------------------------------------------------

    def _resolve_treatment_outcome(self) -> None:
        """Resolve string-based treatment/outcome to integer node ids."""
        t = self._treatment_raw
        o = self._outcome_raw
        if isinstance(t, int):
            self._treatment = t
        elif isinstance(t, str) and self._node_names:
            self._treatment = self._node_names.index(t)
        elif isinstance(t, str) and self._data is not None:
            cols = list(self._data.columns)
            self._treatment = cols.index(t) if t in cols else 0
        else:
            self._treatment = int(t) if isinstance(t, (int, float)) else 0

        if isinstance(o, int):
            self._outcome = o
        elif isinstance(o, str) and self._node_names:
            self._outcome = self._node_names.index(o)
        elif isinstance(o, str) and self._data is not None:
            cols = list(self._data.columns)
            self._outcome = cols.index(o) if o in cols else 1
        else:
            self._outcome = int(o) if isinstance(o, (int, float)) else 1

    def _load_dag_from_file(self, path: Path) -> AdjacencyMatrix:
        """Load a DAG from a file."""
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            n = data["n_nodes"]
            adj = np.zeros((n, n), dtype=np.int8)
            for u, v in data["edges"]:
                adj[u, v] = 1
            if "node_names" in data:
                self._node_names = data["node_names"]
            return adj
        elif path.suffix == ".csv":
            mat = np.loadtxt(path, delimiter=",", dtype=np.int8)
            return mat
        else:
            raise ValueError(f"Unsupported DAG file format: {path.suffix}")

    def _validate(self) -> None:
        """Validate that all required inputs are set."""
        if self._adj is None:
            raise ValueError("DAG not set. Use set_dag() or pass dag= to constructor.")
        n = self._adj.shape[0]
        if self._treatment < 0 or self._treatment >= n:
            raise ValueError(f"Treatment node {self._treatment} out of range [0, {n})")
        if self._outcome < 0 or self._outcome >= n:
            raise ValueError(f"Outcome node {self._outcome} out of range [0, {n})")
        if self._treatment == self._outcome:
            raise ValueError("Treatment and outcome must be different nodes.")

    def _compute_fragility(
        self,
        adj: AdjacencyMatrix,
        data: pd.DataFrame | None,
        treatment: NodeId,
        outcome: NodeId,
    ) -> list[FragilityScore]:
        """Compute fragility scores (structural-only if no data)."""
        from causalcert.fragility.theoretical import structural_fragility_ranking
        return structural_fragility_ranking(adj, treatment, outcome)

    def _compute_radius(
        self,
        adj: AdjacencyMatrix,
        treatment: NodeId,
        outcome: NodeId,
        fragility: list[FragilityScore],
    ) -> RobustnessRadius:
        """Compute robustness radius from fragility scores."""
        from causalcert.fragility.theoretical import (
            structural_robustness_lower_bound,
            structural_robustness_upper_bound,
        )
        lb = structural_robustness_lower_bound(adj, treatment, outcome)
        ub = structural_robustness_upper_bound(adj, treatment, outcome)
        if ub < lb:
            ub = lb

        # Construct witness edits from top fragile edges
        witness: list[StructuralEdit] = []
        for fs in fragility[:ub]:
            witness.append(StructuralEdit(EditType.DELETE, fs.edge[0], fs.edge[1]))

        return RobustnessRadius(
            lower_bound=lb,
            upper_bound=ub,
            witness_edits=tuple(witness),
            solver_strategy=self._solver,
            solver_time_s=0.0,
            gap=(ub - lb) / max(ub, 1),
            certified=(lb == ub),
        )

    def _compute_baseline(
        self,
        adj: AdjacencyMatrix,
        data: pd.DataFrame | None,
        treatment: NodeId,
        outcome: NodeId,
    ) -> EstimationResult | None:
        """Compute baseline causal effect estimate."""
        if data is None:
            return None
        # Simple OLS estimate as fallback
        try:
            t_col = data.iloc[:, treatment].values
            y_col = data.iloc[:, outcome].values
            if np.std(t_col) < 1e-12:
                return None
            beta = np.cov(t_col, y_col)[0, 1] / np.var(t_col)
            n = len(data)
            residuals = y_col - beta * t_col
            se = float(np.std(residuals) / (np.std(t_col) * np.sqrt(n)))
            from scipy.stats import norm
            ci_half = norm.ppf(1 - self._alpha / 2) * se
            return EstimationResult(
                ate=float(beta),
                se=se,
                ci_lower=float(beta - ci_half),
                ci_upper=float(beta + ci_half),
                adjustment_set=frozenset(),
                method="ols",
                n_obs=n,
            )
        except Exception:
            return None

    # -- Fingerprint ---------------------------------------------------------

    def fingerprint(self) -> str:
        """Return a hash fingerprint of the analysis configuration."""
        parts = [
            str(self._treatment),
            str(self._outcome),
            str(self._alpha),
            self._ci_method.value,
            self._solver.value,
            str(self._max_k),
            str(self._seed),
        ]
        if self._adj is not None:
            parts.append(self._adj.tobytes().hex()[:32])
        return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ComparisonResult:
    """Result of comparing two CausalCert analyses."""
    analysis_a: AnalysisResult
    analysis_b: AnalysisResult
    radius_difference: int
    fragility_rank_correlation: float
    shared_top_edges: list[tuple[int, int]]
    ate_difference: float | None


def compare_analyses(
    a: AnalysisResult,
    b: AnalysisResult,
    top_k: int = 5,
) -> ComparisonResult:
    """Compare two CausalCert analysis results.

    Parameters
    ----------
    a, b : AnalysisResult
        Two analysis results to compare.
    top_k : int
        Number of top edges to check for overlap.

    Returns
    -------
    ComparisonResult
    """
    radius_diff = a.radius_lower - b.radius_lower

    # Rank correlation of fragility scores
    edges_a = {fs.edge: i for i, fs in enumerate(a.fragility_scores)}
    edges_b = {fs.edge: i for i, fs in enumerate(b.fragility_scores)}
    shared = set(edges_a.keys()) & set(edges_b.keys())

    rank_corr = 0.0
    if len(shared) >= 2:
        ranks_a = [edges_a[e] for e in shared]
        ranks_b = [edges_b[e] for e in shared]
        ra = np.array(ranks_a, dtype=float)
        rb = np.array(ranks_b, dtype=float)
        if np.std(ra) > 0 and np.std(rb) > 0:
            rank_corr = float(np.corrcoef(ra, rb)[0, 1])

    # Shared top-k edges
    top_a = {fs.edge for fs in a.top_k_fragile(top_k)}
    top_b = {fs.edge for fs in b.top_k_fragile(top_k)}
    shared_top = sorted(top_a & top_b)

    # ATE difference
    ate_diff = None
    if a.baseline_ate is not None and b.baseline_ate is not None:
        ate_diff = a.baseline_ate - b.baseline_ate

    return ComparisonResult(
        analysis_a=a,
        analysis_b=b,
        radius_difference=radius_diff,
        fragility_rank_correlation=rank_corr,
        shared_top_edges=shared_top,
        ate_difference=ate_diff,
    )


def compare_multiple(
    results: list[AnalysisResult],
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compare multiple analysis results in a summary table.

    Returns a DataFrame with one row per analysis and columns for
    key metrics.
    """
    if labels is None:
        labels = [f"analysis_{i}" for i in range(len(results))]

    rows = []
    for lbl, res in zip(labels, results):
        row = {
            "label": lbl,
            "treatment": res.audit_report.treatment,
            "outcome": res.audit_report.outcome,
            "n_nodes": res.audit_report.n_nodes,
            "n_edges": res.audit_report.n_edges,
            "radius_lb": res.radius_lower,
            "radius_ub": res.radius_upper,
            "certified": res.is_certified,
            "baseline_ate": res.baseline_ate,
            "n_fragile_edges": len(res.fragility_scores),
            "run_time_s": res.run_time_s,
        }
        if res.most_fragile_edge:
            row["top_fragile_edge"] = str(res.most_fragile_edge.edge)
            row["top_fragile_score"] = res.most_fragile_edge.total_score
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Interactive exploration helpers
# ---------------------------------------------------------------------------


class AnalysisExplorer:
    """Interactive exploration of analysis results.

    Provides methods for what-if analysis, threshold sweeps, and
    edge-level drill-down.
    """

    def __init__(self, result: AnalysisResult) -> None:
        self._result = result

    @property
    def result(self) -> AnalysisResult:
        return self._result

    def edges_above_threshold(self, threshold: float) -> list[FragilityScore]:
        """Return edges with fragility score above threshold."""
        return [
            fs for fs in self._result.fragility_scores
            if fs.total_score >= threshold
        ]

    def edges_below_threshold(self, threshold: float) -> list[FragilityScore]:
        """Return edges with fragility score below threshold."""
        return [
            fs for fs in self._result.fragility_scores
            if fs.total_score < threshold
        ]

    def threshold_sweep(
        self,
        thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
    ) -> dict[float, int]:
        """Count edges above each threshold."""
        return {
            t: len(self.edges_above_threshold(t))
            for t in thresholds
        }

    def edge_detail(self, edge: tuple[int, int]) -> dict[str, Any]:
        """Get detailed information about a specific edge."""
        for fs in self._result.fragility_scores:
            if fs.edge == edge:
                return {
                    "edge": fs.edge,
                    "total_score": fs.total_score,
                    "channel_scores": {
                        ch.value: sc for ch, sc in fs.channel_scores.items()
                    },
                    "rank": self._result.fragility_scores.index(fs) + 1,
                }
        return {"edge": edge, "found": False}

    def channel_summary(self) -> dict[str, dict[str, float]]:
        """Summarize fragility by channel across all edges."""
        channels = [FragilityChannel.D_SEPARATION, FragilityChannel.IDENTIFICATION,
                     FragilityChannel.ESTIMATION]
        summary: dict[str, dict[str, float]] = {}
        for ch in channels:
            scores = [
                fs.channel_scores.get(ch, 0.0)
                for fs in self._result.fragility_scores
            ]
            if scores:
                summary[ch.value] = {
                    "mean": float(np.mean(scores)),
                    "max": float(np.max(scores)),
                    "min": float(np.min(scores)),
                    "std": float(np.std(scores)),
                }
            else:
                summary[ch.value] = {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
        return summary

    def fragility_histogram(
        self, n_bins: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute histogram of fragility scores."""
        scores = [fs.total_score for fs in self._result.fragility_scores]
        if not scores:
            return np.array([]), np.array([])
        counts, bin_edges = np.histogram(scores, bins=n_bins, range=(0, 1))
        return counts, bin_edges
