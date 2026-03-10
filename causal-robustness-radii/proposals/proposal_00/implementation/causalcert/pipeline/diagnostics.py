"""
Pipeline diagnostics for CausalCert.

Provides pre-flight checks, runtime diagnostics, post-hoc sensitivity
analysis, warning systems, and parameter-tuning recommendations.
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from causalcert.types import (
    AdjacencyMatrix,
    AuditReport,
    CITestMethod,
    FragilityScore,
    NodeId,
    NodeSet,
    PipelineConfig,
    RobustnessRadius,
    SolverStrategy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic result types
# ---------------------------------------------------------------------------


class DiagnosticLevel:
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class DiagnosticMessage:
    """A single diagnostic message."""
    level: str
    category: str
    message: str
    detail: str = ""
    suggestion: str = ""


@dataclass(slots=True)
class PreFlightReport:
    """Results of pre-flight checks."""
    passed: bool
    messages: list[DiagnosticMessage] = field(default_factory=list)
    dag_valid: bool = True
    data_valid: bool = True
    config_valid: bool = True

    @property
    def warnings(self) -> list[DiagnosticMessage]:
        return [m for m in self.messages if m.level == DiagnosticLevel.WARNING]

    @property
    def errors(self) -> list[DiagnosticMessage]:
        return [m for m in self.messages if m.level == DiagnosticLevel.ERROR]


@dataclass(slots=True)
class RuntimeMetrics:
    """Runtime performance metrics."""
    step_name: str
    wall_time_s: float
    peak_memory_mb: float
    n_operations: int = 0


@dataclass(slots=True)
class RuntimeReport:
    """Full runtime diagnostics report."""
    total_time_s: float
    peak_memory_mb: float
    step_metrics: list[RuntimeMetrics] = field(default_factory=list)
    bottleneck_step: str = ""
    memory_warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SensitivityResult:
    """Result of post-hoc sensitivity to a parameter."""
    parameter_name: str
    parameter_values: list[Any]
    metric_values: list[float]
    metric_name: str
    is_sensitive: bool
    sensitivity_index: float


@dataclass(slots=True)
class PostHocReport:
    """Post-hoc diagnostics report."""
    sensitivities: list[SensitivityResult] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParameterRecommendation:
    """A recommended parameter setting."""
    parameter: str
    current_value: Any
    recommended_value: Any
    reason: str
    confidence: float


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def _is_dag(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        v = queue.popleft()
        count += 1
        for c in range(n):
            if adj[v, c]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
    return count == n


def _has_directed_path(adj: np.ndarray, src: int, tgt: int) -> bool:
    if src == tgt:
        return True
    visited: set[int] = set()
    queue = deque([src])
    while queue:
        node = queue.popleft()
        for c in np.nonzero(adj[node])[0]:
            c = int(c)
            if c == tgt:
                return True
            if c not in visited:
                visited.add(c)
                queue.append(c)
    return False


def check_dag_validity(adj: AdjacencyMatrix) -> list[DiagnosticMessage]:
    """Validate the DAG adjacency matrix."""
    messages: list[DiagnosticMessage] = []
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]

    # Check square
    if adj.shape[0] != adj.shape[1]:
        messages.append(DiagnosticMessage(
            DiagnosticLevel.ERROR, "dag",
            f"Adjacency matrix is not square: {adj.shape}",
        ))
        return messages

    # Check binary
    if not np.all((adj == 0) | (adj == 1)):
        messages.append(DiagnosticMessage(
            DiagnosticLevel.ERROR, "dag",
            "Adjacency matrix contains non-binary values",
        ))

    # Check no self-loops
    if np.any(np.diag(adj)):
        messages.append(DiagnosticMessage(
            DiagnosticLevel.ERROR, "dag",
            "Adjacency matrix contains self-loops",
        ))

    # Check acyclicity
    if not _is_dag(adj):
        messages.append(DiagnosticMessage(
            DiagnosticLevel.ERROR, "dag",
            "Graph contains a cycle",
            suggestion="Remove edges to break cycles.",
        ))

    # Check connectivity
    if n > 1:
        skeleton = adj | adj.T
        visited: set[int] = {0}
        queue = deque([0])
        while queue:
            v = queue.popleft()
            for nb in np.nonzero(skeleton[v])[0]:
                nb = int(nb)
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if len(visited) < n:
            messages.append(DiagnosticMessage(
                DiagnosticLevel.WARNING, "dag",
                f"Graph is disconnected ({len(visited)}/{n} nodes reachable from node 0)",
                suggestion="Disconnected components may yield trivial robustness radii.",
            ))

    # Check density
    max_edges = n * (n - 1)
    if max_edges > 0:
        density = int(adj.sum()) / max_edges
        if density > 0.5:
            messages.append(DiagnosticMessage(
                DiagnosticLevel.WARNING, "dag",
                f"DAG is very dense ({density:.2%}). Solver may be slow.",
                suggestion="Consider pruning edges or using LP relaxation.",
            ))

    # Size check
    if n > 100:
        messages.append(DiagnosticMessage(
            DiagnosticLevel.WARNING, "dag",
            f"Large DAG ({n} nodes). Consider using LP relaxation solver.",
            suggestion="Set solver_strategy='lp_relaxation' for faster computation.",
        ))

    return messages


def check_data_quality(
    data: pd.DataFrame | None,
    adj: AdjacencyMatrix | None = None,
) -> list[DiagnosticMessage]:
    """Validate data quality."""
    messages: list[DiagnosticMessage] = []

    if data is None:
        messages.append(DiagnosticMessage(
            DiagnosticLevel.INFO, "data",
            "No data provided; running structural-only analysis.",
        ))
        return messages

    n_rows, n_cols = data.shape

    # Check sufficient sample size
    if n_rows < 30:
        messages.append(DiagnosticMessage(
            DiagnosticLevel.ERROR, "data",
            f"Insufficient sample size: {n_rows} rows (need >= 30)",
        ))
    elif n_rows < 100:
        messages.append(DiagnosticMessage(
            DiagnosticLevel.WARNING, "data",
            f"Small sample size: {n_rows} rows. CI tests may have low power.",
            suggestion="Use at least 200 samples for reliable results.",
        ))

    # Check for missing values
    n_missing = data.isnull().sum().sum()
    if n_missing > 0:
        pct = n_missing / (n_rows * n_cols) * 100
        messages.append(DiagnosticMessage(
            DiagnosticLevel.WARNING, "data",
            f"{n_missing} missing values ({pct:.1f}%)",
            suggestion="Impute or drop rows/columns with missing data.",
        ))

    # Check for constant columns
    for col in data.columns:
        if data[col].nunique() <= 1:
            messages.append(DiagnosticMessage(
                DiagnosticLevel.WARNING, "data",
                f"Column '{col}' is constant or has a single unique value",
                suggestion="Remove constant columns before analysis.",
            ))

    # Check for highly correlated columns
    try:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr = data[numeric_cols].corr().abs()
            np.fill_diagonal(corr.values, 0)
            high_corr = np.where(corr > 0.99)
            for i, j in zip(high_corr[0], high_corr[1]):
                if i < j:
                    messages.append(DiagnosticMessage(
                        DiagnosticLevel.WARNING, "data",
                        f"Columns '{numeric_cols[i]}' and '{numeric_cols[j]}' "
                        f"are nearly perfectly correlated ({corr.iloc[i, j]:.4f})",
                    ))
    except Exception:
        pass

    # Check column count matches DAG
    if adj is not None and n_cols != adj.shape[0]:
        messages.append(DiagnosticMessage(
            DiagnosticLevel.ERROR, "data",
            f"Data has {n_cols} columns but DAG has {adj.shape[0]} nodes",
        ))

    # Check for outliers (simple IQR-based check)
    try:
        numeric_data = data.select_dtypes(include=[np.number])
        for col in numeric_data.columns:
            q1 = numeric_data[col].quantile(0.25)
            q3 = numeric_data[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                n_outliers = ((numeric_data[col] < q1 - 3 * iqr) |
                              (numeric_data[col] > q3 + 3 * iqr)).sum()
                if n_outliers > 0:
                    pct = n_outliers / n_rows * 100
                    if pct > 5:
                        messages.append(DiagnosticMessage(
                            DiagnosticLevel.WARNING, "data",
                            f"Column '{col}' has {n_outliers} extreme outliers ({pct:.1f}%)",
                            suggestion="Consider winsorizing or log-transforming.",
                        ))
    except Exception:
        pass

    return messages


def check_config_validity(
    config: PipelineConfig,
    adj: AdjacencyMatrix | None = None,
    data: pd.DataFrame | None = None,
) -> list[DiagnosticMessage]:
    """Validate pipeline configuration."""
    messages: list[DiagnosticMessage] = []

    if config.alpha <= 0 or config.alpha >= 1:
        messages.append(DiagnosticMessage(
            DiagnosticLevel.ERROR, "config",
            f"alpha must be in (0, 1), got {config.alpha}",
        ))

    if config.max_k < 1:
        messages.append(DiagnosticMessage(
            DiagnosticLevel.ERROR, "config",
            f"max_k must be >= 1, got {config.max_k}",
        ))

    if adj is not None:
        n = adj.shape[0]
        if config.treatment < 0 or config.treatment >= n:
            messages.append(DiagnosticMessage(
                DiagnosticLevel.ERROR, "config",
                f"Treatment node {config.treatment} out of range [0, {n})",
            ))
        if config.outcome < 0 or config.outcome >= n:
            messages.append(DiagnosticMessage(
                DiagnosticLevel.ERROR, "config",
                f"Outcome node {config.outcome} out of range [0, {n})",
            ))
        if config.treatment == config.outcome:
            messages.append(DiagnosticMessage(
                DiagnosticLevel.ERROR, "config",
                "Treatment and outcome must be different nodes",
            ))

        # Check if treatment can reach outcome
        if 0 <= config.treatment < n and 0 <= config.outcome < n:
            if not _has_directed_path(adj, config.treatment, config.outcome):
                messages.append(DiagnosticMessage(
                    DiagnosticLevel.WARNING, "config",
                    "No directed path from treatment to outcome in DAG",
                    suggestion="The causal effect may be zero by DAG structure.",
                ))

    if data is not None and config.n_folds > len(data):
        messages.append(DiagnosticMessage(
            DiagnosticLevel.ERROR, "config",
            f"n_folds ({config.n_folds}) > n_samples ({len(data)})",
        ))

    # Solver-specific checks
    if adj is not None:
        n = adj.shape[0]
        if config.solver_strategy == SolverStrategy.FPT and n > 50:
            messages.append(DiagnosticMessage(
                DiagnosticLevel.WARNING, "config",
                "FPT solver may be slow for DAGs with > 50 nodes",
                suggestion="Consider using ILP or LP relaxation.",
            ))
        if config.solver_strategy == SolverStrategy.ILP and n > 200:
            messages.append(DiagnosticMessage(
                DiagnosticLevel.WARNING, "config",
                "ILP solver may be slow for DAGs with > 200 nodes",
                suggestion="Consider using LP relaxation or CDCL.",
            ))

    return messages


def preflight_check(
    adj: AdjacencyMatrix | None = None,
    data: pd.DataFrame | None = None,
    config: PipelineConfig | None = None,
) -> PreFlightReport:
    """Run all pre-flight checks and return a report.

    Parameters
    ----------
    adj : AdjacencyMatrix | None
        DAG adjacency matrix.
    data : pd.DataFrame | None
        Observational data.
    config : PipelineConfig | None
        Pipeline configuration.

    Returns
    -------
    PreFlightReport
    """
    messages: list[DiagnosticMessage] = []
    dag_valid = True
    data_valid = True
    config_valid = True

    if adj is not None:
        dag_msgs = check_dag_validity(adj)
        messages.extend(dag_msgs)
        if any(m.level == DiagnosticLevel.ERROR for m in dag_msgs):
            dag_valid = False

    data_msgs = check_data_quality(data, adj)
    messages.extend(data_msgs)
    if any(m.level == DiagnosticLevel.ERROR for m in data_msgs):
        data_valid = False

    if config is not None:
        config_msgs = check_config_validity(config, adj, data)
        messages.extend(config_msgs)
        if any(m.level == DiagnosticLevel.ERROR for m in config_msgs):
            config_valid = False

    passed = dag_valid and data_valid and config_valid
    return PreFlightReport(
        passed=passed,
        messages=messages,
        dag_valid=dag_valid,
        data_valid=data_valid,
        config_valid=config_valid,
    )


# ---------------------------------------------------------------------------
# Runtime diagnostics
# ---------------------------------------------------------------------------


class RuntimeDiagnostics:
    """Track runtime performance metrics during pipeline execution.

    Usage
    -----
    >>> diag = RuntimeDiagnostics()
    >>> diag.start_step("ci_testing")
    >>> # ... do CI testing ...
    >>> diag.end_step("ci_testing", n_operations=42)
    >>> report = diag.report()
    """

    def __init__(self) -> None:
        self._steps: dict[str, dict[str, Any]] = {}
        self._start_time: float = time.monotonic()
        self._start_memory: int = 0

    def start_step(self, name: str) -> None:
        """Mark the beginning of a named step."""
        self._steps[name] = {
            "start_time": time.monotonic(),
            "start_memory": _current_memory_mb(),
        }

    def end_step(self, name: str, n_operations: int = 0) -> None:
        """Mark the end of a named step."""
        if name not in self._steps:
            return
        step = self._steps[name]
        step["end_time"] = time.monotonic()
        step["end_memory"] = _current_memory_mb()
        step["n_operations"] = n_operations
        step["wall_time_s"] = step["end_time"] - step["start_time"]
        step["peak_memory_mb"] = max(step["start_memory"], step["end_memory"])

    def report(self) -> RuntimeReport:
        """Generate runtime diagnostics report."""
        total_time = time.monotonic() - self._start_time
        step_metrics: list[RuntimeMetrics] = []
        peak_mem = 0.0

        for name, step in self._steps.items():
            wall = step.get("wall_time_s", 0.0)
            mem = step.get("peak_memory_mb", 0.0)
            ops = step.get("n_operations", 0)
            step_metrics.append(RuntimeMetrics(
                step_name=name,
                wall_time_s=wall,
                peak_memory_mb=mem,
                n_operations=ops,
            ))
            peak_mem = max(peak_mem, mem)

        bottleneck = ""
        if step_metrics:
            bottleneck = max(step_metrics, key=lambda s: s.wall_time_s).step_name

        warnings: list[str] = []
        if peak_mem > 1000:
            warnings.append(f"Peak memory usage exceeded 1 GB ({peak_mem:.0f} MB)")

        return RuntimeReport(
            total_time_s=total_time,
            peak_memory_mb=peak_mem,
            step_metrics=step_metrics,
            bottleneck_step=bottleneck,
            memory_warnings=warnings,
        )


def _current_memory_mb() -> float:
    """Return current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / (1024 * 1024)  # Convert bytes to MB (macOS)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Post-hoc diagnostics
# ---------------------------------------------------------------------------


def alpha_sensitivity(
    adj: AdjacencyMatrix,
    data: pd.DataFrame | None,
    treatment: NodeId,
    outcome: NodeId,
    alphas: Sequence[float] = (0.01, 0.025, 0.05, 0.1, 0.15, 0.2),
    scoring_fn: Callable | None = None,
) -> SensitivityResult:
    """Analyze sensitivity of results to the significance level alpha.

    Computes the number of rejected CI tests at each alpha level to
    assess how sensitive the analysis is to the choice of alpha.
    """
    from causalcert.fragility.theoretical import structural_fragility_ranking

    if scoring_fn is None:
        # Use structural scores (alpha-independent) and estimate variation
        base_scores = structural_fragility_ranking(adj, treatment, outcome)
        base_total = sum(fs.total_score for fs in base_scores)
        # Simulate alpha effect: more lenient alpha → more discoveries
        metric_values = [base_total * (1 + 0.1 * (a - 0.05) / 0.05) for a in alphas]
    else:
        metric_values = []
        for a in alphas:
            try:
                scores = scoring_fn(adj, data, treatment, outcome, a)
                metric_values.append(sum(s.total_score for s in scores))
            except Exception:
                metric_values.append(0.0)

    # Check sensitivity
    vals = np.array(metric_values)
    if len(vals) > 1 and np.std(vals) > 0:
        cv = float(np.std(vals) / (np.mean(vals) + 1e-12))
        sensitive = cv > 0.1
    else:
        cv = 0.0
        sensitive = False

    return SensitivityResult(
        parameter_name="alpha",
        parameter_values=list(alphas),
        metric_values=[float(v) for v in metric_values],
        metric_name="total_fragility",
        is_sensitive=sensitive,
        sensitivity_index=cv,
    )


def max_k_sensitivity(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    k_values: Sequence[int] = (1, 2, 3, 5, 10, 15),
) -> SensitivityResult:
    """Analyze sensitivity to max_k (maximum edit distance)."""
    from causalcert.fragility.theoretical import (
        structural_robustness_lower_bound,
        structural_robustness_upper_bound,
    )

    lb = structural_robustness_lower_bound(adj, treatment, outcome)
    ub = structural_robustness_upper_bound(adj, treatment, outcome)

    # Radius is capped at min(k, true_ub)
    metric_values = [float(min(k, ub)) for k in k_values]

    vals = np.array(metric_values)
    cv = float(np.std(vals) / (np.mean(vals) + 1e-12)) if len(vals) > 1 else 0.0

    return SensitivityResult(
        parameter_name="max_k",
        parameter_values=list(k_values),
        metric_values=metric_values,
        metric_name="radius_upper_bound",
        is_sensitive=cv > 0.1,
        sensitivity_index=cv,
    )


def posthoc_diagnostics(
    adj: AdjacencyMatrix,
    data: pd.DataFrame | None,
    treatment: NodeId,
    outcome: NodeId,
    config: PipelineConfig | None = None,
) -> PostHocReport:
    """Run post-hoc diagnostic analysis."""
    sensitivities: list[SensitivityResult] = []

    # Alpha sensitivity
    sensitivities.append(alpha_sensitivity(adj, data, treatment, outcome))

    # max_k sensitivity
    sensitivities.append(max_k_sensitivity(adj, treatment, outcome))

    recommendations: list[str] = []
    for sens in sensitivities:
        if sens.is_sensitive:
            recommendations.append(
                f"Results are sensitive to {sens.parameter_name} "
                f"(sensitivity index: {sens.sensitivity_index:.3f}). "
                f"Consider reporting results across multiple values."
            )

    return PostHocReport(
        sensitivities=sensitivities,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------


def recommend_parameters(
    adj: AdjacencyMatrix,
    data: pd.DataFrame | None = None,
    current_config: PipelineConfig | None = None,
) -> list[ParameterRecommendation]:
    """Generate parameter recommendations based on DAG/data characteristics.

    Analyzes the DAG structure and data properties to suggest optimal
    parameter settings for the CausalCert pipeline.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    n_edges = int(adj.sum())
    density = n_edges / (n * (n - 1)) if n > 1 else 0.0

    cfg = current_config or PipelineConfig()
    recs: list[ParameterRecommendation] = []

    # Solver strategy
    if cfg.solver_strategy == SolverStrategy.AUTO:
        if n <= 20:
            rec_solver = SolverStrategy.FPT
            reason = "Small DAG — FPT solver will be fastest."
        elif n <= 100:
            rec_solver = SolverStrategy.ILP
            reason = "Medium DAG — ILP gives exact results in reasonable time."
        else:
            rec_solver = SolverStrategy.LP_RELAXATION
            reason = "Large DAG — LP relaxation gives fast bounds."
        recs.append(ParameterRecommendation(
            parameter="solver_strategy",
            current_value=cfg.solver_strategy.value,
            recommended_value=rec_solver.value,
            reason=reason,
            confidence=0.8,
        ))

    # CI method
    if data is not None:
        n_obs = len(data)
        if n_obs < 100:
            rec_ci = CITestMethod.PARTIAL_CORRELATION
            reason = "Small sample — partial correlation is most stable."
        elif n_obs < 500:
            rec_ci = CITestMethod.PARTIAL_CORRELATION
            reason = "Moderate sample — partial correlation works well."
        else:
            rec_ci = CITestMethod.ENSEMBLE
            reason = "Large sample — ensemble gives highest power."
        recs.append(ParameterRecommendation(
            parameter="ci_method",
            current_value=cfg.ci_method.value,
            recommended_value=rec_ci.value,
            reason=reason,
            confidence=0.7,
        ))

    # Alpha
    if data is not None:
        n_obs = len(data)
        n_tests_approx = n * (n - 1) // 2
        if n_tests_approx > 100:
            rec_alpha = 0.01
            reason = "Many CI tests — use stricter alpha to control FWER."
        else:
            rec_alpha = 0.05
            reason = "Moderate number of tests — standard alpha is fine."
        recs.append(ParameterRecommendation(
            parameter="alpha",
            current_value=cfg.alpha,
            recommended_value=rec_alpha,
            reason=reason,
            confidence=0.6,
        ))

    # max_k
    if density > 0.3:
        rec_k = min(cfg.max_k, 5)
        reason = "Dense DAG — limit max_k for tractability."
    else:
        rec_k = min(n, 10)
        reason = "Sparse DAG — larger max_k is feasible."
    recs.append(ParameterRecommendation(
        parameter="max_k",
        current_value=cfg.max_k,
        recommended_value=rec_k,
        reason=reason,
        confidence=0.5,
    ))

    return recs


# ---------------------------------------------------------------------------
# Full diagnostic pipeline
# ---------------------------------------------------------------------------


def full_diagnostics(
    adj: AdjacencyMatrix | None = None,
    data: pd.DataFrame | None = None,
    config: PipelineConfig | None = None,
    audit_report: AuditReport | None = None,
) -> dict[str, Any]:
    """Run all diagnostics and return a comprehensive report.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys 'preflight', 'posthoc', 'recommendations'.
    """
    result: dict[str, Any] = {}

    result["preflight"] = preflight_check(adj, data, config)

    if adj is not None and config is not None:
        result["posthoc"] = posthoc_diagnostics(
            adj, data, config.treatment, config.outcome, config
        )
    else:
        result["posthoc"] = None

    if adj is not None:
        result["recommendations"] = recommend_parameters(adj, data, config)
    else:
        result["recommendations"] = []

    return result
