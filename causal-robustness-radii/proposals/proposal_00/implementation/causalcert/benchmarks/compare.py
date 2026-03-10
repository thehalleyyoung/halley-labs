"""Comparison of CausalCert with IDA, SID, E-value, and constraint-based methods.

These are *conceptual* comparisons — they describe what each method measures,
provide synthetic scenarios, and format results side-by-side.  They do **not**
wrap third-party implementations; we focus on CausalCert's outputs and relate
them to the other methods' published behaviour.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ====================================================================
# Data containers
# ====================================================================

@dataclass(frozen=True)
class MethodDescription:
    """Metadata for a causal-robustness or sensitivity method."""
    name: str
    full_name: str
    what_it_measures: str
    input_requirements: str
    output_type: str
    reference: str


@dataclass
class ComparisonResult:
    """Side-by-side result row for one DAG + method pair."""
    dag_name: str
    method: str
    metric_name: str
    metric_value: float | str
    runtime_s: float = 0.0
    notes: str = ""


@dataclass
class ComparisonReport:
    """Full comparison report across multiple methods and DAGs."""
    rows: list[ComparisonResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "DAG": r.dag_name, "Method": r.method,
                "Metric": r.metric_name, "Value": r.metric_value,
                "Runtime(s)": r.runtime_s, "Notes": r.notes,
            }
            for r in self.rows
        ])

    def summary(self) -> str:
        df = self.to_dataframe()
        return df.to_string(index=False)


# ====================================================================
# Method Descriptions
# ====================================================================

IDA = MethodDescription(
    name="IDA",
    full_name="Intervention calculus when the DAG is Absent",
    what_it_measures=(
        "Bounds on the total causal effect consistent with the Markov "
        "equivalence class of estimated CPDAGs."
    ),
    input_requirements="Observational data; optionally a CPDAG.",
    output_type="Set of possible causal effects (multiset of scalars).",
    reference="Maathuis et al. (2009) Annals of Statistics",
)

SID = MethodDescription(
    name="SID",
    full_name="Structural Intervention Distance",
    what_it_measures=(
        "Number of interventional distributions on which two DAGs disagree."
    ),
    input_requirements="Two DAGs (estimated vs. true).",
    output_type="Non-negative integer.",
    reference="Peters & Bühlmann (2015) JMLR",
)

E_VALUE = MethodDescription(
    name="E-value",
    full_name="E-value for unmeasured confounding",
    what_it_measures=(
        "Minimum strength of unmeasured confounding (risk ratio) needed to "
        "explain away an observed association."
    ),
    input_requirements="Point estimate and confidence bound of a risk ratio.",
    output_type="Scalar ≥ 1.",
    reference="VanderWeele & Ding (2017) Annals of Internal Medicine",
)

PC_ALGORITHM = MethodDescription(
    name="PC",
    full_name="PC algorithm",
    what_it_measures="Estimated CPDAG from conditional-independence tests.",
    input_requirements="Observational data, significance level α.",
    output_type="CPDAG (partially directed graph).",
    reference="Spirtes, Glymour, Scheines (2000)",
)

GES_ALGORITHM = MethodDescription(
    name="GES",
    full_name="Greedy Equivalence Search",
    what_it_measures="Score-optimal CPDAG in the BIC-equivalence class.",
    input_requirements="Observational data.",
    output_type="CPDAG.",
    reference="Chickering (2002) JMLR",
)

CAUSALCERT = MethodDescription(
    name="CausalCert",
    full_name="CausalCert — Causal Robustness Radii",
    what_it_measures=(
        "Minimum number of single-edge edits (add/delete/reverse) to a DAG "
        "that overturn a causal conclusion."
    ),
    input_requirements="DAG adjacency matrix, observational data, treatment, outcome.",
    output_type="Integer robustness radius + per-edge fragility scores.",
    reference="(this paper)",
)

ALL_METHODS = [IDA, SID, E_VALUE, PC_ALGORITHM, GES_ALGORITHM, CAUSALCERT]


def list_methods() -> list[MethodDescription]:
    """Return all method descriptions."""
    return list(ALL_METHODS)


def method_comparison_table() -> str:
    """Formatted table comparing the six methods."""
    lines = [
        "=" * 95,
        "Method Comparison",
        "=" * 95,
        f"{'Method':<14s}  {'Measures':<50s}  {'Output':<20s}",
        "-" * 95,
    ]
    for m in ALL_METHODS:
        lines.append(
            f"{m.name:<14s}  {m.what_it_measures[:50]:<50s}  {m.output_type[:20]:<20s}"
        )
    lines.append("")
    return "\n".join(lines)


# ====================================================================
# IDA Comparison (conceptual)
# ====================================================================

def compare_with_ida(
    adj: NDArray[np.int8],
    data: pd.DataFrame,
    treatment: int,
    outcome: int,
    seed: int = 42,
) -> ComparisonResult:
    """Run CausalCert and report how it relates to IDA bounds.

    IDA provides a *set* of possible causal effects consistent with the
    MEC; CausalCert tells you how many edge edits separate the assumed DAG
    from one that yields a qualitatively different conclusion.
    """
    from causalcert.pipeline.config import PipelineRunConfig
    from causalcert.pipeline.orchestrator import CausalCertPipeline

    config = PipelineRunConfig(
        treatment=treatment, outcome=outcome, alpha=0.05, solver_strategy="auto",
    )
    pipeline = CausalCertPipeline(config)
    t0 = time.perf_counter()
    report = pipeline.run(adj_matrix=adj, data=data)
    elapsed = time.perf_counter() - t0

    return ComparisonResult(
        dag_name="custom",
        method="CausalCert_vs_IDA",
        metric_name="robustness_radius",
        metric_value=report.radius.lower_bound,
        runtime_s=elapsed,
        notes=(
            f"IDA would report a set of effects; CausalCert radius "
            f"[{report.radius.lower_bound},{report.radius.upper_bound}] "
            f"measures structural fragility."
        ),
    )


# ====================================================================
# SID Comparison (conceptual)
# ====================================================================

def compare_with_sid(
    adj_true: NDArray[np.int8],
    adj_estimated: NDArray[np.int8],
    data: pd.DataFrame,
    treatment: int,
    outcome: int,
) -> ComparisonResult:
    """Relate CausalCert radius to SID between two DAGs.

    SID counts disagreements on interventional distributions.
    CausalCert's radius tells you the minimum edits to change the
    conclusion — a complementary notion of distance.
    """
    from causalcert.pipeline.config import PipelineRunConfig
    from causalcert.pipeline.orchestrator import CausalCertPipeline

    # Edit distance (Hamming on adjacency matrices)
    hamming = int(np.sum(adj_true != adj_estimated))

    config = PipelineRunConfig(
        treatment=treatment, outcome=outcome, alpha=0.05, solver_strategy="auto",
    )
    pipeline = CausalCertPipeline(config)
    t0 = time.perf_counter()
    report = pipeline.run(adj_matrix=adj_true, data=data)
    elapsed = time.perf_counter() - t0

    return ComparisonResult(
        dag_name="custom",
        method="CausalCert_vs_SID",
        metric_name="robustness_radius",
        metric_value=report.radius.lower_bound,
        runtime_s=elapsed,
        notes=(
            f"Hamming distance (true vs est) = {hamming}; "
            f"CausalCert radius = [{report.radius.lower_bound},"
            f"{report.radius.upper_bound}]."
        ),
    )


# ====================================================================
# E-Value Comparison (conceptual)
# ====================================================================

def compute_e_value(rr: float) -> float:
    """Compute the E-value for a risk ratio *rr* ≥ 1.

    E = RR + sqrt(RR * (RR - 1)).
    """
    if rr < 1:
        rr = 1.0 / rr
    return rr + np.sqrt(rr * (rr - 1.0))


def compare_with_e_value(
    adj: NDArray[np.int8],
    data: pd.DataFrame,
    treatment: int,
    outcome: int,
    observed_rr: float = 2.0,
) -> ComparisonResult:
    """Run CausalCert and compare with the E-value sensitivity measure.

    The E-value quantifies *unmeasured confounding* needed to explain away
    an association.  CausalCert quantifies *structural DAG mis-specification*
    needed to overturn a conclusion.  They address different threat models.
    """
    from causalcert.pipeline.config import PipelineRunConfig
    from causalcert.pipeline.orchestrator import CausalCertPipeline

    e_val = compute_e_value(observed_rr)

    config = PipelineRunConfig(
        treatment=treatment, outcome=outcome, alpha=0.05, solver_strategy="auto",
    )
    pipeline = CausalCertPipeline(config)
    t0 = time.perf_counter()
    report = pipeline.run(adj_matrix=adj, data=data)
    elapsed = time.perf_counter() - t0

    return ComparisonResult(
        dag_name="custom",
        method="CausalCert_vs_E-value",
        metric_name="robustness_radius",
        metric_value=report.radius.lower_bound,
        runtime_s=elapsed,
        notes=(
            f"E-value = {e_val:.2f} (unmeasured confounding); "
            f"CausalCert radius = [{report.radius.lower_bound},"
            f"{report.radius.upper_bound}] (structural)."
        ),
    )


# ====================================================================
# PC / GES Comparison
# ====================================================================

def compare_with_structure_learning(
    adj_assumed: NDArray[np.int8],
    data: pd.DataFrame,
    treatment: int,
    outcome: int,
) -> ComparisonResult:
    """Relate CausalCert to constraint-based structure learning (PC/GES).

    PC/GES learn a DAG from data; CausalCert asks "how fragile is the
    conclusion *given* a DAG?"  If the radius is small, the conclusion is
    sensitive to the choice among Markov-equivalent DAGs.
    """
    from causalcert.pipeline.config import PipelineRunConfig
    from causalcert.pipeline.orchestrator import CausalCertPipeline

    config = PipelineRunConfig(
        treatment=treatment, outcome=outcome, alpha=0.05, solver_strategy="auto",
    )
    pipeline = CausalCertPipeline(config)
    t0 = time.perf_counter()
    report = pipeline.run(adj_matrix=adj_assumed, data=data)
    elapsed = time.perf_counter() - t0

    n_fragile = sum(1 for fs in report.fragility_ranking if fs.score >= 0.4)

    return ComparisonResult(
        dag_name="custom",
        method="CausalCert_vs_PC/GES",
        metric_name="robustness_radius",
        metric_value=report.radius.lower_bound,
        runtime_s=elapsed,
        notes=(
            f"If PC/GES yield a different DAG, CausalCert's radius "
            f"[{report.radius.lower_bound},{report.radius.upper_bound}] "
            f"indicates how many edits separate the conclusions.  "
            f"Fragile edges: {n_fragile}."
        ),
    )


# ====================================================================
# Full Comparison Runner
# ====================================================================

def run_full_comparison(
    adj: NDArray[np.int8],
    data: pd.DataFrame,
    names: list[str],
    treatment: int,
    outcome: int,
    observed_rr: float = 2.0,
) -> ComparisonReport:
    """Run all conceptual comparisons and assemble a report."""
    report = ComparisonReport()

    report.rows.append(compare_with_ida(adj, data, treatment, outcome))
    report.rows.append(compare_with_e_value(adj, data, treatment, outcome, observed_rr))
    report.rows.append(compare_with_structure_learning(adj, data, treatment, outcome))

    return report


def format_comparison_report(report: ComparisonReport) -> str:
    """Return a human-readable string of the comparison report."""
    lines = ["=" * 80, "CausalCert — Cross-Method Comparison", "=" * 80, ""]
    lines.append(method_comparison_table())
    lines.append("")
    lines.append("-" * 80)
    lines.append("Results:")
    lines.append("-" * 80)

    for r in report.rows:
        lines.append(f"  [{r.method}]  {r.metric_name} = {r.metric_value}")
        lines.append(f"    {r.notes}")
        lines.append("")

    return "\n".join(lines)
