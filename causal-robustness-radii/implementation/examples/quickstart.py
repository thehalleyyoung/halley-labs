#!/usr/bin/env python3
"""CausalCert Quickstart — end-to-end robustness analysis in under a minute.

This script walks through every step of a CausalCert analysis:

  1. Build a synthetic causal DAG inspired by the *smoking → birthweight* literature.
  2. Generate observational data from a linear-Gaussian structural equation model.
  3. Run the full CausalCert pipeline (CI testing, fragility scoring, radius
     computation, causal-effect estimation).
  4. Print fragility scores, the robustness radius, and which edges are
     load-bearing.

Run::

    python examples/quickstart.py
"""
from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Setup – ensure the package root is on sys.path so the example can be
#    run directly from the repository checkout.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causalcert.dag.graph import CausalDAG
from causalcert.dag.conversions import to_dot
from causalcert.data.synthetic import generate_linear_gaussian
from causalcert.fragility.scorer import FragilityScorerImpl
from causalcert.fragility.ranking import rank_edges, EdgeSeverity
from causalcert.pipeline.orchestrator import CausalCertPipeline
from causalcert.pipeline.config import PipelineRunConfig
from causalcert.types import (
    AuditReport,
    EditType,
    FragilityChannel,
    FragilityScore,
    RobustnessRadius,
    SolverStrategy,
)


# =====================================================================
# Section 1 — Build a Smoking–Birthweight DAG
# =====================================================================

def build_smoking_dag() -> tuple[np.ndarray, list[str]]:
    """Return (adjacency_matrix, node_names) for a smoking-birthweight DAG.

    The graph encodes the following causal story:

        SES  →  Smoking  →  Birthweight
        SES  →  Nutrition →  Birthweight
        SES  →  Prenatal  →  Birthweight
        Smoking → Prenatal (smokers attend fewer visits)
        Genetics → Birthweight
        Genetics → Smoking   (genetic predisposition)

    Node ordering: SES(0), Smoking(1), Nutrition(2), Prenatal(3),
                   Genetics(4), Birthweight(5)
    """
    node_names = [
        "SES",          # 0
        "Smoking",      # 1
        "Nutrition",    # 2
        "Prenatal",     # 3
        "Genetics",     # 4
        "Birthweight",  # 5
    ]
    n = len(node_names)
    adj = np.zeros((n, n), dtype=np.int8)

    # SES → Smoking, Nutrition, Prenatal
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[0, 3] = 1

    # Smoking → Birthweight, Prenatal
    adj[1, 5] = 1
    adj[1, 3] = 1

    # Nutrition → Birthweight
    adj[2, 5] = 1

    # Prenatal → Birthweight
    adj[3, 5] = 1

    # Genetics → Birthweight, Smoking
    adj[4, 5] = 1
    adj[4, 1] = 1

    return adj, node_names


def print_dag_summary(adj: np.ndarray, names: list[str]) -> None:
    """Print a human-readable summary of the DAG."""
    n_edges = int(adj.sum())
    print("=" * 60)
    print("DAG Summary")
    print("=" * 60)
    print(f"  Nodes : {len(names)}")
    print(f"  Edges : {n_edges}")
    print(f"  Names : {', '.join(names)}")
    print()
    print("  Edge list:")
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j]:
                print(f"    {names[i]:>12s}  →  {names[j]}")
    print()


# =====================================================================
# Section 2 — Generate Synthetic Observational Data
# =====================================================================

def generate_data(
    adj: np.ndarray,
    node_names: list[str],
    n_samples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate observational data from a linear-Gaussian SEM.

    Each edge weight is drawn uniformly from [0.3, 0.9] and noise terms
    are standard-normal.  The resulting DataFrame has one column per node.
    """
    df, _W = generate_linear_gaussian(
        adj, n=n_samples, noise_scale=1.0,
        edge_weight_range=(0.3, 0.9), seed=seed,
    )
    df.columns = node_names
    return df


def print_data_summary(df: pd.DataFrame) -> None:
    """Print basic descriptive statistics of the generated data."""
    print("=" * 60)
    print("Data Summary")
    print("=" * 60)
    print(f"  Shape   : {df.shape[0]} samples × {df.shape[1]} variables")
    print()
    desc = df.describe().T[["mean", "std", "min", "max"]]
    print(desc.to_string(float_format="{:.3f}".format))
    print()


# =====================================================================
# Section 3 — Run the CausalCert Pipeline
# =====================================================================

def run_pipeline(
    adj: np.ndarray,
    df: pd.DataFrame,
    treatment: int = 1,
    outcome: int = 5,
) -> AuditReport:
    """Execute the full CausalCert pipeline and return the audit report.

    Parameters
    ----------
    adj : adjacency matrix
    df : observational data
    treatment : column index of the treatment variable (Smoking=1)
    outcome : column index of the outcome variable (Birthweight=5)
    """
    config = PipelineRunConfig(
        treatment=treatment,
        outcome=outcome,
        alpha=0.05,
        solver_strategy=SolverStrategy.AUTO,
    )

    pipeline = CausalCertPipeline(config)
    t0 = time.perf_counter()
    report = pipeline.run(adj_matrix=adj, data=df)
    elapsed = time.perf_counter() - t0

    print("=" * 60)
    print("Pipeline Results")
    print("=" * 60)
    print(f"  Runtime           : {elapsed:.2f} s")
    print(f"  Robustness radius : [{report.radius.lower_bound}, "
          f"{report.radius.upper_bound}]")
    if report.baseline_estimate is not None:
        er = report.baseline_estimate
        print(f"  Estimated ATE     : {er.ate:.4f}  "
              f"(SE {er.se:.4f})")
        print(f"  95 % CI           : [{er.ci_lower:.4f}, {er.ci_upper:.4f}]")
    print()
    return report


# =====================================================================
# Section 4 — Interpret Fragility Scores
# =====================================================================

def print_fragility_ranking(report: AuditReport, names: list[str]) -> None:
    """Pretty-print the fragility ranking from the audit report."""
    print("=" * 60)
    print("Fragility Ranking  (higher = more fragile)")
    print("=" * 60)

    ranking = report.fragility_ranking
    for idx, fs in enumerate(ranking, 1):
        src, dst = fs.edge
        label = f"{names[src]} → {names[dst]}"
        bar = "█" * int(fs.total_score * 40)
        print(f"  {idx:>2d}. {label:<30s}  {fs.total_score:.4f}  {bar}")
    print()


def identify_load_bearing_edges(
    report: AuditReport,
    names: list[str],
    threshold: float = 0.4,
) -> list[tuple[str, str, float]]:
    """Return edges whose fragility score exceeds the threshold.

    These are the *load-bearing* edges: removing or reversing any one of
    them would likely overturn the causal conclusion.
    """
    load_bearing: list[tuple[str, str, float]] = []
    for fs in report.fragility_ranking:
        if fs.total_score >= threshold:
            src, dst = fs.edge
            load_bearing.append((names[src], names[dst], fs.total_score))

    print("=" * 60)
    print(f"Load-Bearing Edges  (fragility ≥ {threshold})")
    print("=" * 60)
    if not load_bearing:
        print("  None — the conclusion is broadly robust.")
    else:
        for src, dst, sc in load_bearing:
            print(f"  • {src} → {dst}  (score {sc:.4f})")
    print()
    return load_bearing


def print_channel_breakdown(report: AuditReport, names: list[str]) -> None:
    """Show per-channel fragility for the top-3 edges."""
    print("=" * 60)
    print("Channel Breakdown  (top-3 edges)")
    print("=" * 60)
    for fs in report.fragility_ranking[:3]:
        src, dst = fs.edge
        label = f"{names[src]} → {names[dst]}"
        print(f"  {label}:")
        for ch, val in fs.channel_scores.items():
            print(f"    {ch.name:<20s}  {val:.4f}")
    print()


# =====================================================================
# Section 5 — DOT Export (optional visualization)
# =====================================================================

def export_dot(adj: np.ndarray, names: list[str], path: str | None = None) -> str:
    """Generate a Graphviz DOT string for the DAG.

    If *path* is given the DOT file is also written to disk so you can
    render it with ``dot -Tpng dag.dot -o dag.png``.
    """
    dot = to_dot(adj, node_names=names)
    if path is not None:
        Path(path).write_text(dot)
        print(f"  DOT file written to {path}")
    return dot


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print(textwrap.dedent("""\
    ╔══════════════════════════════════════════════════════════╗
    ║        CausalCert — Quickstart Example                  ║
    ╚══════════════════════════════════════════════════════════╝
    """))

    # 1. Build DAG
    adj, names = build_smoking_dag()
    print_dag_summary(adj, names)

    # 2. Generate data
    df = generate_data(adj, names, n_samples=2000, seed=42)
    print_data_summary(df)

    # 3. Run pipeline  (treatment=Smoking, outcome=Birthweight)
    report = run_pipeline(adj, df, treatment=1, outcome=5)

    # 4. Interpret results
    print_fragility_ranking(report, names)
    identify_load_bearing_edges(report, names, threshold=0.4)
    print_channel_breakdown(report, names)

    # 5. Export DOT (optional)
    export_dot(adj, names)

    print("Done.  See the README for next steps.")


if __name__ == "__main__":
    main()
