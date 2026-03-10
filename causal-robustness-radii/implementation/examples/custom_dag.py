#!/usr/bin/env python3
"""Demonstrate custom DAG specification and analysis with CausalCert.

This example shows three ways to specify a DAG:

  1. **DOT format** — Graphviz text.
  2. **JSON format** — nodes + edges dictionary.
  3. **Adjacency matrix** — NumPy array.

It then loads (or generates) observational data, runs the pipeline,
and produces an HTML audit report.

Run::

    python examples/custom_dag.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causalcert.dag.conversions import from_dot, from_json, to_dot
from causalcert.dag.graph import CausalDAG
from causalcert.data.loader import load_csv
from causalcert.data.synthetic import generate_linear_gaussian
from causalcert.pipeline.config import PipelineRunConfig
from causalcert.pipeline.orchestrator import CausalCertPipeline
from causalcert.reporting.html_report import to_html_report
from causalcert.types import AuditReport


# =====================================================================
# Section 1 — Specify a DAG via DOT string
# =====================================================================

EXAMPLE_DOT = """\
digraph MedicalTrial {
    Age       -> Treatment;
    Age       -> BloodPressure;
    Age       -> Outcome;
    Treatment -> BloodPressure;
    Treatment -> Outcome;
    BloodPressure -> Outcome;
    Sex       -> Treatment;
    Sex       -> BloodPressure;
}
"""


def dag_from_dot_string(dot: str) -> tuple[np.ndarray, list[str]]:
    """Parse a DOT string into an adjacency matrix and node names."""
    adj, names = from_dot(dot)
    print(f"  DOT → {len(names)} nodes, {int(adj.sum())} edges")
    print(f"  Nodes: {names}")
    return adj, names


# =====================================================================
# Section 2 — Specify a DAG via JSON
# =====================================================================

EXAMPLE_JSON: dict[str, Any] = {
    "nodes": ["Age", "Sex", "Treatment", "BloodPressure", "Outcome"],
    "edges": [
        ["Age", "Treatment"],
        ["Age", "BloodPressure"],
        ["Age", "Outcome"],
        ["Treatment", "BloodPressure"],
        ["Treatment", "Outcome"],
        ["BloodPressure", "Outcome"],
        ["Sex", "Treatment"],
        ["Sex", "BloodPressure"],
    ],
}


def dag_from_json_dict(spec: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Build a DAG from a JSON-style dictionary (nodes + edges)."""
    json_str = json.dumps(spec)
    adj, names = from_json(json_str)
    print(f"  JSON → {len(names)} nodes, {int(adj.sum())} edges")
    print(f"  Nodes: {names}")
    return adj, names


# =====================================================================
# Section 3 — Specify a DAG via adjacency matrix
# =====================================================================

def dag_from_matrix() -> tuple[np.ndarray, list[str]]:
    """Construct a DAG directly from a NumPy adjacency matrix."""
    names = ["Age", "Sex", "Treatment", "BloodPressure", "Outcome"]
    n = len(names)
    adj = np.zeros((n, n), dtype=np.int8)
    adj[0, 2] = 1  # Age → Treatment
    adj[0, 3] = 1  # Age → BloodPressure
    adj[0, 4] = 1  # Age → Outcome
    adj[1, 2] = 1  # Sex → Treatment
    adj[1, 3] = 1  # Sex → BloodPressure
    adj[2, 3] = 1  # Treatment → BloodPressure
    adj[2, 4] = 1  # Treatment → Outcome
    adj[3, 4] = 1  # BloodPressure → Outcome
    print(f"  Matrix → {n} nodes, {int(adj.sum())} edges")
    print(f"  Nodes: {names}")
    return adj, names


# =====================================================================
# Section 4 — Load or generate observational data
# =====================================================================

def load_or_generate_data(
    adj: np.ndarray,
    names: list[str],
    csv_path: str | None = None,
    n_samples: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Load data from CSV or fall back to synthetic generation.

    Parameters
    ----------
    csv_path : If given, load data from this CSV file.  The column names
        must match *names*.
    n_samples : Number of synthetic samples if generating.
    """
    if csv_path is not None:
        path = Path(csv_path)
        if path.exists():
            print(f"\n  Loading data from {csv_path}")
            df = load_csv(csv_path)
            missing = set(names) - set(df.columns)
            if missing:
                raise ValueError(
                    f"CSV is missing columns: {missing}. "
                    f"Expected: {names}"
                )
            return df[names]

    print(f"\n  Generating synthetic data (n={n_samples}, seed={seed})")
    rng = np.random.default_rng(seed)
    weights = adj.astype(np.float64) * rng.uniform(0.3, 0.8, size=adj.shape)
    data = generate_linear_gaussian(
        adj_matrix=adj,
        weights=weights,
        n_samples=n_samples,
        noise_scale=1.0,
        seed=seed,
    )
    return pd.DataFrame(data, columns=names)


# =====================================================================
# Section 5 — Run the CausalCert pipeline
# =====================================================================

def run_analysis(
    adj: np.ndarray,
    df: pd.DataFrame,
    names: list[str],
    treatment_name: str = "Treatment",
    outcome_name: str = "Outcome",
) -> AuditReport:
    """Run the pipeline and print key results."""
    treatment_idx = names.index(treatment_name)
    outcome_idx = names.index(outcome_name)

    config = PipelineRunConfig(
        treatment=treatment_idx,
        outcome=outcome_idx,
        alpha=0.05,
        solver_strategy="auto",
    )
    pipeline = CausalCertPipeline(config)

    t0 = time.perf_counter()
    report = pipeline.run(adj_matrix=adj, data=df)
    elapsed = time.perf_counter() - t0

    print(f"\n  Pipeline completed in {elapsed:.2f} s")
    print(f"  Robustness radius: [{report.radius.lower_bound}, "
          f"{report.radius.upper_bound}]")
    print(f"  Number of fragile edges: "
          f"{sum(1 for fs in report.fragility_ranking if fs.score >= 0.4)}")
    return report


# =====================================================================
# Section 6 — Generate HTML report
# =====================================================================

def generate_report(report: AuditReport, output_path: str | None = None) -> str:
    """Render an HTML audit report and optionally write to disk."""
    html = to_html_report(report)

    if output_path is not None:
        Path(output_path).write_text(html)
        print(f"\n  HTML report written to {output_path}")
    else:
        print(f"\n  HTML report generated ({len(html)} chars).  "
              "Pass --output to write to disk.")
    return html


# =====================================================================
# Section 7 — Comparison of all three input methods
# =====================================================================

def compare_input_methods() -> None:
    """Show that all three DAG-specification methods yield identical results."""
    print("\n" + "=" * 60)
    print("Comparing DAG specification methods")
    print("=" * 60)

    adj_dot, names_dot = dag_from_dot_string(EXAMPLE_DOT)
    adj_json, names_json = dag_from_json_dict(EXAMPLE_JSON)
    adj_mat, names_mat = dag_from_matrix()

    # Normalise node orderings for comparison
    order_json = [names_json.index(n) for n in names_mat]
    adj_json_reordered = adj_json[np.ix_(order_json, order_json)]

    order_dot = [names_dot.index(n) for n in names_mat]
    adj_dot_reordered = adj_dot[np.ix_(order_dot, order_dot)]

    dot_match = np.array_equal(adj_dot_reordered, adj_mat)
    json_match = np.array_equal(adj_json_reordered, adj_mat)

    print(f"\n  DOT  ↔ Matrix : {'✓ identical' if dot_match else '✗ differ'}")
    print(f"  JSON ↔ Matrix : {'✓ identical' if json_match else '✗ differ'}")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print(textwrap.dedent("""\
    ╔══════════════════════════════════════════════════════════╗
    ║    CausalCert — Custom DAG Example                      ║
    ╚══════════════════════════════════════════════════════════╝
    """))

    # --- 1. Build DAG from DOT ---
    print("1. DAG from DOT string:")
    adj, names = dag_from_dot_string(EXAMPLE_DOT)

    # --- 2. Also demonstrate JSON and matrix methods ---
    print("\n2. DAG from JSON dict:")
    dag_from_json_dict(EXAMPLE_JSON)

    print("\n3. DAG from adjacency matrix:")
    dag_from_matrix()

    # --- 3. Compare ---
    compare_input_methods()

    # --- 4. Load / generate data ---
    print("\n4. Data:")
    df = load_or_generate_data(adj, names, csv_path=None, n_samples=1500)
    print(f"  Shape: {df.shape}")

    # --- 5. Run analysis ---
    print("\n5. Running CausalCert pipeline:")
    report = run_analysis(adj, df, names, "Treatment", "Outcome")

    # --- 6. Generate HTML report ---
    print("\n6. Generating HTML report:")
    generate_report(report)

    print("\nDone.")


if __name__ == "__main__":
    main()
