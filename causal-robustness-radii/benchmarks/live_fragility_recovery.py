#!/usr/bin/env python3
"""Run small, honest end-to-end CausalCert recovery examples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_ROOT = REPO_ROOT / "implementation"
sys.path.insert(0, str(IMPLEMENTATION_ROOT))

from causalcert.data.synthetic import generate_linear_gaussian_with_treatment
from causalcert.pipeline.config import PipelineRunConfig
from causalcert.pipeline.orchestrator import CausalCertPipeline


def run_case(name: str, adj: np.ndarray, treatment: int, outcome: int) -> dict[str, object]:
    try:
        data, _, true_ate = generate_linear_gaussian_with_treatment(
            adj,
            treatment=treatment,
            outcome=outcome,
            n=600,
            true_ate=1.0,
            seed=42,
        )
        config = PipelineRunConfig(
            treatment=treatment,
            outcome=outcome,
            seed=42,
            max_k=3,
            cache_dir=None,
        )
        report = CausalCertPipeline(config).run(adj_matrix=adj, data=data, predicate=None)
        top = report.fragility_ranking[0] if report.fragility_ranking else None
        return {
            "status": "ok",
            "true_ate": true_ate,
            "n_nodes": int(adj.shape[0]),
            "n_edges": int(adj.sum()),
            "radius_lower_bound": int(report.radius.lower_bound),
            "radius_upper_bound": int(report.radius.upper_bound),
            "radius_certified": bool(report.radius.certified),
            "solver_strategy": report.radius.solver_strategy.value,
            "top_fragility_edge": list(top.edge) if top else None,
            "top_fragility_score": round(float(top.total_score), 4) if top else None,
            "ci_tests_run": len(report.ci_results),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def main() -> None:
    cases = {
        "chain3": {
            "adj": np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=int),
            "treatment": 0,
            "outcome": 2,
        }
    }

    results = {
        "script": "benchmarks/live_fragility_recovery.py",
        "seed": 42,
        "cases": {
            name: run_case(name, spec["adj"], spec["treatment"], spec["outcome"])
            for name, spec in cases.items()
        },
    }

    output_path = REPO_ROOT / "benchmarks" / "live_fragility_recovery_results.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
