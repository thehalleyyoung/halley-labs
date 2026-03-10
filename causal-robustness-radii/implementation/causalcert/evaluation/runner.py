"""
Evaluation runner — orchestrates benchmark suites.

Runs synthetic DGP, semi-synthetic, and published-DAG evaluations and
collects results into summary tables.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluationConfig:
    """Configuration for an evaluation run.

    Attributes
    ----------
    n_synthetic : int
        Number of synthetic DGP instances.
    n_samples : int
        Samples per instance.
    n_repeats : int
        Repetitions per configuration.
    include_semi_synthetic : bool
        Whether to run semi-synthetic benchmarks.
    include_ablation : bool
        Whether to run ablation studies.
    include_scalability : bool
        Whether to run scalability profiling.
    seed : int
        Random seed.
    max_k : int
        Maximum edit distance.
    n_nodes_range : tuple[int, int]
        Range of DAG sizes for synthetic DGPs.
    density_range : tuple[float, float]
        Range of densities for synthetic DGPs.
    radius_range : tuple[int, int]
        Range of true radii.
    semi_synthetic_dags : list[str] | None
        DAG names for semi-synthetic benchmarks. None for defaults.
    ablation_conditions : list[str] | None
        Ablation conditions to test. None for defaults.
    scalability_nodes : list[int] | None
        Node counts for scalability profiling.
    output_dir : str | None
        Directory for saving result tables.
    """

    n_synthetic: int = 100
    n_samples: int = 1000
    n_repeats: int = 10
    include_semi_synthetic: bool = True
    include_ablation: bool = True
    include_scalability: bool = True
    seed: int = 42
    max_k: int = 5
    n_nodes_range: tuple[int, int] = (8, 20)
    density_range: tuple[float, float] = (0.1, 0.3)
    radius_range: tuple[int, int] = (1, 4)
    semi_synthetic_dags: list[str] | None = None
    ablation_conditions: list[str] | None = None
    scalability_nodes: list[int] | None = None
    output_dir: str | None = None


class EvaluationRunner:
    """Top-level evaluation runner.

    Parameters
    ----------
    config : EvaluationConfig
        Evaluation configuration.
    """

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()
        self._results: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> dict[str, pd.DataFrame]:
        """Run all evaluation suites.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping from suite name to results DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}

        logger.info("=== Starting synthetic DGP evaluation ===")
        t0 = time.perf_counter()
        results["synthetic"] = self.run_synthetic()
        logger.info("Synthetic: %.1fs, %d rows", time.perf_counter() - t0, len(results["synthetic"]))

        if self.config.include_semi_synthetic:
            logger.info("=== Starting semi-synthetic evaluation ===")
            t0 = time.perf_counter()
            results["semi_synthetic"] = self.run_semi_synthetic()
            logger.info("Semi-synthetic: %.1fs, %d rows", time.perf_counter() - t0, len(results["semi_synthetic"]))

        if self.config.include_ablation:
            logger.info("=== Starting ablation studies ===")
            t0 = time.perf_counter()
            results["ablation"] = self.run_ablation()
            logger.info("Ablation: %.1fs, %d rows", time.perf_counter() - t0, len(results["ablation"]))

        if self.config.include_scalability:
            logger.info("=== Starting scalability profiling ===")
            t0 = time.perf_counter()
            results["scalability"] = self.run_scalability()
            logger.info("Scalability: %.1fs, %d rows", time.perf_counter() - t0, len(results["scalability"]))

        self._results = results

        # Save if output_dir is set
        if self.config.output_dir:
            self._save_results(results)

        return results

    def run_synthetic(self) -> pd.DataFrame:
        """Run synthetic DGP evaluation.

        Generates random DAGs with known radii, runs the pipeline, and
        compares estimated vs true radius.

        Returns
        -------
        pd.DataFrame
        """
        from causalcert.evaluation.dgp import SyntheticDGP
        from causalcert.evaluation.metrics import (
            compute_all_metrics,
            coverage_rate,
            exact_match_rate,
            interval_width,
        )
        from causalcert.types import RobustnessRadius

        dgp = SyntheticDGP(seed=self.config.seed)
        instances = dgp.batch_generate(
            n_instances=self.config.n_synthetic,
            n_nodes_range=self.config.n_nodes_range,
            density_range=self.config.density_range,
            radius_range=self.config.radius_range,
            n_samples=self.config.n_samples,
        )

        rows: list[dict[str, Any]] = []
        estimated_radii: list[RobustnessRadius] = []

        for inst in instances:
            try:
                from causalcert.pipeline.config import PipelineRunConfig
                from causalcert.pipeline.orchestrator import CausalCertPipeline

                cfg = PipelineRunConfig(
                    treatment=inst.treatment,
                    outcome=inst.outcome,
                    max_k=self.config.max_k,
                    n_folds=2,
                    seed=self.config.seed,
                    cache_dir=None,
                )
                cfg.steps.estimation = False

                t0 = time.perf_counter()
                pipeline = CausalCertPipeline(cfg)
                report = pipeline.run(inst.adj, inst.data)
                elapsed = time.perf_counter() - t0

                radius = report.radius
                estimated_radii.append(radius)

                covered = radius.lower_bound <= inst.true_radius <= radius.upper_bound
                exact = radius.lower_bound == radius.upper_bound == inst.true_radius

                rows.append({
                    "name": inst.name,
                    "n_nodes": inst.adj.shape[0],
                    "n_edges": int(inst.adj.sum()),
                    "true_radius": inst.true_radius,
                    "est_lower": radius.lower_bound,
                    "est_upper": radius.upper_bound,
                    "covered": covered,
                    "exact": exact,
                    "width": radius.upper_bound - radius.lower_bound,
                    "n_ci_tests": len(report.ci_results),
                    "runtime_s": round(elapsed, 3),
                })

            except Exception as exc:
                logger.warning("Synthetic instance %s failed: %s", inst.name, exc)
                rows.append({
                    "name": inst.name,
                    "n_nodes": inst.adj.shape[0],
                    "n_edges": int(inst.adj.sum()),
                    "true_radius": inst.true_radius,
                    "est_lower": None,
                    "est_upper": None,
                    "covered": False,
                    "exact": False,
                    "width": None,
                    "n_ci_tests": 0,
                    "runtime_s": 0.0,
                })

        df = pd.DataFrame(rows)

        if not df.empty and estimated_radii:
            metrics = compute_all_metrics(instances[:len(estimated_radii)], estimated_radii)
            logger.info("Synthetic metrics: %s", metrics)

        return df

    def run_semi_synthetic(self) -> pd.DataFrame:
        """Run semi-synthetic benchmark evaluation.

        Returns
        -------
        pd.DataFrame
        """
        from causalcert.evaluation.semi_synthetic import SemiSyntheticBenchmark

        bench = SemiSyntheticBenchmark(
            dag_names=self.config.semi_synthetic_dags,
            n_samples=self.config.n_samples,
            n_repeats=self.config.n_repeats,
            seed=self.config.seed,
        )
        results = bench.run()
        return bench.summary_table(results)

    def run_ablation(self) -> pd.DataFrame:
        """Run ablation studies.

        Returns
        -------
        pd.DataFrame
        """
        from causalcert.evaluation.ablation import AblationHarness
        from causalcert.evaluation.dgp import SyntheticDGP

        # Generate instances for ablation (smaller set)
        dgp = SyntheticDGP(seed=self.config.seed)
        instances = dgp.batch_generate(
            n_instances=min(self.config.n_synthetic, 30),
            n_nodes_range=self.config.n_nodes_range,
            density_range=self.config.density_range,
            radius_range=self.config.radius_range,
            n_samples=self.config.n_samples,
        )

        harness = AblationHarness(
            conditions=self.config.ablation_conditions,
            dgp_instances=instances,
            base_config={
                "max_k": self.config.max_k,
                "seed": self.config.seed,
                "cache_dir": None,
            },
        )
        results = harness.run()
        return harness.aggregate_summary(results)

    def run_scalability(self) -> pd.DataFrame:
        """Run scalability profiling.

        Returns
        -------
        pd.DataFrame
        """
        from causalcert.evaluation.scalability import ScalabilityProfiler, profile_pipeline

        nodes_list = self.config.scalability_nodes or [10, 20, 50, 100]

        profiler = ScalabilityProfiler(
            n_nodes_list=nodes_list,
            density_list=[0.1, 0.2],
            n_repeats=2,
            seed=self.config.seed,
        )

        # Profile d-separation computation
        def dsep_fn(adj: np.ndarray) -> dict[str, Any]:
            from causalcert.dag.dsep import DSeparationOracle
            from causalcert.dag.graph import CausalDAG
            dag = CausalDAG(adj)
            oracle = DSeparationOracle(dag)
            implications = oracle.all_ci_implications()
            return {"n_implications": len(implications)}

        # Profile fragility scoring
        def fragility_fn(adj: np.ndarray) -> dict[str, Any]:
            from causalcert.dag.graph import CausalDAG
            from causalcert.fragility.scorer import compute_fragility_scores
            dag = CausalDAG(adj)
            n = adj.shape[0]
            scores = compute_fragility_scores(dag, 0, n - 1)
            return {"n_scores": len(scores)}

        results = profiler.profile_multiple({
            "dsep": dsep_fn,
            "fragility": fragility_fn,
        })

        return profiler.aggregate_summary(results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_results(self, results: dict[str, pd.DataFrame]) -> None:
        """Save result tables to output_dir."""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)

        for name, df in results.items():
            path = os.path.join(self.config.output_dir, f"{name}.csv")
            df.to_csv(path, index=False)
            logger.info("Saved %s results to %s", name, path)

    def summary_report(self) -> str:
        """Generate a text summary of all evaluation results.

        Returns
        -------
        str
        """
        lines: list[str] = ["CausalCert Evaluation Report", "=" * 40]

        for suite_name, df in self._results.items():
            lines.append(f"\n--- {suite_name} ---")
            lines.append(f"  Rows: {len(df)}")

            if "covered" in df.columns:
                lines.append(f"  Coverage: {df['covered'].mean():.3f}")
            if "exact" in df.columns:
                lines.append(f"  Exact match: {df['exact'].mean():.3f}")
            if "width" in df.columns:
                lines.append(f"  Mean width: {df['width'].mean():.2f}")
            if "runtime_s" in df.columns:
                lines.append(f"  Mean runtime: {df['runtime_s'].mean():.3f}s")
            if "mean_coverage" in df.columns:
                lines.append(f"  Mean coverage: {df['mean_coverage'].mean():.3f}")
            if "mean_time" in df.columns:
                lines.append(f"  Mean time: {df['mean_time'].mean():.4f}s")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def quick_evaluation(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Run a quick evaluation with minimal settings.

    Parameters
    ----------
    seed : int

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    config = EvaluationConfig(
        n_synthetic=20,
        n_samples=200,
        n_repeats=3,
        include_semi_synthetic=False,
        include_ablation=False,
        include_scalability=False,
        seed=seed,
        n_nodes_range=(6, 12),
        radius_range=(1, 2),
    )
    runner = EvaluationRunner(config)
    return runner.run_all()


def full_evaluation(
    output_dir: str = "eval_results",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Run a comprehensive evaluation.

    Parameters
    ----------
    output_dir : str
    seed : int

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    config = EvaluationConfig(
        n_synthetic=100,
        n_samples=1000,
        n_repeats=10,
        include_semi_synthetic=True,
        include_ablation=True,
        include_scalability=True,
        seed=seed,
        output_dir=output_dir,
    )
    runner = EvaluationRunner(config)
    return runner.run_all()
