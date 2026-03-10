"""
Ablation study harness.

Systematically disables or swaps individual pipeline components to
measure their contribution to the overall performance of CausalCert.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from causalcert.evaluation.dgp import DGPInstance
from causalcert.types import RobustnessRadius

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AblationResult:
    """Result of a single ablation condition.

    Attributes
    ----------
    condition : str
        Description of the ablation (e.g. ``"no_ensemble"``, ``"ilp_only"``).
    dgp_name : str
        DGP name.
    coverage : float
        Coverage rate.
    interval_width : float
        Average interval width.
    runtime_s : float
        Total runtime in seconds.
    extra : dict[str, Any]
        Additional metrics.
    """

    condition: str
    dgp_name: str
    coverage: float = 0.0
    interval_width: float = 0.0
    runtime_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Condition → config modification mapping
# ---------------------------------------------------------------------------


def _apply_ablation_condition(
    condition: str,
    base_config: dict[str, Any],
) -> dict[str, Any]:
    """Apply an ablation condition to a pipeline configuration dict.

    Parameters
    ----------
    condition : str
    base_config : dict[str, Any]
        Baseline configuration (copied before modifying).

    Returns
    -------
    dict[str, Any]
    """
    import copy
    cfg = copy.deepcopy(base_config)

    if condition == "full":
        pass  # No change, baseline

    elif condition == "no_ensemble":
        cfg["ci_method"] = "partial_correlation"

    elif condition == "no_fdr":
        cfg["multiplicity_correction"] = None

    elif condition == "ilp_only":
        cfg["solver_strategy"] = "ILP"

    elif condition == "lp_only":
        cfg["solver_strategy"] = "LP_RELAXATION"

    elif condition == "fpt_only":
        cfg["solver_strategy"] = "FPT"

    elif condition == "cdcl_only":
        cfg["solver_strategy"] = "CDCL"

    elif condition == "no_fragility_pruning":
        cfg["fragility_pruning"] = False

    elif condition == "no_incremental_dsep":
        cfg["incremental_dsep"] = False

    elif condition == "no_estimation":
        cfg.setdefault("steps", {})["estimation"] = False

    elif condition == "no_cache":
        cfg["cache_dir"] = None

    elif condition == "kernel_ci":
        cfg["ci_method"] = "kernel"

    elif condition == "crt_ci":
        cfg["ci_method"] = "crt"

    elif condition == "rank_ci":
        cfg["ci_method"] = "rank"

    elif condition == "max_k_3":
        cfg["max_k"] = 3

    elif condition == "max_k_10":
        cfg["max_k"] = 10

    else:
        logger.warning("Unknown ablation condition: %s (using baseline)", condition)

    return cfg


def _run_single_instance(
    instance: DGPInstance,
    config_dict: dict[str, Any],
) -> tuple[RobustnessRadius | None, float]:
    """Run the pipeline on a single DGP instance.

    Returns
    -------
    tuple[RobustnessRadius | None, float]
        ``(radius, elapsed_seconds)``
    """
    try:
        from causalcert.pipeline.config import PipelineRunConfig
        from causalcert.pipeline.orchestrator import CausalCertPipeline

        cfg = PipelineRunConfig(
            treatment=instance.treatment,
            outcome=instance.outcome,
            max_k=config_dict.get("max_k", 5),
            n_folds=config_dict.get("n_folds", 2),
            seed=config_dict.get("seed", 42),
            cache_dir=config_dict.get("cache_dir", None),
        )

        # Apply solver strategy
        strategy = config_dict.get("solver_strategy")
        if strategy:
            from causalcert.types import SolverStrategy
            cfg.solver.strategy = SolverStrategy[strategy]

        # Apply CI method
        ci_method = config_dict.get("ci_method")
        if ci_method:
            from causalcert.types import CITestMethod
            try:
                cfg.ci_test.method = CITestMethod[ci_method.upper()]
            except KeyError:
                pass

        # Apply steps config
        steps = config_dict.get("steps", {})
        if "estimation" in steps:
            cfg.steps.estimation = steps["estimation"]

        t0 = time.perf_counter()
        pipeline = CausalCertPipeline(cfg)
        report = pipeline.run(instance.adj, instance.data)
        elapsed = time.perf_counter() - t0

        return report.radius, elapsed

    except Exception as exc:
        logger.warning("Pipeline run failed: %s", exc)
        return None, 0.0


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------


class AblationHarness:
    """Ablation study harness for CausalCert.

    Parameters
    ----------
    conditions : list[str]
        Ablation conditions to test.
    dgp_instances : Sequence[DGPInstance]
        DGP instances to evaluate on.
    base_config : dict[str, Any] | None
        Baseline pipeline config (as dict).
    """

    def __init__(
        self,
        conditions: list[str] | None = None,
        dgp_instances: Sequence[DGPInstance] | None = None,
        base_config: dict[str, Any] | None = None,
    ) -> None:
        self.conditions = conditions or [
            "full",
            "no_ensemble",
            "no_fdr",
            "ilp_only",
            "lp_only",
            "no_fragility_pruning",
            "no_incremental_dsep",
        ]
        self.dgp_instances = list(dgp_instances) if dgp_instances else []
        self.base_config: dict[str, Any] = base_config or {
            "max_k": 5,
            "n_folds": 2,
            "seed": 42,
            "cache_dir": None,
        }

    def run(self) -> list[AblationResult]:
        """Run all ablation conditions.

        Returns
        -------
        list[AblationResult]
        """
        if not self.dgp_instances:
            logger.warning("No DGP instances provided; generating defaults")
            self._generate_default_instances()

        results: list[AblationResult] = []

        for condition in self.conditions:
            logger.info("Running ablation condition: %s", condition)
            cfg = _apply_ablation_condition(condition, self.base_config)

            condition_radii: list[RobustnessRadius] = []
            condition_times: list[float] = []
            condition_covered: list[bool] = []

            for inst in self.dgp_instances:
                radius, elapsed = _run_single_instance(inst, cfg)

                if radius is not None:
                    condition_radii.append(radius)
                    condition_times.append(elapsed)
                    covered = radius.lower_bound <= inst.true_radius <= radius.upper_bound
                    condition_covered.append(covered)
                else:
                    condition_times.append(0.0)
                    condition_covered.append(False)

                results.append(AblationResult(
                    condition=condition,
                    dgp_name=inst.name,
                    coverage=1.0 if condition_covered[-1] else 0.0,
                    interval_width=(
                        radius.upper_bound - radius.lower_bound
                        if radius is not None else float("nan")
                    ),
                    runtime_s=elapsed,
                    extra={
                        "true_radius": inst.true_radius,
                        "est_lower": radius.lower_bound if radius else None,
                        "est_upper": radius.upper_bound if radius else None,
                    },
                ))

            # Log condition summary
            n_covered = sum(condition_covered)
            n_total = len(condition_covered)
            cov_rate = n_covered / max(n_total, 1)
            avg_time = np.mean(condition_times) if condition_times else 0
            logger.info(
                "  Condition %s: coverage=%.2f (%d/%d), avg_time=%.3fs",
                condition, cov_rate, n_covered, n_total, avg_time,
            )

        return results

    def _generate_default_instances(self) -> None:
        """Generate a small set of DGP instances for ablation."""
        from causalcert.evaluation.dgp import SyntheticDGP

        dgp = SyntheticDGP(seed=42)
        self.dgp_instances = dgp.batch_generate(
            n_instances=20,
            n_nodes_range=(8, 15),
            density_range=(0.15, 0.3),
            radius_range=(1, 3),
            n_samples=500,
        )

    def run_single_condition(
        self,
        condition: str,
    ) -> list[AblationResult]:
        """Run a single ablation condition.

        Parameters
        ----------
        condition : str

        Returns
        -------
        list[AblationResult]
        """
        if not self.dgp_instances:
            self._generate_default_instances()

        cfg = _apply_ablation_condition(condition, self.base_config)
        results: list[AblationResult] = []

        for inst in self.dgp_instances:
            radius, elapsed = _run_single_instance(inst, cfg)
            covered = (
                radius.lower_bound <= inst.true_radius <= radius.upper_bound
                if radius else False
            )
            results.append(AblationResult(
                condition=condition,
                dgp_name=inst.name,
                coverage=1.0 if covered else 0.0,
                interval_width=(
                    radius.upper_bound - radius.lower_bound
                    if radius is not None else float("nan")
                ),
                runtime_s=elapsed,
                extra={
                    "true_radius": inst.true_radius,
                    "est_lower": radius.lower_bound if radius else None,
                    "est_upper": radius.upper_bound if radius else None,
                },
            ))

        return results

    def summary_table(self, results: list[AblationResult]) -> pd.DataFrame:
        """Create a pivot table of ablation results.

        Parameters
        ----------
        results : list[AblationResult]

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for r in results:
            row = {
                "condition": r.condition,
                "dgp_name": r.dgp_name,
                "coverage": r.coverage,
                "interval_width": r.interval_width,
                "runtime_s": round(r.runtime_s, 3),
            }
            row.update(r.extra)
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def aggregate_summary(self, results: list[AblationResult]) -> pd.DataFrame:
        """Aggregate ablation results per condition.

        Parameters
        ----------
        results : list[AblationResult]

        Returns
        -------
        pd.DataFrame
        """
        df = self.summary_table(results)
        if df.empty:
            return pd.DataFrame()

        agg = df.groupby("condition").agg(
            mean_coverage=("coverage", "mean"),
            mean_width=("interval_width", "mean"),
            mean_runtime=("runtime_s", "mean"),
            total_runtime=("runtime_s", "sum"),
            n_instances=("dgp_name", "count"),
        ).reset_index()

        # Sort by coverage descending
        agg = agg.sort_values("mean_coverage", ascending=False)

        # Round
        for col in ["mean_coverage", "mean_width"]:
            agg[col] = agg[col].round(3)
        for col in ["mean_runtime", "total_runtime"]:
            agg[col] = agg[col].round(2)

        return agg

    def delta_table(self, results: list[AblationResult]) -> pd.DataFrame:
        """Compute performance deltas relative to the ``"full"`` baseline.

        Parameters
        ----------
        results : list[AblationResult]

        Returns
        -------
        pd.DataFrame
            Columns: condition, delta_coverage, delta_width, delta_runtime.
        """
        agg = self.aggregate_summary(results)
        if agg.empty or "full" not in agg["condition"].values:
            return pd.DataFrame()

        baseline = agg[agg["condition"] == "full"].iloc[0]
        rows = []
        for _, row in agg.iterrows():
            rows.append({
                "condition": row["condition"],
                "delta_coverage": round(row["mean_coverage"] - baseline["mean_coverage"], 4),
                "delta_width": round(row["mean_width"] - baseline["mean_width"], 3),
                "delta_runtime": round(row["mean_runtime"] - baseline["mean_runtime"], 3),
                "speedup": (
                    round(baseline["mean_runtime"] / row["mean_runtime"], 2)
                    if row["mean_runtime"] > 0 else float("inf")
                ),
            })

        return pd.DataFrame(rows).sort_values("delta_coverage")
