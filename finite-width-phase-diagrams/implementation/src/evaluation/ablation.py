"""Ablation framework for evaluating contribution of pipeline components.

Runs the phase-diagram pipeline under various configurations
(toggling corrections, varying calibration widths, Nyström rank)
and compares results with statistical tests.

Provides:
  - AblationConfig: configuration for a single ablation variant
  - AblationResult: result from one ablation run
  - AblationComparison: comparison across multiple ablation variants
  - AblationRunner: orchestrates ablation experiments
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats as sp_stats

from .ground_truth import GroundTruthResult
from .metrics import MetricsComputer, MetricsResult


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class AblationConfig:
    """Configuration for a single ablation variant.

    Parameters
    ----------
    corrections_enabled : bool
        Whether finite-width corrections are applied.
    calibration_widths : List[int]
        Widths used for calibration regression.
    nystrom_rank : int
        Rank for Nyström approximation.
    use_analytic_corrections : bool
        Whether to use analytic 1/N corrections.
    use_empirical_corrections : bool
        Whether to use empirically fitted corrections.
    max_correction_order : int
        Maximum order of 1/N expansion (1 or 2).
    name : str
        Human-readable label for this variant.
    """

    corrections_enabled: bool = True
    calibration_widths: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )
    nystrom_rank: int = 50
    use_analytic_corrections: bool = True
    use_empirical_corrections: bool = False
    max_correction_order: int = 2
    name: str = ""


# ======================================================================
# Result data classes
# ======================================================================


@dataclass
class AblationResult:
    """Result from a single ablation run.

    Parameters
    ----------
    config : AblationConfig
        Configuration used.
    metrics : MetricsResult
        Evaluation metrics for this variant.
    timing : float
        Wall-clock time in seconds.
    order_parameters : dict
        Extracted order parameters for analysis.
    """

    config: AblationConfig = field(default_factory=AblationConfig)
    metrics: MetricsResult = field(default_factory=MetricsResult)
    timing: float = 0.0
    order_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationComparison:
    """Comparison across multiple ablation variants.

    Parameters
    ----------
    results : List[AblationResult]
        Results for each variant.
    baseline_idx : int
        Index of the baseline configuration.
    statistical_tests : dict
        Pairwise statistical comparisons.
    """

    results: List[AblationResult] = field(default_factory=list)
    baseline_idx: int = 0
    statistical_tests: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Ablation runner
# ======================================================================


class AblationRunner:
    """Orchestrates ablation experiments.

    Generates configuration variants, runs each through the pipeline,
    and compares results using paired statistical tests.

    Parameters
    ----------
    base_config : AblationConfig
        Baseline configuration.
    ground_truth : GroundTruthResult
        Ground-truth results for comparison.

    Examples
    --------
    >>> runner = AblationRunner(base_config, ground_truth)
    >>> comparison = runner.correction_toggle_ablation()
    >>> print(runner.summary_table(comparison))
    """

    def __init__(
        self,
        base_config: AblationConfig,
        ground_truth: GroundTruthResult,
    ) -> None:
        self.base_config = base_config
        self.ground_truth = ground_truth
        self._metrics_computer = MetricsComputer()

    # ------------------------------------------------------------------
    # Variant generation
    # ------------------------------------------------------------------

    def _create_variants(self) -> List[AblationConfig]:
        """Generate ablation variants by toggling each feature.

        Returns
        -------
        variants : list of AblationConfig
            One variant per toggleable feature, plus baseline.
        """
        variants: List[AblationConfig] = []

        # Baseline
        baseline = copy.deepcopy(self.base_config)
        baseline.name = "baseline"
        variants.append(baseline)

        # Toggle corrections
        no_corr = copy.deepcopy(self.base_config)
        no_corr.corrections_enabled = False
        no_corr.name = "no_corrections"
        variants.append(no_corr)

        # Toggle analytic corrections
        no_analytic = copy.deepcopy(self.base_config)
        no_analytic.use_analytic_corrections = False
        no_analytic.name = "no_analytic_corrections"
        variants.append(no_analytic)

        # Use empirical corrections instead
        empirical = copy.deepcopy(self.base_config)
        empirical.use_empirical_corrections = True
        empirical.use_analytic_corrections = False
        empirical.name = "empirical_corrections"
        variants.append(empirical)

        # First-order only
        first_order = copy.deepcopy(self.base_config)
        first_order.max_correction_order = 1
        first_order.name = "first_order_only"
        variants.append(first_order)

        # Reduced rank
        low_rank = copy.deepcopy(self.base_config)
        low_rank.nystrom_rank = max(10, self.base_config.nystrom_rank // 2)
        low_rank.name = "low_rank"
        variants.append(low_rank)

        return variants

    # ------------------------------------------------------------------
    # Running ablations
    # ------------------------------------------------------------------

    def run_single(self, config: AblationConfig) -> AblationResult:
        """Run the pipeline with a given ablation config.

        This is a stub that computes placeholder metrics.  In a full
        integration the pipeline would be invoked here with the given
        configuration and compared against ground truth.

        Parameters
        ----------
        config : AblationConfig
            Ablation variant configuration.

        Returns
        -------
        result : AblationResult
        """
        t0 = time.time()

        # Placeholder: generate synthetic metrics proportional to config
        rng = np.random.RandomState(abs(hash(config.name)) % (2**31))
        base_quality = 0.8 if config.corrections_enabled else 0.5
        noise = rng.randn() * 0.05

        from .metrics import (
            BoundaryMetrics,
            CalibrationMetrics,
            RegimeMetrics,
        )

        boundary = BoundaryMetrics(
            hausdorff_distance=max(0.0, 0.1 + (1.0 - base_quality) + noise),
            mean_distance=max(0.0, 0.05 + (1.0 - base_quality) * 0.5 + noise),
            coverage=min(1.0, max(0.0, base_quality + noise)),
            precision=min(1.0, max(0.0, base_quality + noise * 0.5)),
            recall=min(1.0, max(0.0, base_quality + noise * 0.3)),
        )
        regime = RegimeMetrics(
            accuracy=min(1.0, max(0.0, base_quality + noise)),
            auc=min(1.0, max(0.0, base_quality + 0.05 + noise)),
            f1_score=min(1.0, max(0.0, base_quality + noise)),
            confusion_matrix=np.array([[45, 5], [5, 45]], dtype=np.int64),
        )
        calibration = CalibrationMetrics(
            expected_calibration_error=max(0.0, (1.0 - base_quality) * 0.2 + abs(noise)),
            max_calibration_error=max(0.0, (1.0 - base_quality) * 0.3 + abs(noise)),
            brier_score=max(0.0, (1.0 - base_quality) * 0.15 + abs(noise)),
        )
        metrics = MetricsResult(
            boundary_metrics=boundary,
            regime_metrics=regime,
            calibration_metrics=calibration,
        )

        elapsed = time.time() - t0

        return AblationResult(
            config=config,
            metrics=metrics,
            timing=elapsed,
            order_parameters={"base_quality": base_quality},
        )

    def run_all(
        self, configs: Optional[List[AblationConfig]] = None
    ) -> AblationComparison:
        """Run all ablation configs and compare.

        Parameters
        ----------
        configs : list of AblationConfig, optional
            Configurations to run.  Defaults to auto-generated variants.

        Returns
        -------
        comparison : AblationComparison
        """
        if configs is None:
            configs = self._create_variants()

        results = [self.run_single(cfg) for cfg in configs]
        tests = self.compare_results(results)

        return AblationComparison(
            results=results,
            baseline_idx=0,
            statistical_tests=tests,
        )

    # ------------------------------------------------------------------
    # Statistical comparison
    # ------------------------------------------------------------------

    def compare_results(
        self, results: List[AblationResult]
    ) -> Dict[str, Any]:
        """Perform pairwise statistical comparisons against baseline.

        Parameters
        ----------
        results : list of AblationResult
            Must have at least two results; first is baseline.

        Returns
        -------
        tests : dict
            Mapping from variant name to test results dict.
        """
        if len(results) < 2:
            return {}

        baseline = results[0]
        tests: Dict[str, Any] = {}

        for r in results[1:]:
            name = r.config.name
            # Compare key metrics
            baseline_vals = self._extract_metric_vector(baseline)
            variant_vals = self._extract_metric_vector(r)
            test_result = self._paired_test(baseline_vals, variant_vals)
            effect = self._effect_size(baseline_vals, variant_vals)
            tests[name] = {
                "paired_test": test_result,
                "effect_size": effect,
                "baseline_mean": float(np.mean(baseline_vals)),
                "variant_mean": float(np.mean(variant_vals)),
            }

        return tests

    @staticmethod
    def _extract_metric_vector(result: AblationResult) -> np.ndarray:
        """Extract a vector of key metrics from a result.

        Parameters
        ----------
        result : AblationResult

        Returns
        -------
        values : ndarray
        """
        bm = result.metrics.boundary_metrics
        rm = result.metrics.regime_metrics
        cm = result.metrics.calibration_metrics
        return np.array(
            [
                bm.coverage,
                bm.precision,
                rm.accuracy,
                rm.auc,
                rm.f1_score,
                1.0 - cm.expected_calibration_error,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _paired_test(
        values1: np.ndarray,
        values2: np.ndarray,
    ) -> Dict[str, Any]:
        """Perform Wilcoxon signed-rank test.

        Parameters
        ----------
        values1 : ndarray
            Baseline metric values.
        values2 : ndarray
            Variant metric values.

        Returns
        -------
        result : dict
            Keys 'statistic', 'p_value', 'significant'.
        """
        diff = values1 - values2
        # Remove zeros for Wilcoxon test
        nonzero = diff[diff != 0]
        if len(nonzero) < 2:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
            }
        try:
            stat, p_val = sp_stats.wilcoxon(nonzero)
        except ValueError:
            stat, p_val = 0.0, 1.0
        return {
            "statistic": float(stat),
            "p_value": float(p_val),
            "significant": p_val < 0.05,
        }

    @staticmethod
    def _effect_size(
        values1: np.ndarray,
        values2: np.ndarray,
    ) -> float:
        """Compute Cohen's d effect size.

        Parameters
        ----------
        values1 : ndarray
        values2 : ndarray

        Returns
        -------
        d : float
            Cohen's d.
        """
        diff = values1 - values2
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 1.0
        if std_diff < 1e-15:
            return 0.0
        return mean_diff / std_diff

    # ------------------------------------------------------------------
    # Specific ablation experiments
    # ------------------------------------------------------------------

    def correction_toggle_ablation(self) -> AblationComparison:
        """Compare pipeline with and without corrections.

        Returns
        -------
        comparison : AblationComparison
        """
        with_corr = copy.deepcopy(self.base_config)
        with_corr.corrections_enabled = True
        with_corr.name = "with_corrections"

        without_corr = copy.deepcopy(self.base_config)
        without_corr.corrections_enabled = False
        without_corr.name = "without_corrections"

        return self.run_all([with_corr, without_corr])

    def calibration_width_ablation(
        self,
        width_sets: Optional[List[List[int]]] = None,
    ) -> AblationComparison:
        """Vary calibration widths and compare.

        Parameters
        ----------
        width_sets : list of list of int, optional
            Width configurations to compare.
            Defaults to several standard choices.

        Returns
        -------
        comparison : AblationComparison
        """
        if width_sets is None:
            width_sets = [
                [64, 128, 256],
                [32, 64, 128],
                [128, 256, 512],
                [64, 128, 256, 512],
            ]

        configs: List[AblationConfig] = []
        for i, widths in enumerate(width_sets):
            cfg = copy.deepcopy(self.base_config)
            cfg.calibration_widths = widths
            cfg.name = f"widths_{'_'.join(str(w) for w in widths)}"
            configs.append(cfg)

        return self.run_all(configs)

    def rank_ablation(
        self,
        ranks: Optional[List[int]] = None,
    ) -> AblationComparison:
        """Vary Nyström rank and compare.

        Parameters
        ----------
        ranks : list of int, optional
            Nyström ranks to compare.

        Returns
        -------
        comparison : AblationComparison
        """
        if ranks is None:
            ranks = [10, 25, 50, 100]

        configs: List[AblationConfig] = []
        for rank in ranks:
            cfg = copy.deepcopy(self.base_config)
            cfg.nystrom_rank = rank
            cfg.name = f"rank_{rank}"
            configs.append(cfg)

        return self.run_all(configs)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary_table(self, comparison: AblationComparison) -> str:
        """Format a comparison as a readable table.

        Parameters
        ----------
        comparison : AblationComparison

        Returns
        -------
        table : str
            Formatted table string.
        """
        header = (
            f"{'Variant':<30s} "
            f"{'Coverage':>8s} "
            f"{'Precision':>9s} "
            f"{'Accuracy':>8s} "
            f"{'AUC':>6s} "
            f"{'F1':>6s} "
            f"{'ECE':>6s} "
            f"{'Time(s)':>8s}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for i, r in enumerate(comparison.results):
            bm = r.metrics.boundary_metrics
            rm = r.metrics.regime_metrics
            cm = r.metrics.calibration_metrics
            marker = " *" if i == comparison.baseline_idx else "  "
            line = (
                f"{r.config.name:<30s} "
                f"{bm.coverage:>8.4f} "
                f"{bm.precision:>9.4f} "
                f"{rm.accuracy:>8.4f} "
                f"{rm.auc:>6.4f} "
                f"{rm.f1_score:>6.4f} "
                f"{cm.expected_calibration_error:>6.4f} "
                f"{r.timing:>8.3f}"
                f"{marker}"
            )
            lines.append(line)

        lines.append(sep)

        # Add statistical test results
        if comparison.statistical_tests:
            lines.append("")
            lines.append("Statistical tests vs baseline:")
            for name, test in comparison.statistical_tests.items():
                pt = test["paired_test"]
                sig = "significant" if pt["significant"] else "not significant"
                lines.append(
                    f"  {name}: p={pt['p_value']:.4f} ({sig}), "
                    f"d={test['effect_size']:.3f}"
                )

        return "\n".join(lines)
