"""Benchmark runners for the CPA pipeline.

Provides benchmark execution, result aggregation, statistical comparison,
and reporting utilities.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.generators import (
    BenchmarkResult,
    CSVMGenerator,
    FSVPGenerator,
    GroundTruth,
    SemiSyntheticGenerator,
    TPSGenerator,
)
from benchmarks.metrics import (
    ArchiveMetrics,
    BaselineComparisons,
    CertificateMetrics,
    ClassificationMetrics,
    TippingPointMetrics,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Run record
# =====================================================================


@dataclass
class RunRecord:
    """Record of a single benchmark run.

    Attributes
    ----------
    scenario : str
        Benchmark scenario name (e.g. "FSVP-A").
    replication : int
        Replication index.
    seed : int
        Random seed used.
    wall_time : float
        Wall time in seconds.
    classification_metrics : dict
        Classification metrics.
    tipping_metrics : dict
        Tipping-point detection metrics.
    certificate_metrics : dict
        Certificate metrics.
    archive_metrics : dict
        Archive metrics.
    metadata : dict
        Additional metadata.
    """

    scenario: str = ""
    replication: int = 0
    seed: int = 0
    wall_time: float = 0.0
    classification_metrics: Dict[str, Any] = field(default_factory=dict)
    tipping_metrics: Dict[str, Any] = field(default_factory=dict)
    certificate_metrics: Dict[str, Any] = field(default_factory=dict)
    archive_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        """Return summary statistics."""
        s: Dict[str, float] = {"wall_time": self.wall_time}

        if self.classification_metrics:
            s["macro_f1"] = self.classification_metrics.get("macro_f1", 0.0)
            s["accuracy"] = self.classification_metrics.get("accuracy", 0.0)

        if self.tipping_metrics:
            s["tp_f1"] = self.tipping_metrics.get("f1", 0.0)
            s["tp_mad"] = self.tipping_metrics.get("mad", float("inf"))

        if self.certificate_metrics:
            s["coverage"] = self.certificate_metrics.get("coverage", 0.0)

        if self.archive_metrics:
            s["qd_score"] = self.archive_metrics.get("qd_score", 0.0)

        return s


# =====================================================================
# Benchmark runner
# =====================================================================


class BenchmarkRunner:
    """Runs CPA benchmarks with replication.

    Parameters
    ----------
    n_replications : int
        Number of replications per scenario.
    base_seed : int
        Base random seed (seed = base_seed + replication).
    output_dir : str or Path
        Directory for results.
    pipeline_fn : callable, optional
        Function(data, ground_truth, config) → result.
        If None, uses a dummy passthrough.

    Examples
    --------
    >>> runner = BenchmarkRunner(n_replications=5)
    >>> records = runner.run_scenario("FSVP-A")
    """

    # Predefined scenarios
    SCENARIOS = {
        "FSVP-A": {
            "generator": "fsvp",
            "params": {"n_variables": 6, "K": 10, "N_per_context": 200},
        },
        "FSVP-B": {
            "generator": "fsvp",
            "params": {"n_variables": 10, "K": 20, "N_per_context": 500},
        },
        "FSVP-C": {
            "generator": "fsvp",
            "params": {"n_variables": 15, "K": 30, "N_per_context": 1000},
        },
        "CSVM-A": {
            "generator": "csvm",
            "params": {"n_variables": 6, "K": 8, "N_per_context": 200},
        },
        "CSVM-B": {
            "generator": "csvm",
            "params": {"n_variables": 10, "K": 15, "N_per_context": 500},
        },
        "TPS-A": {
            "generator": "tps",
            "params": {"n_variables": 6, "K": 10, "N_per_context": 200},
        },
        "TPS-B": {
            "generator": "tps",
            "params": {"n_variables": 10, "K": 20, "N_per_context": 500},
        },
    }

    def __init__(
        self,
        n_replications: int = 10,
        base_seed: int = 42,
        output_dir: str | Path = "benchmark_results",
        pipeline_fn: Optional[Callable] = None,
    ):
        self.n_replications = n_replications
        self.base_seed = base_seed
        self.output_dir = Path(output_dir)
        self.pipeline_fn = pipeline_fn or self._dummy_pipeline

        self._class_metrics = ClassificationMetrics()
        self._tp_metrics = TippingPointMetrics()
        self._cert_metrics = CertificateMetrics()
        self._archive_metrics = ArchiveMetrics()

    def run_scenario(
        self,
        scenario_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[RunRecord]:
        """Run all replications of a scenario.

        Parameters
        ----------
        scenario_name : str
            Name of predefined scenario, or custom.
        config : dict, optional
            Override scenario parameters.

        Returns
        -------
        list of RunRecord
        """
        if scenario_name in self.SCENARIOS:
            spec = dict(self.SCENARIOS[scenario_name])
            if config:
                spec["params"].update(config)
        elif config:
            spec = config
        else:
            raise ValueError(f"Unknown scenario '{scenario_name}' and no config given")

        records: List[RunRecord] = []

        for rep in range(self.n_replications):
            seed = self.base_seed + rep
            logger.info(
                f"Scenario {scenario_name}, replication {rep + 1}/{self.n_replications}, seed={seed}"
            )

            try:
                record = self._run_single(scenario_name, spec, rep, seed)
                records.append(record)
            except Exception as exc:
                logger.error(f"Replication {rep} failed: {exc}")
                records.append(
                    RunRecord(
                        scenario=scenario_name,
                        replication=rep,
                        seed=seed,
                        metadata={"error": str(exc)},
                    )
                )

        return records

    def run_all_scenarios(
        self,
        scenarios: Optional[List[str]] = None,
    ) -> Dict[str, List[RunRecord]]:
        """Run multiple scenarios.

        Parameters
        ----------
        scenarios : list of str, optional
            Subset of scenarios (default: all).

        Returns
        -------
        dict
            Scenario → list of RunRecord.
        """
        if scenarios is None:
            scenarios = list(self.SCENARIOS.keys())

        all_records: Dict[str, List[RunRecord]] = {}
        for name in scenarios:
            logger.info(f"=== Running scenario: {name} ===")
            all_records[name] = self.run_scenario(name)

        return all_records

    def _run_single(
        self,
        scenario_name: str,
        spec: Dict[str, Any],
        replication: int,
        seed: int,
    ) -> RunRecord:
        """Execute a single replication."""
        generator = self._make_generator(spec)
        bench_result = generator.generate(seed=seed)

        t0 = time.time()
        pipeline_output = self.pipeline_fn(
            bench_result.data, bench_result.ground_truth, spec.get("params", {})
        )
        wall_time = time.time() - t0

        record = RunRecord(
            scenario=scenario_name,
            replication=replication,
            seed=seed,
            wall_time=wall_time,
        )

        gt = bench_result.ground_truth

        # Classification metrics
        if pipeline_output and "classifications" in pipeline_output:
            record.classification_metrics = self._class_metrics.evaluate(
                pipeline_output["classifications"],
                gt.classifications,
            )

        # Tipping-point metrics
        if pipeline_output and "tipping_points" in pipeline_output and gt.tipping_points:
            record.tipping_metrics = self._tp_metrics.evaluate(
                pipeline_output["tipping_points"],
                gt.tipping_points,
                K=len(bench_result.data),
            )

        # Certificate metrics
        if pipeline_output and "certificates" in pipeline_output:
            stable_vars = {
                v for v, c in gt.classifications.items() if c == "invariant"
            }
            record.certificate_metrics = self._cert_metrics.evaluate(
                pipeline_output["certificates"],
                stable_vars,
            )

        # Archive metrics
        if pipeline_output and "archive_descriptors" in pipeline_output:
            record.archive_metrics = self._archive_metrics.evaluate(
                pipeline_output["archive_descriptors"],
                pipeline_output.get("archive_fitnesses", np.ones(len(pipeline_output["archive_descriptors"]))),
                pipeline_output.get("archive_capacity", 256),
            )

        record.metadata = {
            "generator": spec.get("generator", "unknown"),
            "params": spec.get("params", {}),
        }

        return record

    def _make_generator(self, spec: Dict[str, Any]) -> Any:
        """Create generator from spec."""
        gtype = spec.get("generator", "fsvp")
        params = spec.get("params", {})

        n_vars = params.get("n_variables", 6)
        K = params.get("K", 10)
        N = params.get("N_per_context", 200)

        if gtype == "fsvp":
            return FSVPGenerator(n_variables=n_vars, K=K, N_per_context=N)
        elif gtype == "csvm":
            return CSVMGenerator(n_variables=n_vars, K=K, N_per_context=N)
        elif gtype == "tps":
            return TPSGenerator(n_variables=n_vars, K=K, N_per_context=N)
        elif gtype == "semi_synthetic":
            return SemiSyntheticGenerator(n_variables=n_vars, K=K, N_per_context=N)
        else:
            raise ValueError(f"Unknown generator type: {gtype}")

    def _dummy_pipeline(
        self,
        data: Dict[str, np.ndarray],
        ground_truth: GroundTruth,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dummy pipeline that returns trivial results.

        Used when no pipeline function is provided.
        """
        first_data = next(iter(data.values()))
        p = first_data.shape[1]
        variable_names = [f"X{i}" for i in range(p)]

        return {
            "classifications": {v: "invariant" for v in variable_names},
            "tipping_points": [],
            "certificates": {v: True for v in variable_names},
        }


# =====================================================================
# Result aggregator
# =====================================================================


class ResultAggregator:
    """Aggregate benchmark results over replications.

    Examples
    --------
    >>> agg = ResultAggregator()
    >>> summary = agg.aggregate(records)
    """

    def aggregate(
        self,
        records: List[RunRecord],
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics over replications.

        Parameters
        ----------
        records : list of RunRecord
            Records from a single scenario.

        Returns
        -------
        dict
            Metric → {mean, std, median, min, max, n}.
        """
        valid = [r for r in records if "error" not in r.metadata]
        if not valid:
            return {"error": "All replications failed"}

        summaries = [r.summary() for r in valid]
        all_keys = set()
        for s in summaries:
            all_keys.update(s.keys())

        result: Dict[str, Dict[str, Any]] = {}
        for key in sorted(all_keys):
            values = [s[key] for s in summaries if key in s and np.isfinite(s[key])]
            if not values:
                result[key] = {"mean": float("nan"), "n": 0}
                continue

            arr = np.array(values)
            result[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": len(arr),
            }

        return result

    def aggregate_multi(
        self,
        all_records: Dict[str, List[RunRecord]],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Aggregate across scenarios.

        Parameters
        ----------
        all_records : dict
            Scenario → list of RunRecord.

        Returns
        -------
        dict
            Scenario → metric → stats.
        """
        return {
            name: self.aggregate(records) for name, records in all_records.items()
        }


# =====================================================================
# Statistical comparison
# =====================================================================


class StatisticalComparison:
    """Statistical tests for comparing methods.

    Examples
    --------
    >>> comp = StatisticalComparison()
    >>> result = comp.paired_test(scores_a, scores_b)
    """

    def paired_test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        test: str = "wilcoxon",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Paired statistical test.

        Parameters
        ----------
        scores_a, scores_b : np.ndarray
            Paired scores from two methods.
        test : str
            "wilcoxon" or "t-test".
        alpha : float
            Significance level.

        Returns
        -------
        dict
            Statistic, p-value, significant, effect size.
        """
        from scipy import stats as sp_stats

        a = np.asarray(scores_a, dtype=float)
        b = np.asarray(scores_b, dtype=float)

        if len(a) != len(b):
            raise ValueError("Score arrays must have equal length")

        n = len(a)
        diff = a - b
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

        # Effect size (Cohen's d for paired)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        if test == "wilcoxon" and n >= 6:
            try:
                stat, pval = sp_stats.wilcoxon(a, b, alternative="two-sided")
                stat = float(stat)
                pval = float(pval)
            except ValueError:
                stat = 0.0
                pval = 1.0
        elif test == "t-test" or n < 6:
            stat, pval = sp_stats.ttest_rel(a, b)
            stat = float(stat)
            pval = float(pval)
        else:
            stat = 0.0
            pval = 1.0

        return {
            "test": test if (test != "wilcoxon" or n >= 6) else "t-test",
            "statistic": stat,
            "p_value": pval,
            "significant": pval < alpha,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "cohens_d": cohens_d,
            "n": n,
            "alpha": alpha,
        }

    def multi_comparison(
        self,
        method_scores: Dict[str, np.ndarray],
        reference: str = "CPA",
        test: str = "wilcoxon",
        correction: str = "bonferroni",
        alpha: float = 0.05,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple methods against a reference.

        Parameters
        ----------
        method_scores : dict
            Method → scores array.
        reference : str
            Reference method name.
        test : str
            Statistical test to use.
        correction : str
            Multiple comparison correction: "bonferroni" or "holm".
        alpha : float
            Family-wise significance level.

        Returns
        -------
        dict
            Method → test result.
        """
        if reference not in method_scores:
            raise ValueError(f"Reference '{reference}' not in method_scores")

        ref_scores = method_scores[reference]
        competitors = {k: v for k, v in method_scores.items() if k != reference}
        n_tests = len(competitors)

        results: Dict[str, Dict[str, Any]] = {}

        if correction == "bonferroni":
            adj_alpha = alpha / max(n_tests, 1)
            for name, scores in competitors.items():
                result = self.paired_test(ref_scores, scores, test=test, alpha=adj_alpha)
                result["correction"] = "bonferroni"
                result["adjusted_alpha"] = adj_alpha
                results[name] = result

        elif correction == "holm":
            raw_results: List[Tuple[str, Dict[str, Any]]] = []
            for name, scores in competitors.items():
                result = self.paired_test(ref_scores, scores, test=test, alpha=alpha)
                raw_results.append((name, result))

            raw_results.sort(key=lambda x: x[1]["p_value"])

            for rank, (name, result) in enumerate(raw_results):
                adj_alpha = alpha / (n_tests - rank)
                result["significant"] = result["p_value"] < adj_alpha
                result["correction"] = "holm"
                result["adjusted_alpha"] = adj_alpha
                results[name] = result

        else:
            for name, scores in competitors.items():
                result = self.paired_test(ref_scores, scores, test=test, alpha=alpha)
                result["correction"] = "none"
                results[name] = result

        return results


# =====================================================================
# Reporting
# =====================================================================


class BenchmarkReporter:
    """Generate benchmark reports.

    Examples
    --------
    >>> reporter = BenchmarkReporter(output_dir="results")
    >>> reporter.summary_table(aggregated_results)
    >>> reporter.latex_table(aggregated_results)
    """

    def __init__(self, output_dir: str | Path = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def summary_table(
        self,
        aggregated: Dict[str, Dict[str, Dict[str, Any]]],
        metrics: Optional[List[str]] = None,
    ) -> str:
        """Generate ASCII summary table.

        Parameters
        ----------
        aggregated : dict
            Scenario → metric → stats.
        metrics : list of str, optional
            Metrics to include (default: macro_f1, accuracy, tp_f1, wall_time).

        Returns
        -------
        str
            Formatted ASCII table.
        """
        if metrics is None:
            metrics = ["macro_f1", "accuracy", "tp_f1", "wall_time"]

        # Header
        col_width = 14
        header = f"{'Scenario':<15}"
        for m in metrics:
            header += f" {m:>{col_width}}"
        header += "\n" + "-" * len(header)

        lines = [header]

        for scenario, stats in sorted(aggregated.items()):
            if isinstance(stats, dict) and "error" in stats:
                lines.append(f"{scenario:<15}  [all failed]")
                continue

            row = f"{scenario:<15}"
            for m in metrics:
                if m in stats:
                    mean = stats[m].get("mean", float("nan"))
                    std = stats[m].get("std", 0.0)
                    if np.isfinite(mean):
                        row += f" {mean:>7.3f}±{std:<5.3f}"
                    else:
                        row += f" {'N/A':>{col_width}}"
                else:
                    row += f" {'—':>{col_width}}"
            lines.append(row)

        return "\n".join(lines)

    def latex_table(
        self,
        aggregated: Dict[str, Dict[str, Dict[str, Any]]],
        metrics: Optional[List[str]] = None,
        caption: str = "Benchmark results",
        label: str = "tab:benchmark",
    ) -> str:
        """Generate LaTeX table.

        Parameters
        ----------
        aggregated : dict
            Scenario → metric → stats.
        metrics : list of str, optional
            Metrics to include.
        caption : str
            Table caption.
        label : str
            LaTeX label.

        Returns
        -------
        str
            LaTeX table source.
        """
        if metrics is None:
            metrics = ["macro_f1", "accuracy", "tp_f1", "wall_time"]

        n_cols = 1 + len(metrics)
        col_spec = "l" + "c" * len(metrics)

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
        ]

        header_cells = ["Scenario"] + [m.replace("_", "\\_") for m in metrics]
        lines.append(" & ".join(header_cells) + " \\\\")
        lines.append("\\midrule")

        for scenario, stats in sorted(aggregated.items()):
            if isinstance(stats, dict) and "error" in stats:
                cells = [scenario.replace("_", "\\_")] + ["—"] * len(metrics)
            else:
                cells = [scenario.replace("_", "\\_")]
                for m in metrics:
                    if m in stats:
                        mean = stats[m].get("mean", float("nan"))
                        std = stats[m].get("std", 0.0)
                        if np.isfinite(mean):
                            cells.append(f"${mean:.3f} \\pm {std:.3f}$")
                        else:
                            cells.append("N/A")
                    else:
                        cells.append("—")

            lines.append(" & ".join(cells) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(lines)

    def save_json(
        self,
        records: Dict[str, List[RunRecord]],
        filename: str = "benchmark_results.json",
    ) -> Path:
        """Save all records as JSON.

        Parameters
        ----------
        records : dict
            Scenario → list of RunRecord.
        filename : str
            Output filename.

        Returns
        -------
        Path
            Output file path.
        """
        out_path = self.output_dir / filename

        def _to_serializable(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, set):
                return sorted(obj)
            return obj

        data: Dict[str, Any] = {}
        for scenario, recs in records.items():
            data[scenario] = []
            for rec in recs:
                rec_dict = asdict(rec)
                data[scenario].append(rec_dict)

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=_to_serializable)

        logger.info(f"Saved results to {out_path}")
        return out_path

    def comparison_table(
        self,
        comparison_results: Dict[str, Dict[str, Any]],
        metric_name: str = "macro_f1",
    ) -> str:
        """Format statistical comparison results.

        Parameters
        ----------
        comparison_results : dict
            Method → test result from StatisticalComparison.
        metric_name : str
            Metric being compared.

        Returns
        -------
        str
            Formatted table.
        """
        lines = [
            f"Statistical Comparison ({metric_name})",
            "=" * 70,
            f"{'Method':<15} {'Mean Diff':>10} {'p-value':>10} {'Sig':>5} {'Effect':>10}",
            "-" * 70,
        ]

        for method, result in sorted(comparison_results.items()):
            sig = "***" if result.get("significant") else ""
            d = result.get("cohens_d", 0.0)
            if abs(d) >= 0.8:
                eff = "large"
            elif abs(d) >= 0.5:
                eff = "medium"
            elif abs(d) >= 0.2:
                eff = "small"
            else:
                eff = "negl."

            lines.append(
                f"{method:<15} {result.get('mean_diff', 0.0):>10.4f} "
                f"{result.get('p_value', 1.0):>10.4f} {sig:>5} {eff:>10}"
            )

        return "\n".join(lines)

    def full_report(
        self,
        all_records: Dict[str, List[RunRecord]],
        save: bool = True,
    ) -> str:
        """Generate full benchmark report.

        Parameters
        ----------
        all_records : dict
            Scenario → RunRecords.
        save : bool
            Whether to save to files.

        Returns
        -------
        str
            Full report text.
        """
        aggregator = ResultAggregator()
        aggregated = aggregator.aggregate_multi(all_records)

        sections = [
            "=" * 70,
            "CPA Benchmark Report",
            "=" * 70,
            "",
            "Summary",
            "-" * 70,
            self.summary_table(aggregated),
            "",
        ]

        # Per-scenario detail
        for scenario, records in sorted(all_records.items()):
            sections.append(f"\nScenario: {scenario}")
            sections.append("-" * 40)

            valid = [r for r in records if "error" not in r.metadata]
            failed = len(records) - len(valid)

            sections.append(
                f"  Replications: {len(valid)} successful, {failed} failed"
            )

            if valid:
                times = [r.wall_time for r in valid]
                sections.append(
                    f"  Wall time: {np.mean(times):.2f} ± {np.std(times):.2f}s"
                )

                f1s = [
                    r.classification_metrics.get("macro_f1", 0.0)
                    for r in valid
                    if r.classification_metrics
                ]
                if f1s:
                    sections.append(
                        f"  Macro F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}"
                    )

        report = "\n".join(sections)

        if save:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            report_path = self.output_dir / "benchmark_report.txt"
            report_path.write_text(report)

            latex = self.latex_table(aggregated)
            latex_path = self.output_dir / "benchmark_table.tex"
            latex_path.write_text(latex)

            self.save_json(all_records)

            logger.info(f"Report saved to {self.output_dir}")

        return report
