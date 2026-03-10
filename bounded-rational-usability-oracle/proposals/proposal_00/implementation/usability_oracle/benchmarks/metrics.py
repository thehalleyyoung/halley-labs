"""
usability_oracle.benchmarks.metrics — Statistical metrics for benchmark evaluation.

Provides accuracy, precision, recall, F1, confusion matrices, timing
statistics, scalability analysis, and rank correlations for comparing
model outputs against ground truth.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from usability_oracle.core.enums import RegressionVerdict
from usability_oracle.benchmarks.suite import BenchmarkResult


# ---------------------------------------------------------------------------
# BenchmarkMetrics
# ---------------------------------------------------------------------------

class BenchmarkMetrics:
    """Compute evaluation metrics over a list of :class:`BenchmarkResult`."""

    # ------------------------------------------------------------------
    # Classification metrics
    # ------------------------------------------------------------------

    @staticmethod
    def accuracy(results: list[BenchmarkResult]) -> float:
        """Fraction of cases where actual == expected."""
        if not results:
            return 0.0
        correct = sum(1 for r in results if r.correct)
        return correct / len(results)

    @staticmethod
    def precision(results: list[BenchmarkResult], positive_class: str = "regression") -> float:
        """Precision for *positive_class*."""
        tp = sum(
            1 for r in results
            if r.actual_verdict.value == positive_class and r.expected_verdict.value == positive_class
        )
        fp = sum(
            1 for r in results
            if r.actual_verdict.value == positive_class and r.expected_verdict.value != positive_class
        )
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @staticmethod
    def recall(results: list[BenchmarkResult], positive_class: str = "regression") -> float:
        """Recall (sensitivity) for *positive_class*."""
        tp = sum(
            1 for r in results
            if r.actual_verdict.value == positive_class and r.expected_verdict.value == positive_class
        )
        fn = sum(
            1 for r in results
            if r.actual_verdict.value != positive_class and r.expected_verdict.value == positive_class
        )
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def f1_score(results: list[BenchmarkResult], positive_class: str = "regression") -> float:
        """F1 score for *positive_class*."""
        p = BenchmarkMetrics.precision(results, positive_class)
        r = BenchmarkMetrics.recall(results, positive_class)
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    @staticmethod
    def confusion_matrix(results: list[BenchmarkResult]) -> np.ndarray:
        """Build a confusion matrix (rows = expected, columns = actual)."""
        labels = sorted(
            {r.expected_verdict for r in results} | {r.actual_verdict for r in results},
            key=lambda v: v.value,
        )
        idx = {v: i for i, v in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for r in results:
            cm[idx[r.expected_verdict], idx[r.actual_verdict]] += 1
        return cm

    # ------------------------------------------------------------------
    # Timing statistics
    # ------------------------------------------------------------------

    @staticmethod
    def timing_statistics(results: list[BenchmarkResult]) -> dict[str, float]:
        """Compute timing statistics across all results."""
        if not results:
            return {}
        times = np.array([r.timing for r in results])
        return {
            "mean": float(np.mean(times)),
            "median": float(np.median(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "p50": float(np.percentile(times, 50)),
            "p90": float(np.percentile(times, 90)),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)),
            "total": float(np.sum(times)),
        }

    # ------------------------------------------------------------------
    # Scalability
    # ------------------------------------------------------------------

    @staticmethod
    def scalability_curve(results: list[BenchmarkResult]) -> list[tuple[int, float]]:
        """Estimate (size, time) curve from results with 'n_elements' metadata.

        Returns a sorted list of (size, median_time) tuples.
        """
        buckets: dict[int, list[float]] = {}
        for r in results:
            n = r.metadata.get("n_elements")
            if n is not None:
                buckets.setdefault(int(n), []).append(r.timing)
        curve: list[tuple[int, float]] = []
        for size in sorted(buckets):
            curve.append((size, float(np.median(buckets[size]))))
        return curve

    # ------------------------------------------------------------------
    # Rank correlation
    # ------------------------------------------------------------------

    @staticmethod
    def rank_correlation(model_rankings: list[float], human_rankings: list[float]) -> float:
        """Spearman's rank-correlation coefficient (ρ) between two rankings.

        Uses the standard formula:

            ρ = 1 − (6 Σd²) / (n(n²−1))

        where d is the difference between ranks.
        """
        n = min(len(model_rankings), len(human_rankings))
        if n < 2:
            return 0.0
        # Convert values to ranks
        m_ranks = _to_ranks(model_rankings[:n])
        h_ranks = _to_ranks(human_rankings[:n])
        d_sq = sum((m - h) ** 2 for m, h in zip(m_ranks, h_ranks))
        rho = 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1))
        return rho

    # ------------------------------------------------------------------
    # Per-category breakdown
    # ------------------------------------------------------------------

    @staticmethod
    def per_category_accuracy(results: list[BenchmarkResult]) -> dict[str, float]:
        """Compute accuracy broken down by category metadata."""
        cats: dict[str, list[bool]] = {}
        for r in results:
            cat = r.metadata.get("category", "unknown")
            cats.setdefault(cat, []).append(r.correct)
        return {cat: sum(vals) / len(vals) for cat, vals in cats.items() if vals}

    # ------------------------------------------------------------------
    # Per-bottleneck-type metrics
    # ------------------------------------------------------------------

    @staticmethod
    def per_bottleneck_metrics(
        results: list[BenchmarkResult],
    ) -> dict[str, dict[str, float]]:
        """Metrics broken down by bottleneck type stored in metadata."""
        buckets: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            bt = r.metadata.get("bottleneck_type", "unknown")
            buckets.setdefault(bt, []).append(r)

        out: dict[str, dict[str, float]] = {}
        for bt, bucket_results in buckets.items():
            n = len(bucket_results)
            n_correct = sum(1 for r in bucket_results if r.correct)
            times = np.array([r.timing for r in bucket_results])
            out[bt] = {
                "count": float(n),
                "accuracy": n_correct / n if n > 0 else 0.0,
                "mean_time": float(np.mean(times)),
                "max_time": float(np.max(times)),
            }
        return out

    # ------------------------------------------------------------------
    # Sensitivity (true positive) per severity bucket
    # ------------------------------------------------------------------

    @staticmethod
    def sensitivity_by_severity(
        results: list[BenchmarkResult],
        n_bins: int = 5,
    ) -> dict[str, dict[str, float]]:
        """Compute recall/sensitivity stratified by mutation severity."""
        # Get severity from metadata
        severity_results: list[tuple[float, bool]] = []
        for r in results:
            sev = r.metadata.get("severity")
            if sev is not None:
                severity_results.append((float(sev), r.correct))

        if not severity_results:
            return {}

        edges = np.linspace(0, 1, n_bins + 1)
        out: dict[str, dict[str, float]] = {}
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            label = f"{lo:.2f}-{hi:.2f}"
            bucket = [c for s, c in severity_results if lo <= s < hi or (b == n_bins - 1 and s == hi)]
            if bucket:
                out[label] = {
                    "count": float(len(bucket)),
                    "accuracy": sum(bucket) / len(bucket),
                }
        return out

    # ------------------------------------------------------------------
    # Matthews correlation coefficient
    # ------------------------------------------------------------------

    @staticmethod
    def matthews_correlation(
        results: list[BenchmarkResult],
        positive_class: Any = None,
    ) -> float:
        """Matthews Correlation Coefficient from benchmark results."""
        preds = []
        labels = []
        for r in results:
            p = 1 if r.actual_verdict == positive_class else 0
            l = 1 if r.expected_verdict == positive_class else 0
            preds.append(p)
            labels.append(l)

        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)

        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        if denom < 1e-12:
            return 0.0
        return float(tp * tn - fp * fn) / denom

    # ------------------------------------------------------------------
    # Cohen's kappa
    # ------------------------------------------------------------------

    @staticmethod
    def cohens_kappa(results: list[BenchmarkResult]) -> float:
        """Cohen's kappa: agreement between expected and actual verdicts."""
        n = len(results)
        if n == 0:
            return 0.0
        p_o = sum(1 for r in results if r.correct) / n

        all_verdicts = sorted(
            {r.expected_verdict for r in results} | {r.actual_verdict for r in results},
            key=lambda v: v.value,
        )
        p_e = 0.0
        for v in all_verdicts:
            p_expected = sum(1 for r in results if r.expected_verdict == v) / n
            p_actual = sum(1 for r in results if r.actual_verdict == v) / n
            p_e += p_expected * p_actual

        if abs(1.0 - p_e) < 1e-12:
            return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
        return (p_o - p_e) / (1.0 - p_e)

    # ------------------------------------------------------------------
    # Scalability fitting
    # ------------------------------------------------------------------

    @staticmethod
    def scalability_fit(
        sizes: Sequence[float],
        times: Sequence[float],
    ) -> dict[str, float]:
        """Fit timing data to power-law: time ∝ size^α.

        Returns exponent and R² goodness-of-fit.
        """
        s = np.asarray(sizes, dtype=float)
        t = np.asarray(times, dtype=float)
        mask = (s > 0) & (t > 0)
        s, t = s[mask], t[mask]
        if len(s) < 2:
            return {"exponent": 0.0, "r_squared": 0.0}

        log_s = np.log(s)
        log_t = np.log(t)

        # Simple linear regression in log space
        n = len(log_s)
        sx = float(np.sum(log_s))
        sy = float(np.sum(log_t))
        sxx = float(np.sum(log_s ** 2))
        sxy = float(np.sum(log_s * log_t))

        denom = n * sxx - sx ** 2
        if abs(denom) < 1e-12:
            return {"exponent": 0.0, "r_squared": 0.0}

        alpha = (n * sxy - sx * sy) / denom
        intercept = (sy - alpha * sx) / n

        # R²
        predicted = alpha * log_s + intercept
        ss_res = float(np.sum((log_t - predicted) ** 2))
        ss_tot = float(np.sum((log_t - np.mean(log_t)) ** 2))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        return {
            "exponent": float(alpha),
            "intercept": float(intercept),
            "r_squared": float(r_sq),
        }

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    @staticmethod
    def full_summary(
        results: list[BenchmarkResult],
        positive_class: Any = None,
    ) -> dict[str, Any]:
        """Produce a comprehensive summary dict of all available metrics."""
        n = len(results)
        if n == 0:
            return {"error": "no results"}

        n_correct = sum(1 for r in results if r.correct)
        times = np.array([r.timing for r in results])

        return {
            "n_cases": n,
            "accuracy": n_correct / n,
            "cohens_kappa": BenchmarkMetrics.cohens_kappa(results),
            "mcc": BenchmarkMetrics.matthews_correlation(results, positive_class),
            "per_category": BenchmarkMetrics.per_category_accuracy(results),
            "per_bottleneck": BenchmarkMetrics.per_bottleneck_metrics(results),
            "timing": {
                "mean": float(np.mean(times)),
                "median": float(np.median(times)),
                "std": float(np.std(times)),
                "p95": float(np.percentile(times, 95)),
                "p99": float(np.percentile(times, 99)),
            },
        }
