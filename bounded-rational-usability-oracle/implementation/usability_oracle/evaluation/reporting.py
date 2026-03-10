"""
usability_oracle.evaluation.reporting — Evaluation report generation.

Produces human-readable summaries, LaTeX tables, plotting commands, and
statistical significance tests for evaluation results.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# EvaluationReporter
# ---------------------------------------------------------------------------

class EvaluationReporter:
    """Generate formatted evaluation reports."""

    def __init__(self, decimal_places: int = 4) -> None:
        self._dp = decimal_places

    # ------------------------------------------------------------------
    # Main report
    # ------------------------------------------------------------------

    def report(self, results: dict[str, Any]) -> str:
        """Generate a plain-text evaluation report from *results*.

        Expected keys in *results*:
          - "system": dict with accuracy, precision, recall, f1, roc_auc
          - "baselines": dict[str, dict] per baseline
          - "ordinal": OrdinalResult-like dict
          - "ablation": AblationResult-like dict
        """
        sections: list[str] = []
        sections.append("=" * 70)
        sections.append("  Usability Oracle — Evaluation Report")
        sections.append("=" * 70)

        # System performance
        sys = results.get("system", {})
        if sys:
            sections.append("\n## System Performance")
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                val = sys.get(key)
                if val is not None:
                    sections.append(f"  {key:20s}: {val:.{self._dp}f}")

        # Baselines
        baselines = results.get("baselines", {})
        if baselines:
            sections.append("\n## Baseline Comparison")
            sections.append(f"  {'Baseline':<20s} {'Accuracy':>10s} {'F1':>10s}")
            sections.append("  " + "-" * 42)
            for name, metrics in baselines.items():
                acc = metrics.get("accuracy", 0.0)
                f1 = metrics.get("f1", 0.0)
                sections.append(f"  {name:<20s} {acc:>10.{self._dp}f} {f1:>10.{self._dp}f}")

        # Ordinal
        ordinal = results.get("ordinal", {})
        if ordinal:
            sections.append("\n## Ordinal Agreement")
            for key in ("spearman_rho", "kendall_tau", "concordance_rate"):
                val = ordinal.get(key)
                if val is not None:
                    sections.append(f"  {key:20s}: {val:.{self._dp}f}")
            ci_lo = ordinal.get("ci_lower", 0)
            ci_hi = ordinal.get("ci_upper", 0)
            sections.append(f"  {'95% CI':20s}: [{ci_lo:.{self._dp}f}, {ci_hi:.{self._dp}f}]")

        # Ablation
        ablation = results.get("ablation", {})
        if ablation:
            sections.append("\n## Ablation Study")
            sections.append(f"  Full score: {ablation.get('full_score', 0):.{self._dp}f}")
            contributions = ablation.get("contributions", {})
            for comp in sorted(contributions, key=lambda k: contributions[k], reverse=True):
                sections.append(f"  {comp:30s}  Δ = {contributions[comp]:+.{self._dp}f}")

        # Significance
        sig = results.get("significance", {})
        if sig:
            sections.append("\n## Statistical Significance")
            for test_name, test_result in sig.items():
                p = test_result.get("p_value", 1.0)
                star = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
                sections.append(f"  {test_name:30s}  p = {p:.4g}{star}")

        sections.append("\n" + "=" * 70)
        return "\n".join(sections)

    # ------------------------------------------------------------------
    # LaTeX table
    # ------------------------------------------------------------------

    def _latex_table(self, metrics: dict[str, dict[str, float]]) -> str:
        """Generate a LaTeX table comparing system vs baselines.

        *metrics* maps method name → {metric_name: value}.
        """
        cols = set()
        for v in metrics.values():
            cols.update(v.keys())
        cols_sorted = sorted(cols)

        lines: list[str] = []
        col_spec = "l" + "r" * len(cols_sorted)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        header = "Method & " + " & ".join(c.replace("_", "\\_") for c in cols_sorted) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        for method, vals in metrics.items():
            row_vals = []
            for c in cols_sorted:
                v = vals.get(c)
                if v is not None:
                    row_vals.append(f"{v:.{self._dp}f}")
                else:
                    row_vals.append("--")
            line = method.replace("_", "\\_") + " & " + " & ".join(row_vals) + " \\\\"
            lines.append(line)

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # ROC plot command
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_roc(fpr: Sequence[float], tpr: Sequence[float], auc_val: float) -> str:
        """Return matplotlib commands to plot the ROC curve."""
        lines = [
            "import matplotlib.pyplot as plt",
            "",
            f"fpr = {list(fpr)}",
            f"tpr = {list(tpr)}",
            f"auc_val = {auc_val:.4f}",
            "",
            "fig, ax = plt.subplots(figsize=(6, 6))",
            "ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Oracle (AUC = {auc_val:.3f})')",
            "ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')",
            "ax.set_xlabel('False Positive Rate')",
            "ax.set_ylabel('True Positive Rate')",
            "ax.set_title('ROC Curve — Usability Regression Detection')",
            "ax.legend(loc='lower right')",
            "ax.set_xlim([0, 1])",
            "ax.set_ylim([0, 1])",
            "ax.set_aspect('equal')",
            "plt.tight_layout()",
            "plt.savefig('roc_curve.pdf', dpi=300)",
            "plt.show()",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Significance tests
    # ------------------------------------------------------------------

    def _significance_tests(
        self,
        system_results: Sequence[float],
        baseline_results: Sequence[float],
    ) -> dict[str, Any]:
        """Run paired significance tests between system and baseline.

        Returns:
            Dict with test names → {statistic, p_value, significant}.
        """
        sys_arr = np.asarray(system_results, dtype=float)
        base_arr = np.asarray(baseline_results, dtype=float)
        n = min(len(sys_arr), len(base_arr))
        sys_arr = sys_arr[:n]
        base_arr = base_arr[:n]

        output: dict[str, Any] = {}

        # Paired t-test
        if _HAS_SCIPY and n >= 2:
            t_stat, p_value = sp_stats.ttest_rel(sys_arr, base_arr)
            output["paired_t_test"] = {
                "statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": float(p_value) < 0.05,
            }

        # Wilcoxon signed-rank test
        if _HAS_SCIPY and n >= 6:
            diffs = sys_arr - base_arr
            nonzero = diffs[diffs != 0]
            if len(nonzero) > 0:
                w_stat, p_value = sp_stats.wilcoxon(nonzero)
                output["wilcoxon"] = {
                    "statistic": float(w_stat),
                    "p_value": float(p_value),
                    "significant": float(p_value) < 0.05,
                }

        # Effect size (Cohen's d)
        diff = sys_arr - base_arr
        if n >= 2:
            d_mean = float(np.mean(diff))
            d_std = float(np.std(diff, ddof=1)) if n > 1 else 1.0
            cohens_d = d_mean / d_std if d_std > 0 else 0.0
            output["cohens_d"] = {
                "statistic": cohens_d,
                "interpretation": (
                    "large" if abs(cohens_d) >= 0.8
                    else "medium" if abs(cohens_d) >= 0.5
                    else "small" if abs(cohens_d) >= 0.2
                    else "negligible"
                ),
            }

        return output

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _summary_statistics(results: Sequence[float]) -> dict[str, float]:
        """Compute descriptive statistics for a sequence of scores."""
        arr = np.asarray(results, dtype=float)
        if len(arr) == 0:
            return {}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "n": len(arr),
        }
