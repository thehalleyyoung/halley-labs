"""Visualization of statistical test results."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Any


_PASS_COLOR = "#2ca02c"
_FAIL_COLOR = "#d62728"
_WARN_COLOR = "#ff7f0e"
_NEUTRAL_COLOR = "#1f77b4"


class TestResultPlotter:
    """Visualize hypothesis test results."""

    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi

    def _save_or_return(self, fig: plt.Figure, save_path: Optional[str]) -> plt.Figure:
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # p-value distribution
    # ------------------------------------------------------------------

    def plot_p_value_distribution(
        self,
        p_values: np.ndarray,
        alpha: float = 0.05,
        title: str = "P-Value Distribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot distribution of p-values with significance threshold."""
        p_values = np.asarray(p_values, dtype=float).ravel()

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        bins = np.linspace(0, 1, 41)
        counts, edges, patches = ax.hist(
            p_values, bins=bins, edgecolor="white", linewidth=0.4, color=_NEUTRAL_COLOR, alpha=0.8,
        )
        # Colour bars below alpha
        for patch, left_edge in zip(patches, edges[:-1]):
            if left_edge + (edges[1] - edges[0]) <= alpha:
                patch.set_facecolor(_FAIL_COLOR)

        ax.axvline(alpha, color=_FAIL_COLOR, linestyle="--", linewidth=1.2, label=f"α = {alpha}")
        # Uniform reference
        ax.axhline(len(p_values) / len(bins), color="grey", linestyle=":", linewidth=0.8, label="Uniform")
        n_sig = int(np.sum(p_values < alpha))
        ax.set_xlabel("p-value")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}  ({n_sig}/{len(p_values)} significant)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # test-statistic histogram
    # ------------------------------------------------------------------

    def plot_test_statistic_histogram(
        self,
        statistics: np.ndarray,
        null_distribution: Optional[np.ndarray] = None,
        observed: Optional[float] = None,
        title: str = "Test Statistic Distribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot test statistic with null distribution overlay."""
        statistics = np.asarray(statistics, dtype=float).ravel()

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if null_distribution is not None:
            null_distribution = np.asarray(null_distribution, dtype=float).ravel()
            ax.hist(null_distribution, bins=60, density=True, alpha=0.45, color="grey", label="Null", edgecolor="white", linewidth=0.3)

        ax.hist(statistics, bins=60, density=True, alpha=0.65, color=_NEUTRAL_COLOR, label="Observed", edgecolor="white", linewidth=0.3)

        if observed is not None:
            ax.axvline(observed, color=_FAIL_COLOR, linestyle="--", linewidth=1.3, label=f"Observed = {observed:.3f}")

        ax.set_xlabel("Test Statistic")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # confidence intervals (forest plot)
    # ------------------------------------------------------------------

    def plot_confidence_intervals(
        self,
        estimates: List[float],
        intervals: List[Tuple[float, float]],
        labels: List[str],
        null_value: float = 0.0,
        title: str = "Confidence Intervals",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Forest plot of confidence intervals."""
        n = len(estimates)
        fig, ax = plt.subplots(figsize=(self.figsize[0], max(4, 0.45 * n)), dpi=self.dpi)

        y_pos = np.arange(n)
        for i, (est, (lo, hi), lab) in enumerate(zip(estimates, intervals, labels)):
            color = _FAIL_COLOR if lo > null_value or hi < null_value else _NEUTRAL_COLOR
            ax.plot([lo, hi], [i, i], color=color, linewidth=2)
            ax.plot(est, i, "o", color=color, markersize=6)

        ax.axvline(null_value, color="grey", linestyle="--", linewidth=0.8, label=f"Null = {null_value}")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Estimate")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # alpha spending
    # ------------------------------------------------------------------

    def plot_alpha_spending(
        self,
        tiers: List[str],
        allocated: List[float],
        spent: List[float],
        title: str = "Alpha Budget Allocation",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize alpha spending across tiers."""
        n = len(tiers)
        x = np.arange(n)
        width = 0.35

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        bars_alloc = ax.bar(x - width / 2, allocated, width, label="Allocated", color=_NEUTRAL_COLOR, edgecolor="white")
        bars_spent = ax.bar(x + width / 2, spent, width, label="Spent", color=_FAIL_COLOR, alpha=0.75, edgecolor="white")

        # Annotate values
        for bar in bars_alloc:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.4f}",
                    ha="center", va="bottom", fontsize=7)
        for bar in bars_spent:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.4f}",
                    ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(tiers, fontsize=9)
        ax.set_ylabel("Alpha")
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # tier summary
    # ------------------------------------------------------------------

    def plot_tier_summary(
        self,
        tier_results: List[Dict[str, Any]],
        title: str = "Tier Summary",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Summary visualisation of all tier results.

        Each dict should have keys: tier, reject (bool), p_value (float), statistic (float).
        """
        n = len(tier_results)
        fig, axes = plt.subplots(1, 3, figsize=(self.figsize[0] + 4, max(4, 0.5 * n)), dpi=self.dpi)

        tiers = [r.get("tier", f"T{i}") for i, r in enumerate(tier_results)]
        y_pos = np.arange(n)

        # Panel 1 – p-values
        p_vals = [r.get("p_value", np.nan) for r in tier_results]
        colors_p = [_FAIL_COLOR if p < 0.05 else _NEUTRAL_COLOR for p in p_vals]
        axes[0].barh(y_pos, p_vals, color=colors_p, edgecolor="white")
        axes[0].axvline(0.05, color="grey", linestyle="--", linewidth=0.8)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(tiers, fontsize=8)
        axes[0].set_xlabel("p-value")
        axes[0].set_title("P-Values")
        axes[0].invert_yaxis()

        # Panel 2 – test statistics
        stats = [r.get("statistic", 0.0) for r in tier_results]
        axes[1].barh(y_pos, stats, color=_NEUTRAL_COLOR, edgecolor="white")
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(tiers, fontsize=8)
        axes[1].set_xlabel("Statistic")
        axes[1].set_title("Test Statistics")
        axes[1].invert_yaxis()

        # Panel 3 – reject / fail
        rejects = [r.get("reject", False) for r in tier_results]
        colors_r = [_FAIL_COLOR if rej else _PASS_COLOR for rej in rejects]
        axes[2].barh(y_pos, [1] * n, color=colors_r, edgecolor="white")
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels(tiers, fontsize=8)
        axes[2].set_xlim(0, 1.5)
        axes[2].set_title("Reject Null?")
        axes[2].invert_yaxis()
        for i, rej in enumerate(rejects):
            axes[2].text(0.5, i, "REJECT" if rej else "FAIL TO REJECT", ha="center", va="center", fontsize=7, color="white", fontweight="bold")

        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # bootstrap distribution
    # ------------------------------------------------------------------

    def plot_bootstrap_distribution(
        self,
        bootstrap_samples: np.ndarray,
        observed: float,
        ci: Tuple[float, float],
        title: str = "Bootstrap Distribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot bootstrap sampling distribution."""
        bootstrap_samples = np.asarray(bootstrap_samples, dtype=float).ravel()

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.hist(bootstrap_samples, bins=80, density=True, alpha=0.65, color=_NEUTRAL_COLOR, edgecolor="white", linewidth=0.3)
        ax.axvline(observed, color=_FAIL_COLOR, linewidth=1.3, label=f"Observed = {observed:.4f}")
        ax.axvline(ci[0], color=_WARN_COLOR, linestyle="--", linewidth=1, label=f"CI low = {ci[0]:.4f}")
        ax.axvline(ci[1], color=_WARN_COLOR, linestyle="--", linewidth=1, label=f"CI high = {ci[1]:.4f}")
        ax.axvspan(ci[0], ci[1], alpha=0.10, color=_WARN_COLOR)

        ax.set_xlabel("Statistic")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # multiple testing correction
    # ------------------------------------------------------------------

    def plot_multiple_testing_correction(
        self,
        raw_p: np.ndarray,
        adjusted_p: np.ndarray,
        alpha: float = 0.05,
        title: str = "Multiple Testing Correction",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Compare raw vs adjusted p-values."""
        raw_p = np.asarray(raw_p, dtype=float).ravel()
        adjusted_p = np.asarray(adjusted_p, dtype=float).ravel()
        order = np.argsort(raw_p)
        raw_sorted = raw_p[order]
        adj_sorted = adjusted_p[order]
        n = len(raw_sorted)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.plot(range(n), raw_sorted, "o-", markersize=3, linewidth=0.8, label="Raw p", color=_NEUTRAL_COLOR)
        ax.plot(range(n), adj_sorted, "s-", markersize=3, linewidth=0.8, label="Adjusted p", color=_WARN_COLOR)
        ax.axhline(alpha, color=_FAIL_COLOR, linestyle="--", linewidth=1, label=f"α = {alpha}")
        ax.set_xlabel("Test (sorted by raw p)")
        ax.set_ylabel("p-value")
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # effect sizes
    # ------------------------------------------------------------------

    def plot_effect_sizes(
        self,
        effects: List[float],
        labels: List[str],
        ci: Optional[List[Tuple[float, float]]] = None,
        title: str = "Effect Sizes",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot effect sizes with CIs."""
        n = len(effects)
        fig, ax = plt.subplots(figsize=(self.figsize[0], max(4, 0.45 * n)), dpi=self.dpi)
        y_pos = np.arange(n)

        colors = [_FAIL_COLOR if e > 0 else _PASS_COLOR for e in effects]
        ax.barh(y_pos, effects, color=colors, edgecolor="white", height=0.6)

        if ci is not None:
            for i, (lo, hi) in enumerate(ci):
                ax.plot([lo, hi], [i, i], color="black", linewidth=1.2)
                ax.plot([lo, lo], [i - 0.15, i + 0.15], color="black", linewidth=1.2)
                ax.plot([hi, hi], [i - 0.15, i + 0.15], color="black", linewidth=1.2)

        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Effect Size")
        ax.set_title(title)
        ax.invert_yaxis()
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # verdict gauge
    # ------------------------------------------------------------------

    def plot_verdict_gauge(
        self,
        confidence: float,
        verdict: str,
        title: str = "Verdict Confidence",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Gauge / dial visualisation of verdict confidence.

        Args:
            confidence: Value in [0, 1].
            verdict: E.g. "COLLUSIVE", "COMPETITIVE", "INCONCLUSIVE".
        """
        confidence = float(np.clip(confidence, 0.0, 1.0))

        fig, ax = plt.subplots(figsize=(6, 4), dpi=self.dpi, subplot_kw={"projection": "polar"})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Draw arc background (semicircle from -π/2 to π/2)
        theta_bg = np.linspace(-np.pi / 2, np.pi / 2, 200)
        ax.fill_between(theta_bg, 0.6, 1.0, color="#eeeeee", alpha=0.5)

        # Colour bands: green -> yellow -> red
        n_seg = 100
        thetas = np.linspace(-np.pi / 2, np.pi / 2, n_seg + 1)
        for i in range(n_seg):
            frac = i / n_seg
            if frac < 0.33:
                c = _PASS_COLOR
            elif frac < 0.66:
                c = _WARN_COLOR
            else:
                c = _FAIL_COLOR
            ax.fill_between([thetas[i], thetas[i + 1]], 0.6, 1.0, color=c, alpha=0.35)

        # Needle
        needle_angle = -np.pi / 2 + confidence * np.pi
        ax.plot([needle_angle, needle_angle], [0, 0.95], color="black", linewidth=2)
        ax.plot(needle_angle, 0.95, "o", color="black", markersize=5)

        ax.set_ylim(0, 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}\n{verdict}  ({confidence:.0%})", fontsize=11, pad=20)

        fig.tight_layout()
        return self._save_or_return(fig, save_path)
