"""ROC and detection performance visualization."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


class ROCPlotter:
    """ROC and detection performance curves."""

    def __init__(self, figsize: Tuple[int, int] = (8, 8), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _save_or_return(self, fig: plt.Figure, save_path: Optional[str]) -> plt.Figure:
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    @staticmethod
    def _binary_curve_points(y_true: np.ndarray, y_scores: np.ndarray):
        """Return (fpr, tpr, thresholds) sorted by descending threshold."""
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_scores = np.asarray(y_scores, dtype=float).ravel()
        desc = np.argsort(-y_scores)
        y_true = y_true[desc]
        y_scores = y_scores[desc]

        distinct_idx = np.where(np.diff(y_scores))[0]
        threshold_idx = np.concatenate([distinct_idx, [len(y_scores) - 1]])

        tps = np.cumsum(y_true)[threshold_idx]
        fps = (threshold_idx + 1) - tps

        total_pos = y_true.sum()
        total_neg = len(y_true) - total_pos

        tpr = np.concatenate([[0], tps / max(total_pos, 1)])
        fpr = np.concatenate([[0], fps / max(total_neg, 1)])
        thresholds = np.concatenate([[y_scores[0] + 1], y_scores[threshold_idx]])
        return fpr, tpr, thresholds

    @staticmethod
    def _precision_recall_points(y_true: np.ndarray, y_scores: np.ndarray):
        """Return (precision, recall, thresholds)."""
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_scores = np.asarray(y_scores, dtype=float).ravel()
        desc = np.argsort(-y_scores)
        y_true = y_true[desc]
        y_scores = y_scores[desc]

        distinct_idx = np.where(np.diff(y_scores))[0]
        threshold_idx = np.concatenate([distinct_idx, [len(y_scores) - 1]])

        tps = np.cumsum(y_true)[threshold_idx]
        predicted_pos = threshold_idx + 1

        total_pos = y_true.sum()
        precision = tps / predicted_pos
        recall = tps / max(total_pos, 1)

        # Prepend point (recall=0, precision=1)
        precision = np.concatenate([[1.0], precision])
        recall = np.concatenate([[0.0], recall])
        thresholds = np.concatenate([[y_scores[0] + 1], y_scores[threshold_idx]])
        return precision, recall, thresholds

    # ------------------------------------------------------------------
    # public: compute helpers
    # ------------------------------------------------------------------

    def compute_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute AUC using trapezoidal rule."""
        fpr, tpr, _ = self._binary_curve_points(y_true, y_scores)
        return float(np.trapz(tpr, fpr))

    def compute_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        criterion: str = "youden",
    ) -> float:
        """Find optimal classification threshold.

        Args:
            criterion: 'youden' (max TPR − FPR) or 'closest' (closest to (0,1)).
        """
        fpr, tpr, thresholds = self._binary_curve_points(y_true, y_scores)
        if criterion == "youden":
            idx = int(np.argmax(tpr - fpr))
        elif criterion == "closest":
            dist = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
            idx = int(np.argmin(dist))
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        return float(thresholds[idx])

    # ------------------------------------------------------------------
    # ROC curve
    # ------------------------------------------------------------------

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot ROC curve with AUC."""
        fpr, tpr, _ = self._binary_curve_points(y_true, y_scores)
        auc_val = float(np.trapz(tpr, fpr))

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.plot(fpr, tpr, linewidth=1.5, color=_COLORS[0], label=f"ROC (AUC = {auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.12, color=_COLORS[0])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # multiple ROC curves
    # ------------------------------------------------------------------

    def plot_multi_roc(
        self,
        curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title: str = "ROC Comparison",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot multiple ROC curves for comparison.

        Args:
            curves: {label: (y_true, y_scores)}.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        for idx, (label, (yt, ys)) in enumerate(curves.items()):
            fpr, tpr, _ = self._binary_curve_points(yt, ys)
            auc_val = float(np.trapz(tpr, fpr))
            ax.plot(fpr, tpr, linewidth=1.3, color=_COLORS[idx % len(_COLORS)],
                    label=f"{label} (AUC={auc_val:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # precision-recall
    # ------------------------------------------------------------------

    def plot_precision_recall(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "Precision–Recall Curve",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Precision-recall curve."""
        precision, recall, _ = self._precision_recall_points(y_true, y_scores)
        ap = float(-np.trapz(precision, recall))

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.step(recall, precision, where="post", linewidth=1.3, color=_COLORS[0],
                label=f"AP = {ap:.3f}")
        ax.fill_between(recall, precision, step="post", alpha=0.12, color=_COLORS[0])

        baseline = np.asarray(y_true, dtype=int).mean()
        ax.axhline(baseline, color="grey", linestyle="--", linewidth=0.7, label=f"Baseline = {baseline:.3f}")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # detection power
    # ------------------------------------------------------------------

    def plot_detection_power(
        self,
        effect_sizes: np.ndarray,
        power_values: np.ndarray,
        target_power: float = 0.80,
        title: str = "Detection Power",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot detection power vs effect size."""
        effect_sizes = np.asarray(effect_sizes, dtype=float).ravel()
        power_values = np.asarray(power_values, dtype=float).ravel()

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.plot(effect_sizes, power_values, "o-", linewidth=1.3, color=_COLORS[0], markersize=4)
        ax.axhline(target_power, color="#d62728", linestyle="--", linewidth=1, label=f"Target = {target_power}")
        ax.fill_between(effect_sizes, power_values, alpha=0.10, color=_COLORS[0])
        ax.set_xlabel("Effect Size")
        ax.set_ylabel("Power (1 − β)")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # type I / type II tradeoff
    # ------------------------------------------------------------------

    def plot_type_error_tradeoff(
        self,
        alphas: np.ndarray,
        type_i: np.ndarray,
        type_ii: np.ndarray,
        title: str = "Type I vs Type II Error Tradeoff",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot type I vs type II error tradeoff."""
        alphas = np.asarray(alphas, dtype=float).ravel()
        type_i = np.asarray(type_i, dtype=float).ravel()
        type_ii = np.asarray(type_ii, dtype=float).ravel()

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.plot(alphas, type_i, "o-", label="Type I (α)", color=_COLORS[0], markersize=4, linewidth=1.2)
        ax.plot(alphas, type_ii, "s-", label="Type II (β)", color=_COLORS[1], markersize=4, linewidth=1.2)
        total = np.asarray(type_i) + np.asarray(type_ii)
        ax.plot(alphas, total, "^--", label="Total error", color="grey", markersize=4, linewidth=0.9)
        ax.set_xlabel("Significance Level (α)")
        ax.set_ylabel("Error Rate")
        ax.set_title(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # DET curve
    # ------------------------------------------------------------------

    def plot_det_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "Detection Error Tradeoff",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Detection error tradeoff (DET) curve."""
        fpr, tpr, _ = self._binary_curve_points(y_true, y_scores)
        fnr = 1.0 - tpr

        # Avoid log(0)
        mask = (fpr > 0) & (fnr > 0)
        fpr_plot = fpr[mask]
        fnr_plot = fnr[mask]

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.plot(fpr_plot, fnr_plot, linewidth=1.4, color=_COLORS[0])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("False Negative Rate")
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # calibration curve
    # ------------------------------------------------------------------

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        title: str = "Calibration Curve",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Calibration curve (reliability diagram)."""
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_prob = np.asarray(y_prob, dtype=float).ravel()

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means = np.zeros(n_bins)
        bin_true_fracs = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for b in range(n_bins):
            mask = (y_prob >= bin_edges[b]) & (y_prob < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = (y_prob >= bin_edges[b]) & (y_prob <= bin_edges[b + 1])
            cnt = mask.sum()
            bin_counts[b] = cnt
            if cnt > 0:
                bin_means[b] = y_prob[mask].mean()
                bin_true_fracs[b] = y_true[mask].mean()

        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] + 2), dpi=self.dpi,
                                 gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

        # Reliability
        axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.7, label="Perfectly calibrated")
        valid = bin_counts > 0
        axes[0].plot(bin_means[valid], bin_true_fracs[valid], "o-", color=_COLORS[0], linewidth=1.3, markersize=5,
                     label="Model")
        axes[0].set_ylabel("Fraction of positives")
        axes[0].set_title(title)
        axes[0].legend(fontsize=9)
        axes[0].set_xlim(-0.02, 1.02)
        axes[0].set_ylim(-0.02, 1.05)

        # Histogram
        axes[1].bar(bin_means, bin_counts, width=1.0 / n_bins * 0.8, color=_COLORS[0], alpha=0.5, edgecolor="white")
        axes[1].set_xlabel("Mean predicted probability")
        axes[1].set_ylabel("Count")

        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # lift chart
    # ------------------------------------------------------------------

    def plot_lift_chart(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "Lift Chart",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Lift chart."""
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_scores = np.asarray(y_scores, dtype=float).ravel()

        order = np.argsort(-y_scores)
        y_sorted = y_true[order]
        n = len(y_sorted)
        baseline_rate = y_true.mean()

        cum_pos = np.cumsum(y_sorted)
        fractions = np.arange(1, n + 1) / n
        cum_rate = cum_pos / np.arange(1, n + 1)
        lift = cum_rate / baseline_rate if baseline_rate > 0 else cum_rate

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.plot(fractions, lift, linewidth=1.3, color=_COLORS[0], label="Model lift")
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7, label="Baseline")
        ax.set_xlabel("Fraction of Population")
        ax.set_ylabel("Lift")
        ax.set_title(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)
