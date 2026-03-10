"""
usability_oracle.evaluation.regression — Regression-detection metrics.

Provides ROC, AUC, precision-recall curves, and optimal-threshold
selection for evaluating the oracle's regression-detection capability
as a binary classifier.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# RegressionDetectionMetrics
# ---------------------------------------------------------------------------

class RegressionDetectionMetrics:
    """Binary classification metrics for regression detection.

    *labels* are binary (1 = regression, 0 = no regression).
    *scores* are continuous confidence scores from the oracle.
    """

    # ------------------------------------------------------------------
    # Basic rates
    # ------------------------------------------------------------------

    @staticmethod
    def true_positive_rate(
        predictions: Sequence[int],
        ground_truth: Sequence[int],
    ) -> float:
        """TPR = TP / (TP + FN)."""
        preds = np.asarray(predictions, dtype=int)
        truth = np.asarray(ground_truth, dtype=int)
        tp = int(np.sum((preds == 1) & (truth == 1)))
        fn = int(np.sum((preds == 0) & (truth == 1)))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def false_positive_rate(
        predictions: Sequence[int],
        ground_truth: Sequence[int],
    ) -> float:
        """FPR = FP / (FP + TN)."""
        preds = np.asarray(predictions, dtype=int)
        truth = np.asarray(ground_truth, dtype=int)
        fp = int(np.sum((preds == 1) & (truth == 0)))
        tn = int(np.sum((preds == 0) & (truth == 0)))
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # ------------------------------------------------------------------
    # ROC curve
    # ------------------------------------------------------------------

    @staticmethod
    def roc_curve(
        scores: Sequence[float],
        labels: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ROC curve (FPR, TPR, thresholds).

        Returns:
            (fpr, tpr, thresholds) arrays sorted by increasing FPR.
        """
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=int)
        # Sort by descending score
        desc = np.argsort(-s)
        s_sorted = s[desc]
        y_sorted = y[desc]

        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        if n_pos == 0 or n_neg == 0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([s.max() + 1, s.min()])

        # Unique thresholds
        thresholds = np.unique(s_sorted)[::-1]
        fpr_list = [0.0]
        tpr_list = [0.0]
        thr_list = [float(thresholds[0]) + 1.0]

        for t in thresholds:
            preds = (s >= t).astype(int)
            tp = int(np.sum((preds == 1) & (y == 1)))
            fp = int(np.sum((preds == 1) & (y == 0)))
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)
            thr_list.append(float(t))

        return np.array(fpr_list), np.array(tpr_list), np.array(thr_list)

    # ------------------------------------------------------------------
    # AUC (trapezoidal rule)
    # ------------------------------------------------------------------

    @staticmethod
    def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
        """Area under the ROC curve via the trapezoidal rule."""
        # Sort by FPR
        order = np.argsort(fpr)
        fpr_s = fpr[order]
        tpr_s = tpr[order]
        return float(np.trapz(tpr_s, fpr_s))

    # ------------------------------------------------------------------
    # Precision-Recall curve
    # ------------------------------------------------------------------

    @staticmethod
    def precision_recall_curve(
        scores: Sequence[float],
        labels: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute precision-recall curve.

        Returns:
            (precision, recall, thresholds) arrays.
        """
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=int)
        desc = np.argsort(-s)
        s_sorted = s[desc]
        y_sorted = y[desc]

        n_pos = int(np.sum(y == 1))
        if n_pos == 0:
            return np.array([1.0]), np.array([0.0]), np.array([])

        thresholds = np.unique(s_sorted)[::-1]
        prec_list: list[float] = []
        rec_list: list[float] = []
        thr_list: list[float] = []

        for t in thresholds:
            preds = (s >= t).astype(int)
            tp = int(np.sum((preds == 1) & (y == 1)))
            fp = int(np.sum((preds == 1) & (y == 0)))
            fn = int(np.sum((preds == 0) & (y == 1)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec_list.append(precision)
            rec_list.append(recall)
            thr_list.append(float(t))

        return np.array(prec_list), np.array(rec_list), np.array(thr_list)

    # ------------------------------------------------------------------
    # Optimal threshold (Youden's J)
    # ------------------------------------------------------------------

    @staticmethod
    def optimal_threshold(
        scores: Sequence[float],
        labels: Sequence[int],
    ) -> float:
        """Find the threshold that maximises Youden's J statistic (TPR − FPR)."""
        fpr, tpr, thresholds = RegressionDetectionMetrics.roc_curve(scores, labels)
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        return float(thresholds[best_idx])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def full_report(
        scores: Sequence[float],
        labels: Sequence[int],
    ) -> dict[str, float]:
        """Compute a comprehensive set of metrics."""
        fpr, tpr, thresholds = RegressionDetectionMetrics.roc_curve(scores, labels)
        roc_auc = RegressionDetectionMetrics.auc(fpr, tpr)
        opt_t = RegressionDetectionMetrics.optimal_threshold(scores, labels)
        preds = (np.asarray(scores) >= opt_t).astype(int)
        tpr_val = RegressionDetectionMetrics.true_positive_rate(preds, labels)
        fpr_val = RegressionDetectionMetrics.false_positive_rate(preds, labels)
        prec, rec, _ = RegressionDetectionMetrics.precision_recall_curve(scores, labels)
        pr_auc = float(np.trapz(prec[np.argsort(rec)], np.sort(rec))) if len(prec) > 1 else 0.0
        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "optimal_threshold": opt_t,
            "tpr_at_optimal": tpr_val,
            "fpr_at_optimal": fpr_val,
        }

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    @staticmethod
    def confusion_matrix(
        predictions: Sequence[int],
        ground_truth: Sequence[int],
    ) -> dict[str, int]:
        """Compute TP, FP, TN, FN counts."""
        preds = np.asarray(predictions, dtype=int)
        truth = np.asarray(ground_truth, dtype=int)
        return {
            "tp": int(np.sum((preds == 1) & (truth == 1))),
            "fp": int(np.sum((preds == 1) & (truth == 0))),
            "tn": int(np.sum((preds == 0) & (truth == 0))),
            "fn": int(np.sum((preds == 0) & (truth == 1))),
        }

    # ------------------------------------------------------------------
    # F1 score
    # ------------------------------------------------------------------

    @staticmethod
    def f1_score(
        predictions: Sequence[int],
        ground_truth: Sequence[int],
    ) -> float:
        """Compute F1 score."""
        cm = RegressionDetectionMetrics.confusion_matrix(predictions, ground_truth)
        tp = cm["tp"]
        precision = tp / (tp + cm["fp"]) if (tp + cm["fp"]) > 0 else 0.0
        recall = tp / (tp + cm["fn"]) if (tp + cm["fn"]) > 0 else 0.0
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # ------------------------------------------------------------------
    # Matthews Correlation Coefficient
    # ------------------------------------------------------------------

    @staticmethod
    def mcc(
        predictions: Sequence[int],
        ground_truth: Sequence[int],
    ) -> float:
        """Matthews Correlation Coefficient — balanced measure for imbalanced data."""
        cm = RegressionDetectionMetrics.confusion_matrix(predictions, ground_truth)
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        if denom < 1e-12:
            return 0.0
        return float(tp * tn - fp * fn) / denom

    # ------------------------------------------------------------------
    # Cohen's kappa
    # ------------------------------------------------------------------

    @staticmethod
    def cohens_kappa(
        predictions: Sequence[int],
        ground_truth: Sequence[int],
    ) -> float:
        """Cohen's kappa for inter-rater agreement."""
        preds = np.asarray(predictions, dtype=int)
        truth = np.asarray(ground_truth, dtype=int)
        n = len(preds)
        if n == 0:
            return 0.0

        p_o = float(np.sum(preds == truth)) / n
        p_yes = (float(np.sum(preds == 1)) / n) * (float(np.sum(truth == 1)) / n)
        p_no = (float(np.sum(preds == 0)) / n) * (float(np.sum(truth == 0)) / n)
        p_e = p_yes + p_no

        if abs(1.0 - p_e) < 1e-12:
            return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
        return (p_o - p_e) / (1.0 - p_e)

    # ------------------------------------------------------------------
    # Calibration (reliability diagram data)
    # ------------------------------------------------------------------

    @staticmethod
    def calibration_curve(
        scores: Sequence[float],
        labels: Sequence[int],
        n_bins: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute calibration curve (mean predicted prob vs fraction positive).

        Returns:
            (mean_predicted, fraction_positive) arrays of length n_bins.
        """
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=int)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        mean_pred = np.zeros(n_bins)
        frac_pos = np.zeros(n_bins)

        for b in range(n_bins):
            mask = (s >= bin_edges[b]) & (s < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = mask | (s == bin_edges[b + 1])
            if np.sum(mask) > 0:
                mean_pred[b] = float(np.mean(s[mask]))
                frac_pos[b] = float(np.mean(y[mask]))

        return mean_pred, frac_pos

    # ------------------------------------------------------------------
    # Brier score
    # ------------------------------------------------------------------

    @staticmethod
    def brier_score(
        scores: Sequence[float],
        labels: Sequence[int],
    ) -> float:
        """Brier score: mean squared difference between predictions and outcomes."""
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        return float(np.mean((s - y) ** 2))

    # ------------------------------------------------------------------
    # Log loss
    # ------------------------------------------------------------------

    @staticmethod
    def log_loss(
        scores: Sequence[float],
        labels: Sequence[int],
        eps: float = 1e-15,
    ) -> float:
        """Binary cross-entropy (log loss)."""
        s = np.clip(np.asarray(scores, dtype=float), eps, 1.0 - eps)
        y = np.asarray(labels, dtype=float)
        return float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)))

    # ------------------------------------------------------------------
    # Operating point analysis
    # ------------------------------------------------------------------

    @staticmethod
    def operating_points(
        scores: Sequence[float],
        labels: Sequence[int],
        target_metrics: dict[str, float] | None = None,
    ) -> list[dict[str, float]]:
        """Find operating points for given target metrics.

        E.g. target_metrics={'tpr': 0.95} finds threshold achieving 95% TPR.
        """
        fpr_arr, tpr_arr, thresholds = RegressionDetectionMetrics.roc_curve(scores, labels)
        points = []

        if target_metrics is None:
            target_metrics = {"tpr": 0.95, "fpr": 0.05}

        for metric_name, target_val in target_metrics.items():
            if metric_name == "tpr":
                # Find threshold closest to target TPR
                idx = int(np.argmin(np.abs(tpr_arr - target_val)))
                points.append({
                    "target": f"TPR={target_val}",
                    "threshold": float(thresholds[idx]),
                    "tpr": float(tpr_arr[idx]),
                    "fpr": float(fpr_arr[idx]),
                })
            elif metric_name == "fpr":
                idx = int(np.argmin(np.abs(fpr_arr - target_val)))
                points.append({
                    "target": f"FPR={target_val}",
                    "threshold": float(thresholds[idx]),
                    "tpr": float(tpr_arr[idx]),
                    "fpr": float(fpr_arr[idx]),
                })

        return points
