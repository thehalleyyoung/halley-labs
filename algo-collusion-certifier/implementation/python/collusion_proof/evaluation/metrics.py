"""Evaluation metrics for CollusionProof.

Provides classification metrics (precision, recall, F1, ROC-AUC, MCC, …),
statistical-testing diagnostics (type-I/II error rates, power), and
bootstrap confidence intervals for any metric.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


class ClassificationMetrics:
    """Accumulate predictions and compute classification metrics.

    Uses binary convention: positive = collusive (1), negative = competitive (0).
    """

    def __init__(self) -> None:
        self.y_true: List[int] = []
        self.y_pred: List[int] = []
        self.y_scores: List[float] = []

    # -- accumulation --------------------------------------------------------

    def add(self, true_label: int, pred_label: int, score: float = 0.0) -> None:
        """Append a single prediction."""
        self.y_true.append(int(true_label))
        self.y_pred.append(int(pred_label))
        self.y_scores.append(float(score))

    def add_batch(
        self,
        true_labels: List[int],
        pred_labels: List[int],
        scores: Optional[List[float]] = None,
    ) -> None:
        """Append a batch of predictions."""
        if scores is None:
            scores = [0.0] * len(true_labels)
        for t, p, s in zip(true_labels, pred_labels, scores):
            self.add(t, p, s)

    # -- confusion matrix ----------------------------------------------------

    def confusion_matrix(self) -> np.ndarray:
        """Return a 2×2 confusion matrix [[TN, FP], [FN, TP]]."""
        yt = np.asarray(self.y_true)
        yp = np.asarray(self.y_pred)
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        return np.array([[tn, fp], [fn, tp]])

    def _tp_fp_fn_tn(self, positive_label: int = 1) -> Tuple[int, int, int, int]:
        yt = np.asarray(self.y_true)
        yp = np.asarray(self.y_pred)
        tp = int(np.sum((yt == positive_label) & (yp == positive_label)))
        fp = int(np.sum((yt != positive_label) & (yp == positive_label)))
        fn = int(np.sum((yt == positive_label) & (yp != positive_label)))
        tn = int(np.sum((yt != positive_label) & (yp != positive_label)))
        return tp, fp, fn, tn

    # -- scalar metrics ------------------------------------------------------

    def precision(self, positive_label: int = 1) -> float:
        tp, fp, _fn, _tn = self._tp_fp_fn_tn(positive_label)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(self, positive_label: int = 1) -> float:
        tp, _fp, fn, _tn = self._tp_fp_fn_tn(positive_label)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def f1_score(self, positive_label: int = 1) -> float:
        p = self.precision(positive_label)
        r = self.recall(positive_label)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def accuracy(self) -> float:
        n = len(self.y_true)
        if n == 0:
            return 0.0
        correct = sum(t == p for t, p in zip(self.y_true, self.y_pred))
        return correct / n

    def specificity(self) -> float:
        """True negative rate: TN / (TN + FP)."""
        _tp, fp, _fn, tn = self._tp_fp_fn_tn()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def type_i_error(self) -> float:
        """False positive rate: FP / (FP + TN)."""
        return 1.0 - self.specificity()

    def type_ii_error(self) -> float:
        """False negative rate: FN / (FN + TP)."""
        return 1.0 - self.recall()

    def power(self) -> float:
        """Statistical power (1 − type II error)."""
        return self.recall()

    def mcc(self) -> float:
        """Matthews correlation coefficient."""
        tp, fp, fn, tn = self._tp_fp_fn_tn()
        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        if denom == 0.0:
            return 0.0
        return (tp * tn - fp * fn) / denom

    def balanced_accuracy(self) -> float:
        """Average of recall for each class."""
        sens = self.recall()
        spec = self.specificity()
        return (sens + spec) / 2.0

    # -- threshold-based curves ----------------------------------------------

    def roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ROC curve from scores.

        Returns ``(fpr, tpr, thresholds)`` sorted by ascending FPR.
        """
        scores = np.asarray(self.y_scores, dtype=float)
        labels = np.asarray(self.y_true, dtype=int)

        # Sort by descending score
        desc_idx = np.argsort(-scores)
        sorted_scores = scores[desc_idx]
        sorted_labels = labels[desc_idx]

        # Unique thresholds (include a point that classifies everything negative)
        thresholds = np.unique(sorted_scores)[::-1]
        thresholds = np.concatenate([[thresholds[0] + 1.0], thresholds])

        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)

        fpr_list = []
        tpr_list = []
        for thresh in thresholds:
            pred_pos = scores >= thresh
            tp = np.sum(pred_pos & (labels == 1))
            fp = np.sum(pred_pos & (labels == 0))
            tpr_list.append(tp / n_pos if n_pos > 0 else 0.0)
            fpr_list.append(fp / n_neg if n_neg > 0 else 0.0)

        return (
            np.asarray(fpr_list),
            np.asarray(tpr_list),
            thresholds,
        )

    def auc(self) -> float:
        """Area under the ROC curve (trapezoidal rule)."""
        fpr, tpr, _ = self.roc_curve()
        # Sort by fpr to ensure correct integration
        order = np.argsort(fpr)
        fpr_sorted = fpr[order]
        tpr_sorted = tpr[order]
        return float(np.trapz(tpr_sorted, fpr_sorted))

    def precision_recall_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute precision-recall curve.

        Returns ``(precision_arr, recall_arr, thresholds)``.
        """
        scores = np.asarray(self.y_scores, dtype=float)
        labels = np.asarray(self.y_true, dtype=int)
        n_pos = np.sum(labels == 1)

        thresholds = np.unique(scores)[::-1]
        prec_list = []
        rec_list = []

        for thresh in thresholds:
            pred_pos = scores >= thresh
            tp = np.sum(pred_pos & (labels == 1))
            fp = np.sum(pred_pos & (labels == 0))
            p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r = tp / n_pos if n_pos > 0 else 0.0
            prec_list.append(p)
            rec_list.append(r)

        return (
            np.asarray(prec_list),
            np.asarray(rec_list),
            thresholds,
        )

    # -- aggregate -----------------------------------------------------------

    def all_metrics(self) -> Dict[str, float]:
        """Return a dictionary of all scalar metrics."""
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1_score": self.f1_score(),
            "specificity": self.specificity(),
            "type_i_error": self.type_i_error(),
            "type_ii_error": self.type_ii_error(),
            "power": self.power(),
            "mcc": self.mcc(),
            "balanced_accuracy": self.balanced_accuracy(),
            "auc": self.auc() if len(set(self.y_scores)) > 1 else float("nan"),
        }

    def summary(self) -> str:
        """Human-readable summary of metrics."""
        m = self.all_metrics()
        cm = self.confusion_matrix()
        lines = [
            "=== Classification Metrics ===",
            f"  n = {len(self.y_true)}",
            f"  Confusion matrix (rows=actual, cols=predicted):",
            f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}",
            f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}",
            f"  Accuracy          : {m['accuracy']:.4f}",
            f"  Precision         : {m['precision']:.4f}",
            f"  Recall / Power    : {m['recall']:.4f}",
            f"  F1 Score          : {m['f1_score']:.4f}",
            f"  Specificity       : {m['specificity']:.4f}",
            f"  Type I error (FPR): {m['type_i_error']:.4f}",
            f"  Type II error(FNR): {m['type_ii_error']:.4f}",
            f"  MCC               : {m['mcc']:.4f}",
            f"  Balanced accuracy : {m['balanced_accuracy']:.4f}",
        ]
        if not np.isnan(m["auc"]):
            lines.append(f"  ROC AUC           : {m['auc']:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone metric functions
# ---------------------------------------------------------------------------

def compute_type_i_error_rate(results: List[Dict], alpha: float = 0.05) -> float:
    """Compute empirical type I error rate from competitive scenarios.

    Parameters
    ----------
    results : list of dicts with keys ``expected_verdict`` and ``actual_verdict``
    alpha : nominal significance level (for reference only)

    Returns
    -------
    Empirical false-positive rate among truly-competitive scenarios.
    """
    competitive = [
        r for r in results
        if r.get("expected_verdict", "").lower() in ("competitive", "comp")
    ]
    if not competitive:
        return 0.0
    false_positives = sum(
        1 for r in competitive
        if r.get("actual_verdict", "").lower() in ("collusive", "col")
    )
    return false_positives / len(competitive)


def compute_power(results: List[Dict]) -> float:
    """Compute empirical power (true positive rate) from collusive scenarios."""
    collusive = [
        r for r in results
        if r.get("expected_verdict", "").lower() in ("collusive", "col")
    ]
    if not collusive:
        return 0.0
    true_positives = sum(
        1 for r in collusive
        if r.get("actual_verdict", "").lower() in ("collusive", "col")
    )
    return true_positives / len(collusive)


def compute_detection_delay(results: List[Dict]) -> Dict[str, float]:
    """Compute detection delay statistics.

    Each result dict should contain ``convergence_round`` (when collusion
    starts) and ``detection_round`` (when the detector fires). The delay
    is the difference.

    Returns dict with keys ``mean``, ``median``, ``std``, ``min``, ``max``.
    """
    delays = []
    for r in results:
        conv = r.get("convergence_round")
        det = r.get("detection_round")
        if conv is not None and det is not None:
            delays.append(max(0, det - conv))

    if not delays:
        return {"mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "min": float("nan"), "max": float("nan")}

    arr = np.asarray(delays, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_robustness_score(results_across_params: List[List[Dict]]) -> float:
    """Compute robustness score across parameter variations.

    Given multiple evaluation runs (one per parameter setting), compute the
    fraction of runs that achieve ≥ 80 % accuracy.  The robustness score is
    this fraction, in [0, 1].
    """
    if not results_across_params:
        return 0.0

    good_runs = 0
    for run_results in results_across_params:
        if not run_results:
            continue
        correct = sum(
            1 for r in run_results
            if r.get("expected_verdict") == r.get("actual_verdict")
        )
        accuracy = correct / len(run_results)
        if accuracy >= 0.8:
            good_runs += 1

    return good_runs / len(results_across_params)


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Bootstrap confidence interval for an arbitrary metric.

    Parameters
    ----------
    y_true, y_pred : label arrays
    metric_fn : callable(y_true, y_pred) -> float
    n_bootstrap : number of bootstrap iterations
    confidence : confidence level for the interval
    seed : random seed

    Returns
    -------
    Dict with ``point``, ``lower``, ``upper``, ``std``.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    point = float(metric_fn(y_true, y_pred))

    boot_values = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_values[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_values, 100 * alpha / 2))
    upper = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))

    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "std": float(np.std(boot_values, ddof=1)),
    }
