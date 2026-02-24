"""Evaluation metrics for phase diagram predictions.

Provides boundary distance metrics, regime classification metrics,
and calibration metrics for assessing prediction quality.

Provides:
  - BoundaryMetrics: Hausdorff distance, coverage, precision/recall
  - RegimeMetrics: accuracy, AUC, F1, confusion matrix
  - CalibrationMetrics: ECE, MCE, Brier score, reliability curve
  - MetricsResult: aggregated metrics container
  - MetricsComputer: computes all metrics from predictions vs ground truth
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg as sp_linalg
from scipy.spatial.distance import cdist


# ======================================================================
# Metric data classes
# ======================================================================


@dataclass
class BoundaryMetrics:
    """Metrics comparing predicted and ground-truth phase boundaries.

    Parameters
    ----------
    hausdorff_distance : float
        Symmetric Hausdorff distance between boundary curves.
    mean_distance : float
        Mean closest-point distance from predicted to ground truth.
    coverage : float
        Fraction of ground truth boundary covered by prediction.
    precision : float
        Fraction of predicted boundary near ground truth.
    recall : float
        Fraction of ground truth boundary near prediction.
    """

    hausdorff_distance: float = 0.0
    mean_distance: float = 0.0
    coverage: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


@dataclass
class RegimeMetrics:
    """Metrics for regime classification quality.

    Parameters
    ----------
    accuracy : float
        Classification accuracy.
    auc : float
        Area under the ROC curve.
    f1_score : float
        F1 score (harmonic mean of precision and recall).
    confusion_matrix : np.ndarray
        2x2 confusion matrix.
    """

    accuracy: float = 0.0
    auc: float = 0.0
    f1_score: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None


@dataclass
class CalibrationMetrics:
    """Metrics for prediction calibration quality.

    Parameters
    ----------
    expected_calibration_error : float
        Expected calibration error (weighted bin-level error).
    max_calibration_error : float
        Maximum calibration error across bins.
    brier_score : float
        Mean squared error of probability predictions.
    reliability_curve : Tuple[np.ndarray, np.ndarray]
        (mean_predicted, fraction_positive) for reliability diagram.
    """

    expected_calibration_error: float = 0.0
    max_calibration_error: float = 0.0
    brier_score: float = 0.0
    reliability_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None


@dataclass
class MetricsResult:
    """Aggregated evaluation metrics.

    Parameters
    ----------
    boundary_metrics : BoundaryMetrics
        Phase boundary comparison metrics.
    regime_metrics : RegimeMetrics
        Regime classification metrics.
    calibration_metrics : CalibrationMetrics
        Calibration quality metrics.
    per_architecture : dict
        Architecture-specific metrics keyed by architecture name.
    """

    boundary_metrics: BoundaryMetrics = field(default_factory=BoundaryMetrics)
    regime_metrics: RegimeMetrics = field(default_factory=RegimeMetrics)
    calibration_metrics: CalibrationMetrics = field(
        default_factory=CalibrationMetrics
    )
    per_architecture: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Metrics computer
# ======================================================================


class MetricsComputer:
    """Compute evaluation metrics for phase diagram predictions.

    Compares predicted phase boundaries and regime classifications
    against ground-truth values from training experiments.

    Examples
    --------
    >>> mc = MetricsComputer()
    >>> hd = mc.hausdorff_distance(predicted_curve, gt_curve)
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Boundary distance metrics
    # ------------------------------------------------------------------

    def hausdorff_distance(
        self,
        curve1: np.ndarray,
        curve2: np.ndarray,
    ) -> float:
        """Compute directed Hausdorff distance from curve1 to curve2.

        For each point in curve1, find the nearest point in curve2;
        return the maximum of these nearest-point distances.

        Parameters
        ----------
        curve1 : ndarray of shape (M, D)
            First set of boundary points.
        curve2 : ndarray of shape (N, D)
            Second set of boundary points.

        Returns
        -------
        distance : float
            Directed Hausdorff distance: max_{p in curve1} min_{q in curve2} ||p - q||.
        """
        curve1 = np.atleast_2d(curve1)
        curve2 = np.atleast_2d(curve2)
        if curve1.shape[0] == 0 or curve2.shape[0] == 0:
            return float("inf")
        D = cdist(curve1, curve2, metric="euclidean")
        return float(np.max(np.min(D, axis=1)))

    def symmetric_hausdorff(
        self,
        curve1: np.ndarray,
        curve2: np.ndarray,
    ) -> float:
        """Compute symmetric Hausdorff distance between two curves.

        Parameters
        ----------
        curve1 : ndarray of shape (M, D)
            First set of boundary points.
        curve2 : ndarray of shape (N, D)
            Second set of boundary points.

        Returns
        -------
        distance : float
            max(H(curve1, curve2), H(curve2, curve1)).
        """
        return max(
            self.hausdorff_distance(curve1, curve2),
            self.hausdorff_distance(curve2, curve1),
        )

    def boundary_coverage(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        tolerance: float = 0.1,
    ) -> float:
        """Fraction of ground truth boundary covered by prediction.

        A ground truth point is "covered" if any predicted point lies
        within *tolerance* distance.

        Parameters
        ----------
        predicted : ndarray of shape (M, D)
            Predicted boundary points.
        ground_truth : ndarray of shape (N, D)
            Ground truth boundary points.
        tolerance : float
            Distance threshold for coverage.

        Returns
        -------
        coverage : float
            Fraction in [0, 1].
        """
        predicted = np.atleast_2d(predicted)
        ground_truth = np.atleast_2d(ground_truth)
        if ground_truth.shape[0] == 0:
            return 1.0
        if predicted.shape[0] == 0:
            return 0.0
        D = cdist(ground_truth, predicted, metric="euclidean")
        min_dists = np.min(D, axis=1)
        covered = np.sum(min_dists <= tolerance)
        return float(covered / ground_truth.shape[0])

    def boundary_precision(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        tolerance: float = 0.1,
    ) -> float:
        """Fraction of predicted boundary near ground truth.

        A predicted point is "precise" if it lies within *tolerance*
        of some ground truth point.

        Parameters
        ----------
        predicted : ndarray of shape (M, D)
            Predicted boundary points.
        ground_truth : ndarray of shape (N, D)
            Ground truth boundary points.
        tolerance : float
            Distance threshold.

        Returns
        -------
        precision : float
            Fraction in [0, 1].
        """
        predicted = np.atleast_2d(predicted)
        ground_truth = np.atleast_2d(ground_truth)
        if predicted.shape[0] == 0:
            return 1.0
        if ground_truth.shape[0] == 0:
            return 0.0
        D = cdist(predicted, ground_truth, metric="euclidean")
        min_dists = np.min(D, axis=1)
        precise = np.sum(min_dists <= tolerance)
        return float(precise / predicted.shape[0])

    def _mean_boundary_distance(
        self,
        curve1: np.ndarray,
        curve2: np.ndarray,
    ) -> float:
        """Mean closest-point distance from curve1 to curve2.

        Parameters
        ----------
        curve1 : ndarray of shape (M, D)
        curve2 : ndarray of shape (N, D)

        Returns
        -------
        mean_dist : float
        """
        curve1 = np.atleast_2d(curve1)
        curve2 = np.atleast_2d(curve2)
        if curve1.shape[0] == 0 or curve2.shape[0] == 0:
            return float("inf")
        D = cdist(curve1, curve2, metric="euclidean")
        return float(np.mean(np.min(D, axis=1)))

    # ------------------------------------------------------------------
    # Regime classification metrics
    # ------------------------------------------------------------------

    def regime_classification_auc(
        self,
        predicted_probs: np.ndarray,
        true_labels: np.ndarray,
    ) -> float:
        """Compute AUC for regime classification.

        Parameters
        ----------
        predicted_probs : ndarray of shape (N,)
            Predicted probability of positive class.
        true_labels : ndarray of shape (N,)
            Binary ground truth labels (0 or 1).

        Returns
        -------
        auc : float
            Area under the ROC curve.
        """
        fpr, tpr = self._roc_curve(predicted_probs, true_labels)
        # Trapezoidal integration
        auc = float(np.trapz(tpr, fpr))
        return abs(auc)

    def _roc_curve(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ROC curve from scores and binary labels.

        Parameters
        ----------
        scores : ndarray of shape (N,)
            Predicted scores or probabilities.
        labels : ndarray of shape (N,)
            Binary labels (0 or 1).

        Returns
        -------
        fpr : ndarray
            False positive rates at each threshold.
        tpr : ndarray
            True positive rates at each threshold.
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        # Sort by descending score
        order = np.argsort(-scores)
        sorted_labels = labels[order]
        sorted_scores = scores[order]

        # Unique thresholds
        thresholds = np.unique(sorted_scores)
        thresholds = np.sort(thresholds)[::-1]

        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)

        if n_pos == 0 or n_neg == 0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0])

        fpr_list = [0.0]
        tpr_list = [0.0]

        for thresh in thresholds:
            predicted_pos = scores >= thresh
            tp = float(np.sum(predicted_pos & (labels == 1)))
            fp = float(np.sum(predicted_pos & (labels == 0)))
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        fpr_list.append(1.0)
        tpr_list.append(1.0)

        return np.array(fpr_list), np.array(tpr_list)

    def confusion_matrix(
        self,
        predicted: np.ndarray,
        true: np.ndarray,
    ) -> np.ndarray:
        """Compute 2x2 confusion matrix.

        Parameters
        ----------
        predicted : ndarray of shape (N,)
            Predicted binary labels.
        true : ndarray of shape (N,)
            True binary labels.

        Returns
        -------
        cm : ndarray of shape (2, 2)
            Confusion matrix: [[TN, FP], [FN, TP]].
        """
        predicted = np.asarray(predicted, dtype=int)
        true = np.asarray(true, dtype=int)
        tn = int(np.sum((predicted == 0) & (true == 0)))
        fp = int(np.sum((predicted == 1) & (true == 0)))
        fn = int(np.sum((predicted == 0) & (true == 1)))
        tp = int(np.sum((predicted == 1) & (true == 1)))
        return np.array([[tn, fp], [fn, tp]], dtype=np.int64)

    def f1_score(
        self,
        predicted: np.ndarray,
        true: np.ndarray,
    ) -> float:
        """Compute F1 score.

        Parameters
        ----------
        predicted : ndarray of shape (N,)
            Predicted binary labels.
        true : ndarray of shape (N,)
            True binary labels.

        Returns
        -------
        f1 : float
            Harmonic mean of precision and recall.
        """
        cm = self.confusion_matrix(predicted, true)
        tp = float(cm[1, 1])
        fp = float(cm[0, 1])
        fn = float(cm[1, 0])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def _accuracy(
        self,
        predicted: np.ndarray,
        true: np.ndarray,
    ) -> float:
        """Compute classification accuracy.

        Parameters
        ----------
        predicted : ndarray of shape (N,)
        true : ndarray of shape (N,)

        Returns
        -------
        accuracy : float
        """
        predicted = np.asarray(predicted, dtype=int)
        true = np.asarray(true, dtype=int)
        if len(predicted) == 0:
            return 0.0
        return float(np.mean(predicted == true))

    # ------------------------------------------------------------------
    # Calibration metrics
    # ------------------------------------------------------------------

    def calibration_error(
        self,
        predicted_probs: np.ndarray,
        true_outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute expected calibration error with reliability diagram data.

        Parameters
        ----------
        predicted_probs : ndarray of shape (N,)
            Predicted probabilities in [0, 1].
        true_outcomes : ndarray of shape (N,)
            Binary outcomes (0 or 1).
        n_bins : int
            Number of bins for calibration histogram.

        Returns
        -------
        ece : float
            Expected calibration error.
        mean_predicted : ndarray of shape (n_bins,)
            Mean predicted probability per bin.
        fraction_positive : ndarray of shape (n_bins,)
            Fraction of positive outcomes per bin.
        """
        predicted_probs = np.asarray(predicted_probs, dtype=np.float64)
        true_outcomes = np.asarray(true_outcomes, dtype=np.float64)
        N = len(predicted_probs)

        if N == 0:
            return 0.0, np.array([]), np.array([])

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        mean_predicted = np.zeros(n_bins, dtype=np.float64)
        fraction_positive = np.zeros(n_bins, dtype=np.float64)
        bin_counts = np.zeros(n_bins, dtype=np.float64)

        for b in range(n_bins):
            lo = bin_edges[b]
            hi = bin_edges[b + 1]
            if b == n_bins - 1:
                mask = (predicted_probs >= lo) & (predicted_probs <= hi)
            else:
                mask = (predicted_probs >= lo) & (predicted_probs < hi)
            count = np.sum(mask)
            bin_counts[b] = count
            if count > 0:
                mean_predicted[b] = np.mean(predicted_probs[mask])
                fraction_positive[b] = np.mean(true_outcomes[mask])

        # ECE: weighted average of |accuracy - confidence| per bin
        ece = 0.0
        for b in range(n_bins):
            if bin_counts[b] > 0:
                ece += (bin_counts[b] / N) * abs(
                    fraction_positive[b] - mean_predicted[b]
                )

        return float(ece), mean_predicted, fraction_positive

    def _max_calibration_error(
        self,
        predicted_probs: np.ndarray,
        true_outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute maximum calibration error across bins.

        Parameters
        ----------
        predicted_probs : ndarray of shape (N,)
        true_outcomes : ndarray of shape (N,)
        n_bins : int

        Returns
        -------
        mce : float
        """
        _, mean_pred, frac_pos = self.calibration_error(
            predicted_probs, true_outcomes, n_bins
        )
        if len(mean_pred) == 0:
            return 0.0
        # Only consider bins that have data
        nonzero = (mean_pred > 0) | (frac_pos > 0)
        if not np.any(nonzero):
            return 0.0
        return float(np.max(np.abs(frac_pos[nonzero] - mean_pred[nonzero])))

    def _brier_score(
        self,
        predicted_probs: np.ndarray,
        true_outcomes: np.ndarray,
    ) -> float:
        """Compute Brier score (mean squared probability error).

        Parameters
        ----------
        predicted_probs : ndarray of shape (N,)
        true_outcomes : ndarray of shape (N,)

        Returns
        -------
        brier : float
        """
        predicted_probs = np.asarray(predicted_probs, dtype=np.float64)
        true_outcomes = np.asarray(true_outcomes, dtype=np.float64)
        if len(predicted_probs) == 0:
            return 0.0
        return float(np.mean((predicted_probs - true_outcomes) ** 2))

    def prediction_confidence_calibration(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """Compute full calibration metrics.

        Parameters
        ----------
        predictions : ndarray of shape (N,)
            Predicted probabilities.
        ground_truth : ndarray of shape (N,)
            Binary ground truth outcomes.
        n_bins : int
            Number of calibration bins.

        Returns
        -------
        metrics : CalibrationMetrics
        """
        ece, mean_pred, frac_pos = self.calibration_error(
            predictions, ground_truth, n_bins
        )
        mce = self._max_calibration_error(predictions, ground_truth, n_bins)
        brier = self._brier_score(predictions, ground_truth)
        return CalibrationMetrics(
            expected_calibration_error=ece,
            max_calibration_error=mce,
            brier_score=brier,
            reliability_curve=(mean_pred, frac_pos),
        )

    # ------------------------------------------------------------------
    # Aggregated metrics
    # ------------------------------------------------------------------

    def compute_all(
        self,
        predicted_diagram: Dict[str, Any],
        ground_truth_diagram: Dict[str, Any],
    ) -> MetricsResult:
        """Compute all metrics comparing predicted to ground truth diagram.

        Parameters
        ----------
        predicted_diagram : dict
            Must contain keys:
              - 'boundary': ndarray of shape (M, D), boundary points
              - 'labels': ndarray of shape (N,), predicted binary labels
              - 'probs': ndarray of shape (N,), predicted probabilities
        ground_truth_diagram : dict
            Must contain keys:
              - 'boundary': ndarray of shape (K, D), ground truth boundary
              - 'labels': ndarray of shape (N,), true binary labels

        Returns
        -------
        result : MetricsResult
        """
        pred_boundary = np.atleast_2d(predicted_diagram["boundary"])
        gt_boundary = np.atleast_2d(ground_truth_diagram["boundary"])
        pred_labels = np.asarray(predicted_diagram["labels"])
        gt_labels = np.asarray(ground_truth_diagram["labels"])
        pred_probs = np.asarray(predicted_diagram["probs"])

        # Boundary metrics
        h_dist = self.symmetric_hausdorff(pred_boundary, gt_boundary)
        m_dist = self._mean_boundary_distance(pred_boundary, gt_boundary)
        cov = self.boundary_coverage(pred_boundary, gt_boundary)
        prec = self.boundary_precision(pred_boundary, gt_boundary)
        rec = self.boundary_coverage(pred_boundary, gt_boundary)

        boundary = BoundaryMetrics(
            hausdorff_distance=h_dist,
            mean_distance=m_dist,
            coverage=cov,
            precision=prec,
            recall=rec,
        )

        # Regime metrics
        acc = self._accuracy(pred_labels, gt_labels)
        auc = self.regime_classification_auc(pred_probs, gt_labels)
        f1 = self.f1_score(pred_labels, gt_labels)
        cm = self.confusion_matrix(pred_labels, gt_labels)

        regime = RegimeMetrics(
            accuracy=acc,
            auc=auc,
            f1_score=f1,
            confusion_matrix=cm,
        )

        # Calibration metrics
        calibration = self.prediction_confidence_calibration(
            pred_probs, gt_labels
        )

        return MetricsResult(
            boundary_metrics=boundary,
            regime_metrics=regime,
            calibration_metrics=calibration,
        )

    # ------------------------------------------------------------------
    # Per-architecture metrics
    # ------------------------------------------------------------------

    def per_architecture_metrics(
        self,
        results_dict: Dict[str, Dict[str, Any]],
    ) -> Dict[str, MetricsResult]:
        """Compute metrics separately for each architecture type.

        Parameters
        ----------
        results_dict : dict
            Mapping from architecture name to a dict with keys:
              - 'predicted': dict with 'boundary', 'labels', 'probs'
              - 'ground_truth': dict with 'boundary', 'labels'

        Returns
        -------
        per_arch : dict
            Mapping from architecture name to MetricsResult.
        """
        per_arch: Dict[str, MetricsResult] = {}
        for arch_name, data in results_dict.items():
            result = self.compute_all(data["predicted"], data["ground_truth"])
            per_arch[arch_name] = result
        return per_arch
