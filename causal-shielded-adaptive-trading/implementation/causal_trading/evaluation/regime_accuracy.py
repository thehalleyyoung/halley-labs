"""
Regime detection accuracy evaluation.

Computes Adjusted Rand Index, Normalized Mutual Information, V-measure,
change-point detection delay, and confusion matrices for regime labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class RegimeAccuracyMetrics:
    """Container for regime accuracy metrics."""
    adjusted_rand_index: float
    normalized_mutual_info: float
    v_measure: float
    homogeneity: float
    completeness: float
    transition_accuracy: float
    mean_detection_delay: float
    median_detection_delay: float
    confusion_matrix: NDArray[np.int64]
    per_regime_precision: NDArray[np.float64]
    per_regime_recall: NDArray[np.float64]
    per_regime_f1: NDArray[np.float64]
    n_true_transitions: int
    n_detected_transitions: int
    false_positive_transitions: int
    missed_transitions: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Regime Accuracy ===",
            f"Adjusted Rand Index:       {self.adjusted_rand_index:.4f}",
            f"Norm. Mutual Information:  {self.normalized_mutual_info:.4f}",
            f"V-measure:                 {self.v_measure:.4f}",
            f"  Homogeneity:             {self.homogeneity:.4f}",
            f"  Completeness:            {self.completeness:.4f}",
            f"Transition Accuracy:       {self.transition_accuracy:.4f}",
            f"Mean Detection Delay:      {self.mean_detection_delay:.2f}",
            f"Median Detection Delay:    {self.median_detection_delay:.2f}",
            f"True Transitions:          {self.n_true_transitions}",
            f"Detected Transitions:      {self.n_detected_transitions}",
            f"False Positive Trans.:     {self.false_positive_transitions}",
            f"Missed Transitions:        {self.missed_transitions}",
        ]
        return "\n".join(lines)


class RegimeAccuracyEvaluator:
    """Evaluates regime detection accuracy.

    Computes clustering-quality metrics (ARI, NMI, V-measure) as well as
    change-point-specific metrics (detection delay, transition accuracy).
    """

    def __init__(self) -> None:
        self._metrics: Optional[RegimeAccuracyMetrics] = None

    # ---- public API --------------------------------------------------------

    def evaluate(
        self,
        true_regimes: NDArray[np.int64],
        predicted_regimes: NDArray[np.int64],
        delay_tolerance: int = 5,
    ) -> RegimeAccuracyMetrics:
        """Evaluate predicted regime labels against ground truth.

        Parameters
        ----------
        true_regimes : 1-D integer array of ground-truth regime labels
        predicted_regimes : 1-D integer array of predicted regime labels
        delay_tolerance : maximum acceptable detection delay (in time-steps)

        Returns
        -------
        RegimeAccuracyMetrics
        """
        true_regimes = np.asarray(true_regimes, dtype=np.int64).ravel()
        predicted_regimes = np.asarray(predicted_regimes, dtype=np.int64).ravel()
        n = len(true_regimes)
        if len(predicted_regimes) != n:
            raise ValueError("Length mismatch between true and predicted regimes.")

        ari = self._adjusted_rand_index(true_regimes, predicted_regimes)
        nmi = self._normalized_mutual_info(true_regimes, predicted_regimes)
        homo, comp, vm = self._v_measure(true_regimes, predicted_regimes)

        # Confusion matrix (rows = true, cols = predicted)
        labels_true = np.unique(true_regimes)
        labels_pred = np.unique(predicted_regimes)
        all_labels = np.union1d(labels_true, labels_pred)
        k = len(all_labels)
        label_map = {int(l): i for i, l in enumerate(all_labels)}
        cm = np.zeros((k, k), dtype=np.int64)
        for t, p in zip(true_regimes, predicted_regimes):
            cm[label_map[int(t)], label_map[int(p)]] += 1

        # Per-regime precision / recall / F1
        prec, rec, f1 = self._per_regime_prf(cm)

        # Transition metrics
        true_cp = self._find_change_points(true_regimes)
        pred_cp = self._find_change_points(predicted_regimes)
        trans_acc, delays, n_fp, n_miss = self._transition_accuracy(
            true_cp, pred_cp, delay_tolerance,
        )
        mean_delay = float(np.mean(delays)) if len(delays) > 0 else 0.0
        med_delay = float(np.median(delays)) if len(delays) > 0 else 0.0

        self._metrics = RegimeAccuracyMetrics(
            adjusted_rand_index=ari,
            normalized_mutual_info=nmi,
            v_measure=vm,
            homogeneity=homo,
            completeness=comp,
            transition_accuracy=trans_acc,
            mean_detection_delay=mean_delay,
            median_detection_delay=med_delay,
            confusion_matrix=cm,
            per_regime_precision=prec,
            per_regime_recall=rec,
            per_regime_f1=f1,
            n_true_transitions=len(true_cp),
            n_detected_transitions=len(pred_cp),
            false_positive_transitions=n_fp,
            missed_transitions=n_miss,
        )
        return self._metrics

    def get_metrics(self) -> RegimeAccuracyMetrics:
        """Return the last computed metrics."""
        if self._metrics is None:
            raise RuntimeError("Call evaluate() first.")
        return self._metrics

    # ---- Adjusted Rand Index -----------------------------------------------

    @staticmethod
    def _adjusted_rand_index(
        true: NDArray[np.int64],
        pred: NDArray[np.int64],
    ) -> float:
        """Compute Adjusted Rand Index from scratch.

        ARI = (RI - E[RI]) / (max(RI) - E[RI])
        where RI is computed from the contingency table.
        """
        n = len(true)
        labels_t = np.unique(true)
        labels_p = np.unique(pred)
        lt_map = {int(l): i for i, l in enumerate(labels_t)}
        lp_map = {int(l): i for i, l in enumerate(labels_p)}
        contingency = np.zeros((len(labels_t), len(labels_p)), dtype=np.int64)
        for i in range(n):
            contingency[lt_map[int(true[i])], lp_map[int(pred[i])]] += 1

        # Sum of C(n_ij, 2)
        sum_comb_c = np.sum(contingency * (contingency - 1)) / 2.0
        a = np.sum(contingency, axis=1)
        b = np.sum(contingency, axis=0)
        sum_comb_a = np.sum(a * (a - 1)) / 2.0
        sum_comb_b = np.sum(b * (b - 1)) / 2.0
        total_comb = n * (n - 1) / 2.0

        expected = sum_comb_a * sum_comb_b / total_comb if total_comb > 0 else 0.0
        max_index = (sum_comb_a + sum_comb_b) / 2.0
        denom = max_index - expected
        if abs(denom) < 1e-12:
            return 1.0 if abs(sum_comb_c - expected) < 1e-12 else 0.0
        return float((sum_comb_c - expected) / denom)

    # ---- Normalized Mutual Information ------------------------------------

    @staticmethod
    def _normalized_mutual_info(
        true: NDArray[np.int64],
        pred: NDArray[np.int64],
    ) -> float:
        """Compute NMI using arithmetic mean normalisation."""
        n = len(true)
        labels_t = np.unique(true)
        labels_p = np.unique(pred)
        lt_map = {int(l): i for i, l in enumerate(labels_t)}
        lp_map = {int(l): i for i, l in enumerate(labels_p)}
        contingency = np.zeros((len(labels_t), len(labels_p)), dtype=np.float64)
        for i in range(n):
            contingency[lt_map[int(true[i])], lp_map[int(pred[i])]] += 1

        # Marginals
        a = np.sum(contingency, axis=1)
        b = np.sum(contingency, axis=0)

        # Entropies
        h_true = -np.sum((a / n) * np.log(np.maximum(a / n, 1e-15)))
        h_pred = -np.sum((b / n) * np.log(np.maximum(b / n, 1e-15)))

        # Mutual information
        mi = 0.0
        for i in range(len(labels_t)):
            for j in range(len(labels_p)):
                if contingency[i, j] > 0:
                    pij = contingency[i, j] / n
                    mi += pij * np.log(pij / (a[i] / n * b[j] / n))

        denom = (h_true + h_pred) / 2.0
        if denom < 1e-12:
            return 1.0 if mi < 1e-12 else 0.0
        return float(mi / denom)

    # ---- V-measure (homogeneity + completeness) ----------------------------

    @staticmethod
    def _v_measure(
        true: NDArray[np.int64],
        pred: NDArray[np.int64],
        beta: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Compute homogeneity, completeness, and V-measure.

        Parameters
        ----------
        true, pred : label arrays
        beta : weight of homogeneity vs completeness

        Returns
        -------
        (homogeneity, completeness, v_measure)
        """
        n = len(true)
        labels_t = np.unique(true)
        labels_p = np.unique(pred)
        lt_map = {int(l): i for i, l in enumerate(labels_t)}
        lp_map = {int(l): i for i, l in enumerate(labels_p)}
        contingency = np.zeros((len(labels_t), len(labels_p)), dtype=np.float64)
        for i in range(n):
            contingency[lt_map[int(true[i])], lp_map[int(pred[i])]] += 1

        a = np.sum(contingency, axis=1)  # class sizes
        b = np.sum(contingency, axis=0)  # cluster sizes

        # H(C)
        h_c = -np.sum((a / n) * np.log(np.maximum(a / n, 1e-15)))
        # H(K)
        h_k = -np.sum((b / n) * np.log(np.maximum(b / n, 1e-15)))

        # H(C|K) = - sum_{k,c} (n_{ck}/n) log(n_{ck}/n_k)
        h_c_k = 0.0
        for j in range(len(labels_p)):
            for i in range(len(labels_t)):
                if contingency[i, j] > 0 and b[j] > 0:
                    h_c_k -= (contingency[i, j] / n) * np.log(contingency[i, j] / b[j])

        # H(K|C) = - sum_{c,k} (n_{ck}/n) log(n_{ck}/n_c)
        h_k_c = 0.0
        for i in range(len(labels_t)):
            for j in range(len(labels_p)):
                if contingency[i, j] > 0 and a[i] > 0:
                    h_k_c -= (contingency[i, j] / n) * np.log(contingency[i, j] / a[i])

        homo = 1.0 - h_c_k / h_c if h_c > 1e-12 else 1.0
        comp = 1.0 - h_k_c / h_k if h_k > 1e-12 else 1.0

        if homo + comp < 1e-12:
            vm = 0.0
        else:
            vm = (1 + beta) * homo * comp / (beta * homo + comp)

        return float(homo), float(comp), float(vm)

    # ---- Change-point / transition metrics ---------------------------------

    @staticmethod
    def _find_change_points(labels: NDArray[np.int64]) -> List[int]:
        """Return indices where the label changes."""
        cps: List[int] = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                cps.append(i)
        return cps

    @staticmethod
    def _transition_accuracy(
        true_cps: List[int],
        pred_cps: List[int],
        tolerance: int = 5,
    ) -> Tuple[float, List[int], int, int]:
        """Evaluate change-point detection.

        A predicted change point matches a true one if it lies within
        *tolerance* time-steps.

        Parameters
        ----------
        true_cps : indices of true change points
        pred_cps : indices of predicted change points
        tolerance : matching tolerance

        Returns
        -------
        (accuracy, matched_delays, n_false_positive, n_missed)
        """
        matched_delays: List[int] = []
        matched_true: set = set()
        matched_pred: set = set()

        # Greedy matching: for each true CP, find closest unmatched pred CP
        for t_idx, tcp in enumerate(true_cps):
            best_dist = tolerance + 1
            best_p = -1
            for p_idx, pcp in enumerate(pred_cps):
                if p_idx in matched_pred:
                    continue
                dist = abs(pcp - tcp)
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    best_p = p_idx
            if best_p >= 0:
                matched_true.add(t_idx)
                matched_pred.add(best_p)
                matched_delays.append(best_dist)

        n_missed = len(true_cps) - len(matched_true)
        n_fp = len(pred_cps) - len(matched_pred)
        acc = len(matched_true) / max(len(true_cps), 1)
        return float(acc), matched_delays, n_fp, n_missed

    # ---- Utility -----------------------------------------------------------

    @staticmethod
    def regime_duration_statistics(
        labels: NDArray[np.int64],
    ) -> Dict[str, Any]:
        """Compute per-regime duration statistics.

        Returns dict with keys per regime label mapping to
        {mean_dur, std_dur, min_dur, max_dur, count}.
        """
        result: Dict[str, Any] = {}
        if len(labels) == 0:
            return result

        current = int(labels[0])
        start = 0
        durations: Dict[int, List[int]] = {}

        for i in range(1, len(labels)):
            if int(labels[i]) != current:
                durations.setdefault(current, []).append(i - start)
                current = int(labels[i])
                start = i
        durations.setdefault(current, []).append(len(labels) - start)

        for lbl, durs in durations.items():
            arr = np.array(durs, dtype=np.float64)
            result[str(lbl)] = {
                "mean_dur": float(np.mean(arr)),
                "std_dur": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "min_dur": int(np.min(arr)),
                "max_dur": int(np.max(arr)),
                "count": len(arr),
            }
        return result
