"""Evaluation metrics for CPA benchmarks.

Provides metrics for classification accuracy, tipping-point detection,
robustness certificate quality, and QD archive performance.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from scipy import stats as sp_stats


# =====================================================================
# Classification metrics
# =====================================================================


class ClassificationMetrics:
    """Evaluation metrics for mechanism classification.

    Compares predicted classifications against ground truth.

    Examples
    --------
    >>> metrics = ClassificationMetrics()
    >>> results = metrics.evaluate(
    ...     predicted={"X0": "invariant", "X1": "plastic"},
    ...     ground_truth={"X0": "invariant", "X1": "invariant"},
    ... )
    """

    VALID_CLASSES = {
        "invariant",
        "structurally_plastic",
        "parametrically_plastic",
        "fully_plastic",
        "emergent",
        "context_sensitive",
        "unclassified",
    }

    def evaluate(
        self,
        predicted: Dict[str, str],
        ground_truth: Dict[str, str],
        classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute classification metrics.

        Parameters
        ----------
        predicted : dict of str → str
            Variable → predicted classification.
        ground_truth : dict of str → str
            Variable → true classification.
        classes : list of str, optional
            Classes to evaluate (default: all observed classes).

        Returns
        -------
        dict
            Dictionary with precision, recall, F1 (macro, micro, weighted),
            accuracy, confusion matrix, and per-class metrics.
        """
        common_vars = sorted(set(predicted.keys()) & set(ground_truth.keys()))
        if not common_vars:
            return {"error": "No common variables"}

        y_pred = [predicted[v] for v in common_vars]
        y_true = [ground_truth[v] for v in common_vars]

        if classes is None:
            classes = sorted(set(y_true) | set(y_pred))

        n = len(common_vars)
        accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / n

        # Per-class metrics
        per_class: Dict[str, Dict[str, float]] = {}
        for cls in classes:
            tp = sum(1 for p, t in zip(y_pred, y_true) if p == cls and t == cls)
            fp = sum(1 for p, t in zip(y_pred, y_true) if p == cls and t != cls)
            fn = sum(1 for p, t in zip(y_pred, y_true) if p != cls and t == cls)
            tn = sum(1 for p, t in zip(y_pred, y_true) if p != cls and t != cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            support = tp + fn

            per_class[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }

        # Macro averages
        precisions = [per_class[c]["precision"] for c in classes if per_class[c]["support"] > 0]
        recalls = [per_class[c]["recall"] for c in classes if per_class[c]["support"] > 0]
        f1s = [per_class[c]["f1"] for c in classes if per_class[c]["support"] > 0]

        macro_precision = float(np.mean(precisions)) if precisions else 0.0
        macro_recall = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        # Micro averages
        total_tp = sum(per_class[c]["tp"] for c in classes)
        total_fp = sum(per_class[c]["fp"] for c in classes)
        total_fn = sum(per_class[c]["fn"] for c in classes)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        # Weighted averages
        total_support = sum(per_class[c]["support"] for c in classes)
        if total_support > 0:
            weighted_precision = sum(
                per_class[c]["precision"] * per_class[c]["support"]
                for c in classes
            ) / total_support
            weighted_recall = sum(
                per_class[c]["recall"] * per_class[c]["support"]
                for c in classes
            ) / total_support
            weighted_f1 = sum(
                per_class[c]["f1"] * per_class[c]["support"]
                for c in classes
            ) / total_support
        else:
            weighted_precision = weighted_recall = weighted_f1 = 0.0

        # Confusion matrix
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)
        for p, t in zip(y_pred, y_true):
            if p in cls_to_idx and t in cls_to_idx:
                conf_matrix[cls_to_idx[t], cls_to_idx[p]] += 1

        return {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "per_class": per_class,
            "confusion_matrix": conf_matrix,
            "classes": classes,
            "n_variables": n,
        }

    def binary_evaluate(
        self,
        predicted: Dict[str, str],
        ground_truth: Dict[str, str],
        positive_class: str = "invariant",
    ) -> Dict[str, float]:
        """Binary classification evaluation (one class vs rest).

        Parameters
        ----------
        predicted : dict
            Predicted classifications.
        ground_truth : dict
            True classifications.
        positive_class : str
            Class to treat as positive.

        Returns
        -------
        dict
        """
        common = sorted(set(predicted.keys()) & set(ground_truth.keys()))
        y_pred = [1 if predicted[v] == positive_class else 0 for v in common]
        y_true = [1 if ground_truth[v] == positive_class else 0 for v in common]

        tp = sum(p == 1 and t == 1 for p, t in zip(y_pred, y_true))
        fp = sum(p == 1 and t == 0 for p, t in zip(y_pred, y_true))
        fn = sum(p == 0 and t == 1 for p, t in zip(y_pred, y_true))
        tn = sum(p == 0 and t == 0 for p, t in zip(y_pred, y_true))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / max(len(common), 1)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        specificity = 1.0 - fpr

        mcc_num = tp * tn - fp * fn
        mcc_den = math.sqrt(
            max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1)
        )
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "fpr": fpr,
            "mcc": mcc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }


import math


# =====================================================================
# Tipping-point metrics
# =====================================================================


class TippingPointMetrics:
    """Metrics for tipping-point detection evaluation.

    Compares detected changepoint locations against ground truth.

    Examples
    --------
    >>> metrics = TippingPointMetrics()
    >>> results = metrics.evaluate(
    ...     predicted=[3, 7],
    ...     ground_truth=[3, 8],
    ...     K=10,
    ... )
    """

    def evaluate(
        self,
        predicted: List[int],
        ground_truth: List[int],
        K: int,
        tolerance: int = 1,
    ) -> Dict[str, float]:
        """Evaluate tipping-point detection.

        Parameters
        ----------
        predicted : list of int
            Detected changepoint locations.
        ground_truth : list of int
            True changepoint locations.
        K : int
            Total number of contexts.
        tolerance : int
            Location tolerance for matching (±tolerance).

        Returns
        -------
        dict
            MAD (mean absolute deviation), precision, recall, F1,
            localization accuracy, and Hausdorff distance.
        """
        if not ground_truth and not predicted:
            return {
                "mad": 0.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "localization_accuracy": 1.0,
                "hausdorff": 0.0,
                "n_predicted": 0,
                "n_true": 0,
            }

        if not ground_truth:
            return {
                "mad": float("inf"),
                "precision": 0.0,
                "recall": 1.0,
                "f1": 0.0,
                "localization_accuracy": 0.0,
                "hausdorff": float("inf"),
                "n_predicted": len(predicted),
                "n_true": 0,
            }

        if not predicted:
            return {
                "mad": float("inf"),
                "precision": 1.0,
                "recall": 0.0,
                "f1": 0.0,
                "localization_accuracy": 0.0,
                "hausdorff": float("inf"),
                "n_predicted": 0,
                "n_true": len(ground_truth),
            }

        # Match predicted to true changepoints
        true_matched: Set[int] = set()
        pred_matched: Set[int] = set()
        deviations: List[float] = []

        for gt in ground_truth:
            best_dist = float("inf")
            best_pred = None
            for pred in predicted:
                dist = abs(pred - gt)
                if dist <= tolerance and dist < best_dist and pred not in pred_matched:
                    best_dist = dist
                    best_pred = pred

            if best_pred is not None:
                true_matched.add(gt)
                pred_matched.add(best_pred)
                deviations.append(best_dist)

        tp = len(true_matched)
        fp = len(predicted) - len(pred_matched)
        fn = len(ground_truth) - len(true_matched)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # MAD: mean absolute deviation of matched pairs
        mad = float(np.mean(deviations)) if deviations else float("inf")

        # Localization accuracy: fraction correctly located within tolerance
        loc_accuracy = tp / max(len(ground_truth), 1)

        # Hausdorff distance
        d_pred_to_true = [
            min(abs(p - g) for g in ground_truth) for p in predicted
        ]
        d_true_to_pred = [
            min(abs(g - p) for p in predicted) for g in ground_truth
        ]
        hausdorff = max(
            max(d_pred_to_true) if d_pred_to_true else 0,
            max(d_true_to_pred) if d_true_to_pred else 0,
        )

        return {
            "mad": mad,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "localization_accuracy": loc_accuracy,
            "hausdorff": float(hausdorff),
            "n_predicted": len(predicted),
            "n_true": len(ground_truth),
            "n_matched": tp,
        }

    def auc_roc(
        self,
        scores: np.ndarray,
        true_locations: List[int],
        K: int,
    ) -> float:
        """Compute AUC-ROC for tipping-point detection.

        Parameters
        ----------
        scores : np.ndarray
            Score for each context boundary (length K-1).
        true_locations : list of int
            True changepoint locations.
        K : int
            Number of contexts.

        Returns
        -------
        float
            AUC-ROC value.
        """
        if len(true_locations) == 0 or len(scores) == 0:
            return 0.5

        labels = np.zeros(K - 1)
        for loc in true_locations:
            if 0 < loc < K:
                labels[loc - 1] = 1

        if np.sum(labels) == 0 or np.sum(labels) == len(labels):
            return 0.5

        # Sort by score descending
        order = np.argsort(-scores)
        sorted_labels = labels[order]

        tp_cumsum = np.cumsum(sorted_labels)
        fp_cumsum = np.cumsum(1 - sorted_labels)

        n_pos = np.sum(labels)
        n_neg = len(labels) - n_pos

        tpr = tp_cumsum / n_pos
        fpr = fp_cumsum / n_neg

        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])

        auc = float(np.trapz(tpr, fpr))
        return auc


# =====================================================================
# Certificate metrics
# =====================================================================


class CertificateMetrics:
    """Metrics for robustness certificate evaluation.

    Examples
    --------
    >>> metrics = CertificateMetrics()
    >>> results = metrics.evaluate(
    ...     certificates={"X0": True, "X1": False},
    ...     ground_truth_stable={"X0", "X1"},
    ... )
    """

    def evaluate(
        self,
        certificates: Dict[str, bool],
        ground_truth_stable: Set[str],
        descriptor_cis: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
        ground_truth_values: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Evaluate certificate quality.

        Parameters
        ----------
        certificates : dict of str → bool
            Variable → certified flag.
        ground_truth_stable : set of str
            Variables known to have stable mechanisms.
        descriptor_cis : dict, optional
            Variable → {component: (lo, hi)} CIs for calibration.
        ground_truth_values : dict, optional
            Variable → true descriptor values for calibration.

        Returns
        -------
        dict
            Coverage, calibration error, and other metrics.
        """
        variables = sorted(certificates.keys())
        n_certified = sum(1 for v in variables if certificates[v])
        n_total = len(variables)

        # Coverage: fraction of truly stable variables that are certified
        truly_stable = [v for v in variables if v in ground_truth_stable]
        if truly_stable:
            coverage = sum(
                1 for v in truly_stable if certificates.get(v, False)
            ) / len(truly_stable)
        else:
            coverage = 0.0

        # False certification rate
        truly_unstable = [v for v in variables if v not in ground_truth_stable]
        if truly_unstable:
            false_cert_rate = sum(
                1 for v in truly_unstable if certificates.get(v, False)
            ) / len(truly_unstable)
        else:
            false_cert_rate = 0.0

        # Calibration error (if CIs and true values provided)
        calibration_error = None
        if descriptor_cis and ground_truth_values:
            calibration_error = self._calibration_error(
                descriptor_cis, ground_truth_values
            )

        result = {
            "n_certified": n_certified,
            "n_total": n_total,
            "certification_rate": n_certified / max(n_total, 1),
            "coverage": coverage,
            "false_certification_rate": false_cert_rate,
            "n_truly_stable": len(truly_stable),
            "n_truly_unstable": len(truly_unstable),
        }

        if calibration_error is not None:
            result["calibration_error"] = calibration_error

        return result

    def _calibration_error(
        self,
        cis: Dict[str, Dict[str, Tuple[float, float]]],
        true_values: Dict[str, np.ndarray],
    ) -> float:
        """Compute calibration error of confidence intervals.

        The calibration error measures whether 95% CIs actually
        contain the true value 95% of the time.

        Parameters
        ----------
        cis : dict
            Variable → {component: (lo, hi)}.
        true_values : dict
            Variable → true 4D descriptor.

        Returns
        -------
        float
            Absolute calibration error.
        """
        components = ["structural", "parametric", "emergence", "sensitivity"]
        n_inside = 0
        n_total = 0

        for var, var_cis in cis.items():
            if var not in true_values:
                continue
            true_vec = np.asarray(true_values[var])

            for c_idx, comp in enumerate(components):
                if comp not in var_cis:
                    continue
                lo, hi = var_cis[comp]
                true_val = true_vec[c_idx] if c_idx < len(true_vec) else 0.0

                if lo <= true_val <= hi:
                    n_inside += 1
                n_total += 1

        if n_total == 0:
            return 0.0

        empirical_coverage = n_inside / n_total
        target_coverage = 0.95
        return abs(empirical_coverage - target_coverage)


# =====================================================================
# Archive metrics
# =====================================================================


class ArchiveMetrics:
    """Metrics for QD archive evaluation.

    Examples
    --------
    >>> metrics = ArchiveMetrics()
    >>> results = metrics.evaluate(
    ...     archive_descriptors=descriptors,
    ...     archive_fitnesses=fitnesses,
    ...     archive_capacity=256,
    ... )
    """

    def evaluate(
        self,
        archive_descriptors: np.ndarray,
        archive_fitnesses: np.ndarray,
        archive_capacity: int,
        reference_descriptors: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate QD archive quality.

        Parameters
        ----------
        archive_descriptors : np.ndarray
            (n_entries, d) descriptor vectors.
        archive_fitnesses : np.ndarray
            (n_entries,) fitness values.
        archive_capacity : int
            Maximum archive capacity.
        reference_descriptors : np.ndarray, optional
            Reference descriptors for diversity comparison.

        Returns
        -------
        dict
            Coverage, QD-score, diversity, and quality metrics.
        """
        n = len(archive_fitnesses)

        if n == 0:
            return {
                "coverage": 0.0,
                "qd_score": 0.0,
                "mean_fitness": 0.0,
                "max_fitness": 0.0,
                "diversity": 0.0,
                "n_entries": 0,
            }

        coverage = n / max(archive_capacity, 1)
        qd_score = float(np.sum(archive_fitnesses))
        mean_fitness = float(np.mean(archive_fitnesses))
        max_fitness = float(np.max(archive_fitnesses))

        # Diversity: mean pairwise distance in descriptor space
        diversity = self._compute_diversity(archive_descriptors)

        result = {
            "coverage": coverage,
            "qd_score": qd_score,
            "mean_fitness": mean_fitness,
            "max_fitness": max_fitness,
            "diversity": diversity,
            "n_entries": n,
        }

        if reference_descriptors is not None and reference_descriptors.shape[0] > 0:
            result["reference_coverage"] = self._reference_coverage(
                archive_descriptors, reference_descriptors
            )

        return result

    def _compute_diversity(self, descriptors: np.ndarray) -> float:
        """Compute mean pairwise distance in descriptor space.

        Parameters
        ----------
        descriptors : np.ndarray
            (n, d) descriptor matrix.

        Returns
        -------
        float
            Mean pairwise Euclidean distance.
        """
        n = descriptors.shape[0]
        if n < 2:
            return 0.0

        # Subsample for efficiency
        max_pairs = 5000
        if n * (n - 1) // 2 > max_pairs:
            idx = np.random.choice(n, size=int(np.sqrt(max_pairs * 2)), replace=False)
            sub = descriptors[idx]
        else:
            sub = descriptors

        n_sub = sub.shape[0]
        total_dist = 0.0
        count = 0

        for i in range(n_sub):
            for j in range(i + 1, n_sub):
                total_dist += float(np.linalg.norm(sub[i] - sub[j]))
                count += 1

        return total_dist / max(count, 1)

    def _reference_coverage(
        self,
        archive_descriptors: np.ndarray,
        reference_descriptors: np.ndarray,
        threshold: float = 0.1,
    ) -> float:
        """Fraction of reference points covered by archive.

        Parameters
        ----------
        archive_descriptors : np.ndarray
            (n_archive, d) descriptors.
        reference_descriptors : np.ndarray
            (n_ref, d) reference descriptors.
        threshold : float
            Distance threshold for "covered".

        Returns
        -------
        float
        """
        n_covered = 0
        for ref in reference_descriptors:
            dists = np.linalg.norm(archive_descriptors - ref, axis=1)
            if np.min(dists) <= threshold:
                n_covered += 1

        return n_covered / max(len(reference_descriptors), 1)

    def convergence_metrics(
        self,
        convergence_history: List[float],
    ) -> Dict[str, float]:
        """Compute convergence metrics from QD-score history.

        Parameters
        ----------
        convergence_history : list of float
            QD-score at each iteration.

        Returns
        -------
        dict
            Convergence rate, final QD-score, and stability.
        """
        if not convergence_history:
            return {"final_qd_score": 0.0, "convergence_rate": 0.0, "stability": 0.0}

        history = np.array(convergence_history)
        n = len(history)

        final = float(history[-1])

        # Convergence rate: slope of log(final - qd_score) vs iteration
        if n > 10 and final > 0:
            residuals = np.maximum(final - history[: n // 2], 1e-10)
            log_res = np.log(residuals)
            x = np.arange(len(log_res))
            if np.std(log_res) > 0:
                slope = float(np.polyfit(x, log_res, 1)[0])
                conv_rate = -slope
            else:
                conv_rate = 0.0
        else:
            conv_rate = 0.0

        # Stability: relative change in last 10% of iterations
        last_fraction = history[int(n * 0.9):]
        if len(last_fraction) > 1 and final > 0:
            stability = 1.0 - float(np.std(last_fraction)) / abs(final)
        else:
            stability = 1.0

        return {
            "final_qd_score": final,
            "convergence_rate": conv_rate,
            "stability": max(0.0, stability),
            "n_iterations": n,
        }


# =====================================================================
# Baseline comparisons
# =====================================================================


class BaselineComparisons:
    """Compare CPA results against baseline methods.

    Baselines:
    - IND-PHC: Independent per-context analysis (no alignment).
    - Pooled: Pool all data and run single discovery.
    - Pairwise: Naive pairwise comparison without alignment.

    Examples
    --------
    >>> comp = BaselineComparisons()
    >>> results = comp.compare_all(cpa_result, ground_truth)
    """

    def ind_phc_baseline(
        self,
        context_data: Dict[str, np.ndarray],
        ground_truth_classifications: Dict[str, str],
    ) -> Dict[str, Any]:
        """IND-PHC baseline: classify based on per-context edge variability.

        Parameters
        ----------
        context_data : dict
            Context → (n, p) data matrix.
        ground_truth_classifications : dict
            Variable → true classification.

        Returns
        -------
        dict
            Baseline classification metrics.
        """
        K = len(context_data)
        if K == 0:
            return {"error": "No contexts"}

        first_data = next(iter(context_data.values()))
        p = first_data.shape[1]
        variable_names = [f"X{i}" for i in range(p)]

        # Simple edge-counting heuristic
        edge_presence: Dict[str, List[int]] = {v: [] for v in variable_names}

        for cid, data in context_data.items():
            n = data.shape[0]
            if n < p + 2:
                continue

            try:
                corr = np.corrcoef(data, rowvar=False)
                from scipy.stats import pearsonr

                for j in range(p):
                    n_parents = 0
                    for i in range(p):
                        if i == j:
                            continue
                        r = corr[i, j]
                        df = n - 2
                        if df > 0 and abs(r) > 0:
                            t = r * np.sqrt(df / (1 - r ** 2 + 1e-12))
                            p_val = 2 * (1 - sp_stats.t.cdf(abs(t), df))
                            if p_val < 0.05:
                                n_parents += 1
                    edge_presence[variable_names[j]].append(n_parents)
            except Exception:
                for j in range(p):
                    edge_presence[variable_names[j]].append(0)

        predicted: Dict[str, str] = {}
        for var in variable_names:
            counts = edge_presence[var]
            if not counts:
                predicted[var] = "unclassified"
            elif np.std(counts) > 0.5:
                predicted[var] = "structurally_plastic"
            else:
                predicted[var] = "invariant"

        metrics = ClassificationMetrics()
        return metrics.evaluate(predicted, ground_truth_classifications)

    def pooled_baseline(
        self,
        context_data: Dict[str, np.ndarray],
        ground_truth_classifications: Dict[str, str],
    ) -> Dict[str, Any]:
        """Pooled baseline: pool all data, classify all as invariant.

        Parameters
        ----------
        context_data : dict
            Context → data.
        ground_truth_classifications : dict
            True classifications.

        Returns
        -------
        dict
        """
        first_data = next(iter(context_data.values()))
        p = first_data.shape[1]
        variable_names = [f"X{i}" for i in range(p)]

        predicted = {v: "invariant" for v in variable_names}

        metrics = ClassificationMetrics()
        return metrics.evaluate(predicted, ground_truth_classifications)

    def compare_all(
        self,
        cpa_classifications: Dict[str, str],
        ground_truth_classifications: Dict[str, str],
        context_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare CPA against all baselines.

        Parameters
        ----------
        cpa_classifications : dict
            CPA predicted classifications.
        ground_truth_classifications : dict
            True classifications.
        context_data : dict, optional
            Data for computing baselines.

        Returns
        -------
        dict
            Method → metrics mapping.
        """
        metrics = ClassificationMetrics()

        results: Dict[str, Dict[str, Any]] = {
            "CPA": metrics.evaluate(
                cpa_classifications, ground_truth_classifications
            ),
        }

        if context_data:
            results["IND-PHC"] = self.ind_phc_baseline(
                context_data, ground_truth_classifications
            )
            results["Pooled"] = self.pooled_baseline(
                context_data, ground_truth_classifications
            )

        return results
