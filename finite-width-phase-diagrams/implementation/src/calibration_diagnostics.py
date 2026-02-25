"""
Calibration diagnostics for soft phase posteriors.

Implements:
- Reliability diagrams (predicted probability vs observed frequency)
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
- Adaptive binning for reliability diagrams
- Brier score decomposition (reliability, resolution, uncertainty)
- Per-class calibration analysis

References:
    Naeini et al., "Obtaining Well Calibrated Probabilities Using Bayesian Binning", AAAI 2015
    Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class CalibrationBin:
    """Single bin in a reliability diagram."""
    bin_lower: float
    bin_upper: float
    bin_center: float
    mean_predicted: float  # mean predicted probability in this bin
    mean_observed: float   # fraction of positive outcomes in this bin
    count: int             # number of predictions in this bin
    gap: float             # |mean_predicted - mean_observed|


@dataclass
class ReliabilityDiagram:
    """Complete reliability diagram data."""
    bins: List[CalibrationBin]
    ece: float              # Expected Calibration Error
    mce: float              # Maximum Calibration Error
    ace: float              # Average Calibration Error (equal-count bins)
    brier_score: float      # Brier score
    brier_reliability: float  # reliability component
    brier_resolution: float   # resolution component
    brier_uncertainty: float  # uncertainty component
    n_samples: int
    class_name: str = ""


@dataclass
class CalibrationReport:
    """Full calibration report across all phase classes."""
    overall_ece: float
    overall_mce: float
    overall_brier: float
    per_class: Dict[str, ReliabilityDiagram]
    n_total: int
    is_well_calibrated: bool  # True if ECE < 0.05
    calibration_summary: str


class CalibrationDiagnostics:
    """Compute calibration diagnostics for soft phase posteriors."""

    def __init__(self, n_bins: int = 10, adaptive: bool = True):
        """
        Args:
            n_bins: Number of bins for reliability diagram.
            adaptive: If True, use adaptive (equal-count) binning.
        """
        self.n_bins = n_bins
        self.adaptive = adaptive

    def compute_reliability_diagram(
        self,
        predicted_probs: np.ndarray,
        true_labels: np.ndarray,
        class_name: str = "",
    ) -> ReliabilityDiagram:
        """Compute reliability diagram for binary predictions.

        Args:
            predicted_probs: Array of predicted probabilities for the positive class.
            true_labels: Array of binary true labels (0 or 1).
            class_name: Name of the class being evaluated.

        Returns:
            ReliabilityDiagram with bins and calibration metrics.
        """
        n = len(predicted_probs)
        if n == 0:
            return self._empty_diagram(class_name)

        predicted_probs = np.asarray(predicted_probs, dtype=float)
        true_labels = np.asarray(true_labels, dtype=float)

        if self.adaptive:
            bins = self._adaptive_bins(predicted_probs, true_labels)
        else:
            bins = self._uniform_bins(predicted_probs, true_labels)

        # ECE: weighted average of |gap|
        ece = sum(b.count * b.gap for b in bins) / max(n, 1)

        # MCE: maximum gap
        mce = max((b.gap for b in bins), default=0.0)

        # ACE: simple average of |gap| (equal-weight bins)
        ace = np.mean([b.gap for b in bins]) if bins else 0.0

        # Brier score and decomposition
        brier = float(np.mean((predicted_probs - true_labels) ** 2))
        base_rate = float(np.mean(true_labels))
        brier_uncertainty = base_rate * (1 - base_rate)

        brier_reliability = sum(
            b.count * (b.mean_predicted - b.mean_observed) ** 2 for b in bins
        ) / max(n, 1)

        brier_resolution = sum(
            b.count * (b.mean_observed - base_rate) ** 2 for b in bins
        ) / max(n, 1)

        return ReliabilityDiagram(
            bins=bins,
            ece=ece,
            mce=mce,
            ace=ace,
            brier_score=brier,
            brier_reliability=brier_reliability,
            brier_resolution=brier_resolution,
            brier_uncertainty=brier_uncertainty,
            n_samples=n,
            class_name=class_name,
        )

    def compute_multiclass_calibration(
        self,
        predicted_probs_dict: Dict[str, np.ndarray],
        true_labels: np.ndarray,
        class_names: List[str],
    ) -> CalibrationReport:
        """Compute calibration report for multi-class phase classification.

        Args:
            predicted_probs_dict: Dict mapping class_name -> array of predicted probabilities.
            true_labels: Array of true class labels (strings).
            class_names: List of class names.

        Returns:
            CalibrationReport with per-class and overall calibration.
        """
        n = len(true_labels)
        per_class = {}

        for cls in class_names:
            # Binary: this class vs not
            probs = predicted_probs_dict.get(cls, np.zeros(n))
            binary_labels = (np.asarray(true_labels) == cls).astype(float)
            diagram = self.compute_reliability_diagram(probs, binary_labels, cls)
            per_class[cls] = diagram

        # Overall ECE (confidence-weighted across classes)
        # Use the predicted class probability as confidence
        confidences = np.zeros(n)
        correct = np.zeros(n)
        for i in range(n):
            max_prob = 0.0
            pred_cls = class_names[0]
            for cls in class_names:
                p = predicted_probs_dict.get(cls, np.zeros(n))[i]
                if p > max_prob:
                    max_prob = p
                    pred_cls = cls
            confidences[i] = max_prob
            correct[i] = 1.0 if pred_cls == true_labels[i] else 0.0

        overall_diagram = self.compute_reliability_diagram(
            confidences, correct, "overall"
        )

        overall_brier = np.mean([
            per_class[cls].brier_score for cls in class_names
        ])

        is_well_calibrated = overall_diagram.ece < 0.05

        summary_parts = [f"ECE={overall_diagram.ece:.4f}"]
        for cls in class_names:
            summary_parts.append(f"{cls}: ECE={per_class[cls].ece:.4f}")
        if is_well_calibrated:
            summary_parts.append("WELL CALIBRATED")
        else:
            summary_parts.append("MISCALIBRATED — consider temperature scaling")

        return CalibrationReport(
            overall_ece=overall_diagram.ece,
            overall_mce=overall_diagram.mce,
            overall_brier=overall_brier,
            per_class=per_class,
            n_total=n,
            is_well_calibrated=is_well_calibrated,
            calibration_summary="; ".join(summary_parts),
        )

    def _uniform_bins(self, probs: np.ndarray, labels: np.ndarray) -> List[CalibrationBin]:
        """Create uniform-width bins."""
        bins = []
        edges = np.linspace(0, 1, self.n_bins + 1)

        for i in range(self.n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == self.n_bins - 1:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)

            count = int(np.sum(mask))
            if count == 0:
                bins.append(CalibrationBin(
                    bin_lower=lo, bin_upper=hi,
                    bin_center=(lo + hi) / 2,
                    mean_predicted=(lo + hi) / 2,
                    mean_observed=0.0, count=0, gap=0.0,
                ))
                continue

            mean_pred = float(np.mean(probs[mask]))
            mean_obs = float(np.mean(labels[mask]))
            gap = abs(mean_pred - mean_obs)
            bins.append(CalibrationBin(
                bin_lower=lo, bin_upper=hi,
                bin_center=(lo + hi) / 2,
                mean_predicted=mean_pred,
                mean_observed=mean_obs,
                count=count, gap=gap,
            ))

        return bins

    def _adaptive_bins(self, probs: np.ndarray, labels: np.ndarray) -> List[CalibrationBin]:
        """Create adaptive (equal-count) bins."""
        n = len(probs)
        if n == 0:
            return []

        sorted_idx = np.argsort(probs)
        bin_size = max(n // self.n_bins, 1)
        bins = []

        for i in range(0, n, bin_size):
            idx = sorted_idx[i:i + bin_size]
            if len(idx) == 0:
                continue

            bin_probs = probs[idx]
            bin_labels = labels[idx]
            lo = float(np.min(bin_probs))
            hi = float(np.max(bin_probs))
            mean_pred = float(np.mean(bin_probs))
            mean_obs = float(np.mean(bin_labels))
            gap = abs(mean_pred - mean_obs)

            bins.append(CalibrationBin(
                bin_lower=lo, bin_upper=hi,
                bin_center=(lo + hi) / 2,
                mean_predicted=mean_pred,
                mean_observed=mean_obs,
                count=len(idx), gap=gap,
            ))

        return bins

    def _empty_diagram(self, class_name: str) -> ReliabilityDiagram:
        return ReliabilityDiagram(
            bins=[], ece=0.0, mce=0.0, ace=0.0,
            brier_score=0.0, brier_reliability=0.0,
            brier_resolution=0.0, brier_uncertainty=0.0,
            n_samples=0, class_name=class_name,
        )


def compute_ece(predicted_probs: np.ndarray, true_labels: np.ndarray,
                n_bins: int = 10) -> float:
    """Quick ECE computation."""
    diag = CalibrationDiagnostics(n_bins=n_bins, adaptive=False)
    result = diag.compute_reliability_diagram(predicted_probs, true_labels)
    return result.ece
