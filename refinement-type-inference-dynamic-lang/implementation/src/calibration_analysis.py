"""
Calibration analysis for the TensorGuard neuro-symbolic pipeline.

Evaluates how well the pipeline's confidence-weighted predictions align
with actual outcomes. Implements Brier score, ECE, MCE, reliability
diagram data, and calibration-sharpness decomposition.

Usage::

    from src.calibration_analysis import (
        CalibrationReport, Prediction, compute_calibration_report,
        load_predictions_from_results,
    )

    preds = load_predictions_from_results("experiments/results/")
    report = compute_calibration_report(preds, n_bins=10)
    print(report.ece, report.brier_score)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ─── Confidence mapping ──────────────────────────────────────────────────────

# Maps pipeline Confidence enum names to numeric scores.
CONFIDENCE_MAP: Dict[str, float] = {
    "FORMAL": 0.99,
    "HIGH": 0.85,
    "MEDIUM": 0.60,
    "LOW": 0.35,
    "NONE": 0.10,
}


# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass
class Prediction:
    """A single pipeline prediction with ground truth."""
    confidence: float        # predicted probability of the *positive* class
    predicted_class: int     # 0 or 1 (binary) or 0..K-1 (multi-class)
    true_class: int          # ground-truth label
    label_name: str = ""     # optional human-readable label


@dataclass
class ReliabilityBin:
    """One bin of the reliability diagram."""
    bin_lower: float
    bin_upper: float
    avg_confidence: float
    avg_accuracy: float
    count: int
    gap: float               # |avg_accuracy - avg_confidence|


@dataclass
class CalibrationReport:
    """Complete calibration analysis."""
    brier_score: float
    ece: float                                  # expected calibration error
    mce: float                                  # maximum calibration error
    calibration_component: float                # calibration part of Brier decomp
    sharpness_component: float                  # sharpness (resolution) part
    uncertainty_component: float                # base-rate uncertainty
    reliability_diagram: List[ReliabilityBin]
    n_predictions: int
    n_bins: int
    mean_confidence: float
    mean_accuracy: float
    overconfidence_ratio: float                 # fraction of bins overconfident

    # Multi-class support
    per_class_ece: Optional[Dict[int, float]] = None

    # Extended calibration metrics
    adaptive_ece: Optional[float] = None  # equal-mass adaptive ECE
    ece_bootstrap_ci: Optional[Tuple[float, float]] = None  # 95% CI via bootstrap
    temperature: Optional[float] = None  # optimal Platt temperature
    calibration_curve: Optional[List[Tuple[float, float]]] = None  # (predicted, actual)

    def to_dict(self) -> dict:
        d = {
            "brier_score": self.brier_score,
            "ece": self.ece,
            "mce": self.mce,
            "calibration_component": self.calibration_component,
            "sharpness_component": self.sharpness_component,
            "uncertainty_component": self.uncertainty_component,
            "n_predictions": self.n_predictions,
            "n_bins": self.n_bins,
            "mean_confidence": self.mean_confidence,
            "mean_accuracy": self.mean_accuracy,
            "overconfidence_ratio": self.overconfidence_ratio,
            "reliability_diagram": [
                {
                    "bin_lower": b.bin_lower,
                    "bin_upper": b.bin_upper,
                    "avg_confidence": b.avg_confidence,
                    "avg_accuracy": b.avg_accuracy,
                    "count": b.count,
                    "gap": b.gap,
                }
                for b in self.reliability_diagram
            ],
        }
        if self.per_class_ece is not None:
            d["per_class_ece"] = {str(k): v for k, v in self.per_class_ece.items()}
        if self.adaptive_ece is not None:
            d["adaptive_ece"] = self.adaptive_ece
        if self.ece_bootstrap_ci is not None:
            d["ece_bootstrap_ci"] = list(self.ece_bootstrap_ci)
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.calibration_curve is not None:
            d["calibration_curve"] = [
                {"predicted": p, "actual": a} for p, a in self.calibration_curve
            ]
        return d


# ─── Core metrics ─────────────────────────────────────────────────────────────

def brier_score(predictions: Sequence[Prediction]) -> float:
    """Mean squared error between confidence and binary outcome."""
    if not predictions:
        return 0.0
    total = 0.0
    for p in predictions:
        outcome = 1.0 if p.predicted_class == p.true_class else 0.0
        total += (p.confidence - outcome) ** 2
    return total / len(predictions)


def _bin_predictions(
    predictions: Sequence[Prediction], n_bins: int
) -> List[ReliabilityBin]:
    """Assign predictions to equal-width bins and compute per-bin stats."""
    bins: List[List[Prediction]] = [[] for _ in range(n_bins)]
    for p in predictions:
        idx = min(int(p.confidence * n_bins), n_bins - 1)
        bins[idx].append(p)

    result: List[ReliabilityBin] = []
    for i, bucket in enumerate(bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        if not bucket:
            result.append(ReliabilityBin(lower, upper, 0.0, 0.0, 0, 0.0))
            continue
        avg_conf = sum(p.confidence for p in bucket) / len(bucket)
        avg_acc = sum(
            1.0 for p in bucket if p.predicted_class == p.true_class
        ) / len(bucket)
        gap = abs(avg_acc - avg_conf)
        result.append(ReliabilityBin(lower, upper, avg_conf, avg_acc, len(bucket), gap))
    return result


def expected_calibration_error(
    predictions: Sequence[Prediction], n_bins: int = 10
) -> Tuple[float, List[ReliabilityBin]]:
    """Weighted average of per-bin |accuracy - confidence|."""
    if not predictions:
        return 0.0, []
    bins = _bin_predictions(predictions, n_bins)
    n = len(predictions)
    ece = sum(b.count / n * b.gap for b in bins)
    return ece, bins


def maximum_calibration_error(bins: Sequence[ReliabilityBin]) -> float:
    """Worst-case calibration error across non-empty bins."""
    non_empty = [b for b in bins if b.count > 0]
    if not non_empty:
        return 0.0
    return max(b.gap for b in non_empty)


def brier_decomposition(
    predictions: Sequence[Prediction], n_bins: int = 10
) -> Tuple[float, float, float]:
    """
    Decompose Brier score into calibration, resolution (sharpness), and uncertainty.

    Uses the Murphy (1973) decomposition:
      Brier = calibration - resolution + uncertainty
    where
      calibration = (1/N) * sum_k n_k * (o_k - c_k)^2
      resolution  = (1/N) * sum_k n_k * (o_k - o_bar)^2
      uncertainty = o_bar * (1 - o_bar)
      o_k = fraction correct in bin k
      c_k = mean confidence in bin k
      o_bar = overall accuracy
    """
    if not predictions:
        return 0.0, 0.0, 0.0

    bins = _bin_predictions(predictions, n_bins)
    n = len(predictions)
    o_bar = sum(
        1.0 for p in predictions if p.predicted_class == p.true_class
    ) / n

    calibration = 0.0
    resolution = 0.0
    for b in bins:
        if b.count == 0:
            continue
        calibration += b.count * (b.avg_accuracy - b.avg_confidence) ** 2
        resolution += b.count * (b.avg_accuracy - o_bar) ** 2

    calibration /= n
    resolution /= n
    uncertainty = o_bar * (1.0 - o_bar)

    return calibration, resolution, uncertainty


def per_class_ece(
    predictions: Sequence[Prediction], n_bins: int = 10
) -> Dict[int, float]:
    """Compute ECE separately for each class (one-vs-rest)."""
    classes = sorted(set(p.true_class for p in predictions))
    result: Dict[int, float] = {}
    for cls in classes:
        # Build binary predictions: positive = belongs to cls
        binary_preds = []
        for p in predictions:
            if p.predicted_class == cls:
                conf = p.confidence
            else:
                conf = 1.0 - p.confidence
            binary_preds.append(Prediction(
                confidence=conf,
                predicted_class=1 if p.predicted_class == cls else 0,
                true_class=1 if p.true_class == cls else 0,
            ))
        ece_val, _ = expected_calibration_error(binary_preds, n_bins)
        result[cls] = ece_val
    return result


# ─── Adaptive ECE (equal-mass binning) ────────────────────────────────────────

def adaptive_ece(predictions: Sequence[Prediction], n_bins: int = 10) -> float:
    """Adaptive ECE using equal-mass (quantile) bins.

    Equal-width binning can produce empty or near-empty bins. Adaptive ECE
    assigns roughly N/n_bins predictions per bin, giving more stable estimates
    (Nguyen & O'Connor, 2015).
    """
    if not predictions:
        return 0.0
    sorted_preds = sorted(predictions, key=lambda p: p.confidence)
    n = len(sorted_preds)
    bin_size = max(1, n // n_bins)
    total_gap = 0.0
    for i in range(0, n, bin_size):
        bucket = sorted_preds[i : i + bin_size]
        if not bucket:
            continue
        avg_conf = sum(p.confidence for p in bucket) / len(bucket)
        avg_acc = sum(
            1.0 for p in bucket if p.predicted_class == p.true_class
        ) / len(bucket)
        total_gap += len(bucket) * abs(avg_acc - avg_conf)
    return total_gap / n


# ─── Bootstrap confidence interval for ECE ────────────────────────────────────

def bootstrap_ece_ci(
    predictions: Sequence[Prediction],
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute (1 - alpha)% bootstrap confidence interval for ECE.

    Uses the percentile method with stratified resampling.
    Returns (lower, upper) bounds.
    """
    if len(predictions) < 2:
        ece_val, _ = expected_calibration_error(predictions, n_bins)
        return (ece_val, ece_val)

    import random
    rng = random.Random(seed)
    preds_list = list(predictions)
    n = len(preds_list)
    ece_samples = []
    for _ in range(n_bootstrap):
        sample = [preds_list[rng.randint(0, n - 1)] for _ in range(n)]
        ece_val, _ = expected_calibration_error(sample, n_bins)
        ece_samples.append(ece_val)

    ece_samples.sort()
    lo_idx = max(0, int(n_bootstrap * alpha / 2) - 1)
    hi_idx = min(n_bootstrap - 1, int(n_bootstrap * (1 - alpha / 2)))
    return (ece_samples[lo_idx], ece_samples[hi_idx])


# ─── Temperature scaling (Platt scaling) ──────────────────────────────────────

def find_optimal_temperature(
    predictions: Sequence[Prediction],
    lr: float = 0.01,
    max_iter: int = 200,
) -> float:
    """Find optimal temperature T for Platt scaling via NLL minimization.

    Minimizes negative log-likelihood:
      NLL = -sum[ y_i * log(sigma(logit_i / T)) + (1-y_i) * log(1 - sigma(logit_i / T)) ]
    where sigma is the sigmoid, and logit_i = log(p_i / (1 - p_i)).

    Uses gradient descent with line search.
    Returns optimal temperature (T >= 0.01).
    """
    if not predictions:
        return 1.0

    eps = 1e-7
    logits = []
    labels = []
    for p in predictions:
        conf = max(eps, min(1 - eps, p.confidence))
        logits.append(math.log(conf / (1 - conf)))
        labels.append(1.0 if p.predicted_class == p.true_class else 0.0)

    T = 1.0
    for _ in range(max_iter):
        nll = 0.0
        grad = 0.0
        for logit, y in zip(logits, labels):
            scaled = logit / T
            scaled = max(-20.0, min(20.0, scaled))  # clamp
            sigma = 1.0 / (1.0 + math.exp(-scaled))
            sigma = max(eps, min(1 - eps, sigma))
            nll -= y * math.log(sigma) + (1 - y) * math.log(1 - sigma)
            # d(NLL)/dT = sum[ (sigma - y) * (-logit / T^2) ]
            grad += (sigma - y) * (-logit / (T * T))
        T -= lr * grad / len(logits)
        T = max(0.01, T)  # prevent collapse
    return T


def apply_temperature(
    predictions: Sequence[Prediction], temperature: float
) -> List[Prediction]:
    """Apply temperature scaling to predictions."""
    eps = 1e-7
    result = []
    for p in predictions:
        conf = max(eps, min(1 - eps, p.confidence))
        logit = math.log(conf / (1 - conf))
        scaled = logit / temperature
        scaled = max(-20.0, min(20.0, scaled))
        new_conf = 1.0 / (1.0 + math.exp(-scaled))
        result.append(Prediction(
            confidence=new_conf,
            predicted_class=p.predicted_class,
            true_class=p.true_class,
            label_name=p.label_name,
        ))
    return result


# ─── Calibration curve data ───────────────────────────────────────────────────

def calibration_curve_data(
    predictions: Sequence[Prediction], n_points: int = 10
) -> List[Tuple[float, float]]:
    """Generate (mean_predicted, fraction_positive) pairs for calibration plot.

    Uses equal-mass bins to produce smooth curves even with imbalanced data.
    """
    if not predictions:
        return []
    sorted_preds = sorted(predictions, key=lambda p: p.confidence)
    n = len(sorted_preds)
    bin_size = max(1, n // n_points)
    curve = []
    for i in range(0, n, bin_size):
        bucket = sorted_preds[i : i + bin_size]
        if not bucket:
            continue
        mean_pred = sum(p.confidence for p in bucket) / len(bucket)
        frac_pos = sum(
            1.0 for p in bucket if p.predicted_class == p.true_class
        ) / len(bucket)
        curve.append((mean_pred, frac_pos))
    return curve


# ─── Top-level report ─────────────────────────────────────────────────────────

def compute_calibration_report(
    predictions: Sequence[Prediction], n_bins: int = 10
) -> CalibrationReport:
    """Compute all calibration metrics and return a CalibrationReport."""
    if not predictions:
        return CalibrationReport(
            brier_score=0.0,
            ece=0.0,
            mce=0.0,
            calibration_component=0.0,
            sharpness_component=0.0,
            uncertainty_component=0.0,
            reliability_diagram=[],
            n_predictions=0,
            n_bins=n_bins,
            mean_confidence=0.0,
            mean_accuracy=0.0,
            overconfidence_ratio=0.0,
        )

    bs = brier_score(predictions)
    ece_val, bins = expected_calibration_error(predictions, n_bins)
    mce_val = maximum_calibration_error(bins)
    cal, res, unc = brier_decomposition(predictions, n_bins)

    mean_conf = sum(p.confidence for p in predictions) / len(predictions)
    mean_acc = sum(
        1.0 for p in predictions if p.predicted_class == p.true_class
    ) / len(predictions)

    non_empty = [b for b in bins if b.count > 0]
    overconf = (
        sum(1 for b in non_empty if b.avg_confidence > b.avg_accuracy) / len(non_empty)
        if non_empty
        else 0.0
    )

    # Multi-class ECE (only if more than 2 classes)
    classes = set(p.true_class for p in predictions)
    pc_ece = per_class_ece(predictions, n_bins) if len(classes) > 2 else None

    # Extended metrics
    a_ece = adaptive_ece(predictions, n_bins)
    boot_ci = bootstrap_ece_ci(predictions, n_bins)
    temp = find_optimal_temperature(predictions)
    curve = calibration_curve_data(predictions, n_points=n_bins)

    return CalibrationReport(
        brier_score=bs,
        ece=ece_val,
        mce=mce_val,
        calibration_component=cal,
        sharpness_component=res,
        uncertainty_component=unc,
        reliability_diagram=bins,
        n_predictions=len(predictions),
        n_bins=n_bins,
        mean_confidence=mean_conf,
        mean_accuracy=mean_acc,
        overconfidence_ratio=overconf,
        per_class_ece=pc_ece,
        adaptive_ece=a_ece,
        ece_bootstrap_ci=boot_ci,
        temperature=temp,
        calibration_curve=curve,
    )


# ─── Loading predictions from experiment results ─────────────────────────────

def _confidence_name_to_score(name: str) -> float:
    """Map pipeline confidence enum name to a numeric score."""
    return CONFIDENCE_MAP.get(name.upper(), 0.5)


def load_predictions_from_results(
    results_dir: str,
    file_pattern: str = "",
) -> List[Prediction]:
    """
    Load predictions from pipeline result JSON files.

    Looks for files matching *pipeline* or *neurosym* in *results_dir*
    (and parent dir) and extracts per-benchmark predictions.
    """
    predictions: List[Prediction] = []
    search_dirs = [results_dir]
    parent = str(Path(results_dir).parent)
    if parent != results_dir:
        search_dirs.append(parent)

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".json"):
                continue
            if file_pattern and file_pattern not in fname:
                continue
            if not file_pattern and not any(
                kw in fname.lower() for kw in ("pipeline", "neurosym")
            ):
                continue
            fpath = os.path.join(d, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            benchmarks = data.get("benchmarks", [])
            for bm in benchmarks:
                has_bug = bm.get("has_bug")
                llm_predicts = bm.get("llm_predicts_bug")
                if has_bug is None or llm_predicts is None:
                    continue

                # Confidence from pipeline or LLM
                conf_name = bm.get("pipeline_confidence", "")
                if conf_name and conf_name.upper() in CONFIDENCE_MAP:
                    conf = _confidence_name_to_score(conf_name)
                else:
                    conf = bm.get("llm_confidence", 0.5)

                pred_class = 1 if llm_predicts else 0
                true_class = 1 if has_bug else 0

                predictions.append(Prediction(
                    confidence=conf,
                    predicted_class=pred_class,
                    true_class=true_class,
                    label_name=bm.get("name", ""),
                ))

    return predictions
