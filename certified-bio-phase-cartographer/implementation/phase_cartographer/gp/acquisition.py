"""
Acquisition functions for GP-guided phase-boundary exploration.

Expected Improvement (EI) targets parameter regions where the GP predicts
high uncertainty about regime identity — i.e., near phase boundaries.
"""

import numpy as np
from typing import List, Tuple, Optional

from .surrogate import GPSurrogate, GPPrediction


def expected_improvement(pred: GPPrediction, y_best: float,
                         xi: float = 0.01) -> float:
    """
    Expected Improvement acquisition function.

    EI(x) = (μ(x) - y_best - ξ) Φ(Z) + σ(x) φ(Z)
    where Z = (μ(x) - y_best - ξ) / σ(x).

    For phase-boundary detection, y_best is the current best boundary score
    (maximum GP posterior variance among certified cells).
    """
    if pred.std < 1e-10:
        return 0.0

    z = (pred.mean - y_best - xi) / pred.std
    from scipy.stats import norm
    return float((pred.mean - y_best - xi) * norm.cdf(z) + pred.std * norm.pdf(z))


def upper_confidence_bound(pred: GPPrediction, beta: float = 2.0) -> float:
    """
    UCB acquisition: μ(x) + β·σ(x).

    Higher β encourages exploration. β = 2.0 gives ~95% confidence bound.
    """
    return pred.mean + beta * pred.std


def boundary_uncertainty(pred: GPPrediction) -> float:
    """
    Boundary-focused acquisition: high score when GP prediction is near
    a regime transition (mean near integer boundary) AND uncertain.

    score = σ(x) × (1 - |μ(x) - round(μ(x))|)
    """
    nearest_int = round(pred.mean)
    boundary_proximity = 1.0 - min(1.0, abs(pred.mean - nearest_int))
    return pred.std * boundary_proximity


def phase_boundary_score(pred: GPPrediction,
                         eigenvalue_sensitivity: float = 0.0,
                         sensitivity_weight: float = 0.3) -> float:
    """Combine boundary_uncertainty with eigenvalue sensitivity.

    score = (1 - w) × boundary_uncertainty(pred) + w × σ_eig
    where σ_eig = min(eigenvalue_sensitivity, 10) / 10  (normalised).

    A high ``eigenvalue_sensitivity`` indicates proximity to a bifurcation,
    so this composite score favours cells that the GP *and* the eigenvalue
    analysis both flag as interesting.
    """
    bu = boundary_uncertainty(pred)
    # Normalise eigenvalue sensitivity to [0, 1]
    normed_eig = min(eigenvalue_sensitivity, 10.0) / 10.0 if (
        eigenvalue_sensitivity < float('inf')) else 1.0
    return (1.0 - sensitivity_weight) * bu + sensitivity_weight * normed_eig


class AcquisitionOptimizer:
    """
    Selects the next parameter box to refine based on acquisition function scores.
    """

    def __init__(self, gp: GPSurrogate, acquisition: str = "boundary_uncertainty"):
        self.gp = gp
        if acquisition == "ei":
            self._acq_fn = lambda p, yb: expected_improvement(p, yb)
        elif acquisition == "ucb":
            self._acq_fn = lambda p, yb: upper_confidence_bound(p)
        else:
            self._acq_fn = lambda p, yb: boundary_uncertainty(p)

    def rank_boxes(self, box_midpoints: np.ndarray,
                   y_best: float = 0.0) -> List[Tuple[int, float]]:
        """
        Rank parameter boxes by acquisition function score.

        Returns list of (box_index, score) sorted descending by score.
        """
        preds = self.gp.predict_batch(box_midpoints)
        scores = [(i, self._acq_fn(preds[i], y_best))
                  for i in range(len(preds))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def select_next(self, box_midpoints: np.ndarray,
                    y_best: float = 0.0) -> int:
        """Return index of the highest-scoring box."""
        ranked = self.rank_boxes(box_midpoints, y_best)
        return ranked[0][0] if ranked else 0
