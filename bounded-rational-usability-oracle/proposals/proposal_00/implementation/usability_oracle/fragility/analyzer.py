"""
usability_oracle.fragility.analyzer — Core fragility analysis engine.

Implements :class:`FragilityAnalyzer`, which computes a comprehensive
fragility assessment for an MDP by sweeping over the rationality
parameter β and analyzing the resulting cost curve.

The fragility score is defined as the normalized maximum gradient of the
cost function C(β):

.. math::
    F = \\frac{\\max_\\beta |dC/d\\beta|}{C_{max} - C_{min}} \\cdot \\Delta\\beta

where :math:`\\Delta\\beta` is the width of the analysis range.  A score
close to 0 means the cost curve is flat (robust); close to 1 means
there are sharp changes (fragile).

References
----------
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*, 469.
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis*. Wiley.
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import interpolate as sp_interp

from usability_oracle.mdp.models import MDP
from usability_oracle.policy.models import Policy
from usability_oracle.taskspec.models import TaskSpec
from usability_oracle.fragility.models import (
    CliffLocation,
    FragilityResult,
    Interval,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: solve policy and estimate cost at given β
# ---------------------------------------------------------------------------

def _solve_and_cost(mdp: MDP, beta: float, n_trajectories: int = 100) -> float:
    """Compute the expected task cost for *mdp* at rationality *beta*.

    Solves the free-energy Bellman equation and estimates expected cost
    via Monte Carlo simulation.

    Parameters
    ----------
    mdp : MDP
    beta : float
    n_trajectories : int

    Returns
    -------
    float
        Mean trajectory cost.
    """
    from usability_oracle.comparison.paired import (
        _solve_softmax_policy,
        _sample_trajectory_costs,
    )

    policy = _solve_softmax_policy(mdp, beta)
    samples = _sample_trajectory_costs(mdp, policy, n_trajectories)
    return float(np.mean(samples))


# ---------------------------------------------------------------------------
# FragilityAnalyzer
# ---------------------------------------------------------------------------


class FragilityAnalyzer:
    """Analyzes the cognitive fragility of an MDP over a β range.

    Fragility measures how sensitive usability is to changes in the user's
    rationality level.  High fragility indicates that the interface is only
    usable for a narrow cognitive capacity band.

    Parameters
    ----------
    resolution : int
        Number of β values to sample in the sweep.
    n_trajectories : int
        Monte Carlo trajectories per β evaluation.
    population_betas : dict[str, float] or None
        Named β values for population percentiles.
        Default: 5th, 25th, 50th, 75th, 95th percentile β values.
    """

    DEFAULT_POPULATION_BETAS: dict[str, float] = {
        "p05_impaired": 0.3,
        "p25_novice": 1.0,
        "p50_average": 3.0,
        "p75_experienced": 7.0,
        "p95_expert": 15.0,
    }

    def __init__(
        self,
        resolution: int = 100,
        n_trajectories: int = 100,
        population_betas: Optional[dict[str, float]] = None,
    ) -> None:
        self.resolution = resolution
        self.n_trajectories = n_trajectories
        self.population_betas = population_betas or self.DEFAULT_POPULATION_BETAS

    def analyze(
        self,
        mdp: MDP,
        task: TaskSpec,
        beta_range: tuple[float, float] = (0.1, 20.0),
    ) -> FragilityResult:
        """Run a full fragility analysis.

        Parameters
        ----------
        mdp : MDP
            The MDP to analyze.
        task : TaskSpec
            Task specification (currently used for metadata).
        beta_range : tuple[float, float]
            ``(β_min, β_max)`` range to sweep.

        Returns
        -------
        FragilityResult
        """
        beta_lo, beta_hi = beta_range
        betas = np.linspace(beta_lo, beta_hi, self.resolution)

        logger.info(
            "Starting fragility analysis: β∈[%.2f, %.2f], resolution=%d",
            beta_lo, beta_hi, self.resolution,
        )

        # 1. Compute cost curve C(β)
        cost_curve = self._compute_cost_curve(mdp, betas)

        # 2. Compute fragility score
        fragility_score = self._compute_fragility_score(
            list(zip(betas.tolist(), cost_curve.tolist()))
        )

        # 3. Find discontinuities (cliff candidates)
        cliff_indices = self._find_discontinuities(cost_curve, betas)

        # 4. Build CliffLocation objects for detected cliffs
        from usability_oracle.fragility.cliff import CliffDetector
        detector = CliffDetector(n_trajectories=self.n_trajectories)
        cliffs = detector.detect(mdp, beta_range, self.resolution)

        # 5. Compute robustness interval
        robustness = self._robustness_interval(cost_curve, betas, threshold=0.1)

        # 6. Population impact analysis
        pop_impact = self._population_impact(mdp, self.population_betas)

        # 7. Beta sensitivity for each cost component
        beta_sensitivity: dict[str, float] = {}
        if len(cost_curve) > 1:
            grad = np.gradient(cost_curve, betas)
            beta_sensitivity["mean_gradient"] = float(np.mean(np.abs(grad)))
            beta_sensitivity["max_gradient"] = float(np.max(np.abs(grad)))
            beta_sensitivity["std_gradient"] = float(np.std(grad))

        return FragilityResult(
            fragility_score=fragility_score,
            cliff_locations=cliffs,
            beta_sensitivity=beta_sensitivity,
            robustness_interval=robustness,
            population_impact=pop_impact,
            cost_curve=list(zip(betas.tolist(), cost_curve.tolist())),
            metadata={
                "beta_range": list(beta_range),
                "resolution": self.resolution,
                "n_trajectories": self.n_trajectories,
                "task_id": task.spec_id,
            },
        )

    def _compute_fragility_score(
        self,
        cost_curve: list[tuple[float, float]],
    ) -> float:
        """Compute the fragility score from the cost-vs-β curve.

        The fragility score is the normalized maximum gradient:

        .. math::
            F = \\frac{\\max_i |C_{i+1} - C_i| / |\\beta_{i+1} - \\beta_i|}
                      {\\max(C) - \\min(C)} \\cdot \\Delta\\beta_{step}

        Clamped to [0, 1].

        Parameters
        ----------
        cost_curve : list[tuple[float, float]]
            ``(β, cost)`` pairs.

        Returns
        -------
        float
            Fragility score in [0, 1].
        """
        if len(cost_curve) < 2:
            return 0.0

        betas = np.array([p[0] for p in cost_curve])
        costs = np.array([p[1] for p in cost_curve])

        cost_range = float(np.max(costs) - np.min(costs))
        if cost_range < 1e-12:
            return 0.0

        # Compute finite differences dC/dβ
        d_beta = np.diff(betas)
        d_cost = np.diff(costs)

        # Avoid division by zero
        valid = d_beta > 1e-12
        gradients = np.zeros_like(d_cost)
        gradients[valid] = np.abs(d_cost[valid] / d_beta[valid])

        max_gradient = float(np.max(gradients))

        # Normalize by cost range
        beta_step = float(np.mean(d_beta))
        fragility = (max_gradient * beta_step) / cost_range

        return float(np.clip(fragility, 0.0, 1.0))

    def _compute_cost_curve(
        self,
        mdp: MDP,
        betas: np.ndarray,
    ) -> np.ndarray:
        """Compute the cost for each β value in the array.

        Parameters
        ----------
        mdp : MDP
        betas : np.ndarray
            Array of β values.

        Returns
        -------
        np.ndarray
            Array of costs, one per β.
        """
        costs = np.zeros(len(betas))
        for i, beta in enumerate(betas):
            try:
                costs[i] = _solve_and_cost(mdp, float(beta), self.n_trajectories)
            except Exception as e:
                logger.warning("Cost computation failed at β=%.3f: %s", beta, e)
                costs[i] = float("nan")

        # Interpolate NaN values
        nan_mask = np.isnan(costs)
        if nan_mask.any() and not nan_mask.all():
            valid_idx = np.where(~nan_mask)[0]
            interp_fn = sp_interp.interp1d(
                betas[valid_idx], costs[valid_idx],
                kind="linear", fill_value="extrapolate",
            )
            costs[nan_mask] = interp_fn(betas[nan_mask])

        return costs

    def _find_discontinuities(
        self,
        cost_curve: np.ndarray,
        betas: np.ndarray,
        threshold_factor: float = 3.0,
    ) -> list[int]:
        """Find indices of discontinuities in the cost curve.

        A discontinuity is defined as a point where the local gradient
        exceeds ``threshold_factor`` times the median gradient.

        Parameters
        ----------
        cost_curve : np.ndarray
        betas : np.ndarray
        threshold_factor : float
            Multiplier on median gradient for cliff detection.

        Returns
        -------
        list[int]
            Indices of detected discontinuities.
        """
        if len(cost_curve) < 3:
            return []

        gradient = np.gradient(cost_curve, betas)
        abs_grad = np.abs(gradient)
        median_grad = float(np.median(abs_grad))

        if median_grad < 1e-12:
            return []

        threshold = threshold_factor * median_grad
        indices = np.where(abs_grad > threshold)[0].tolist()

        logger.debug(
            "Found %d discontinuity candidates (threshold=%.4f)",
            len(indices), threshold,
        )
        return indices

    def _population_impact(
        self,
        mdp: MDP,
        population_betas: dict[str, float],
    ) -> dict[str, float]:
        """Compute cost impact at different population β percentiles.

        Parameters
        ----------
        mdp : MDP
        population_betas : dict[str, float]
            Named β values (e.g., ``{"novice": 1.0, "expert": 10.0}``).

        Returns
        -------
        dict[str, float]
            Cost at each population percentile.
        """
        impact: dict[str, float] = {}
        for name, beta in population_betas.items():
            try:
                cost = _solve_and_cost(mdp, beta, self.n_trajectories)
                impact[name] = cost
            except Exception as e:
                logger.warning("Population cost failed for %s (β=%.2f): %s", name, beta, e)
                impact[name] = float("nan")
        return impact

    def _robustness_interval(
        self,
        cost_curve: np.ndarray,
        betas: np.ndarray,
        threshold: float = 0.1,
    ) -> Interval:
        """Find the widest β interval where cost variation is below threshold.

        The robustness interval is the largest contiguous range of β
        values where:

        .. math::
            \\frac{|C(\\beta) - C(\\beta_{ref})|}{C(\\beta_{ref})} \\leq \\tau

        with :math:`\\beta_{ref}` being the median β in the analysis range
        and :math:`\\tau` being the threshold.

        Parameters
        ----------
        cost_curve : np.ndarray
        betas : np.ndarray
        threshold : float
            Maximum relative cost variation.

        Returns
        -------
        Interval
        """
        if len(cost_curve) < 2:
            return Interval(float(betas[0]), float(betas[-1]))

        # Reference cost at median β
        mid_idx = len(betas) // 2
        ref_cost = cost_curve[mid_idx]
        if abs(ref_cost) < 1e-12:
            return Interval(float(betas[0]), float(betas[-1]))

        relative_deviation = np.abs(cost_curve - ref_cost) / abs(ref_cost)
        within_threshold = relative_deviation <= threshold

        # Find the longest contiguous True run
        best_start = 0
        best_length = 0
        current_start = 0
        current_length = 0

        for i, ok in enumerate(within_threshold):
            if ok:
                if current_length == 0:
                    current_start = i
                current_length += 1
                if current_length > best_length:
                    best_length = current_length
                    best_start = current_start
            else:
                current_length = 0

        if best_length == 0:
            return Interval(float(betas[mid_idx]), float(betas[mid_idx]))

        return Interval(
            float(betas[best_start]),
            float(betas[min(best_start + best_length - 1, len(betas) - 1)]),
        )
