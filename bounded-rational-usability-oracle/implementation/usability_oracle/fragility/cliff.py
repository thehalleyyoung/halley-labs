"""
usability_oracle.fragility.cliff — Cliff (phase transition) detection.

Implements :class:`CliffDetector`, which locates *cliffs* in the
cost-vs-β curve — points where the bounded-rational policy undergoes
a phase transition, causing a sudden change in expected task cost.

Detection algorithm
-------------------
1. Compute the cost curve C(β) at a fine grid of β values.
2. Compute the gradient |dC/dβ| via central finite differences.
3. Find peaks in the gradient using ``scipy.signal.find_peaks``.
4. Refine each peak location to sub-grid precision via bisection.
5. Classify the cliff type (policy switch, state collapse, information cliff).

Cliff types
-----------
- **Policy switch**: the optimal action at some state changes.
- **State collapse**: multiple states become indistinguishable.
- **Information cliff**: the policy's information cost dominates.

References
----------
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs.
- Todorov, E. (2007). Linearly-solvable Markov decision processes. *NIPS*.
"""

from __future__ import annotations

import math
import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

from usability_oracle.mdp.models import MDP
from usability_oracle.policy.models import Policy
from usability_oracle.fragility.models import CliffLocation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: solve policy at β
# ---------------------------------------------------------------------------

def _solve_and_cost(mdp: MDP, beta: float, n_trajectories: int = 100) -> float:
    """Compute the expected task cost for *mdp* at rationality *beta*."""
    from usability_oracle.comparison.paired import (
        _solve_softmax_policy,
        _sample_trajectory_costs,
    )
    policy = _solve_softmax_policy(mdp, beta)
    samples = _sample_trajectory_costs(mdp, policy, n_trajectories)
    return float(np.mean(samples))


def _solve_policy(mdp: MDP, beta: float) -> Policy:
    """Compute the bounded-rational policy at *beta*."""
    from usability_oracle.comparison.paired import _solve_softmax_policy
    return _solve_softmax_policy(mdp, beta)


# ---------------------------------------------------------------------------
# CliffDetector
# ---------------------------------------------------------------------------


class CliffDetector:
    """Detects cognitive cliffs in the cost-vs-β landscape.

    A cliff is a narrow β region where the cost changes rapidly,
    indicating a phase transition in the user's optimal strategy.

    Parameters
    ----------
    n_trajectories : int
        Monte Carlo trajectories per β evaluation.
    peak_prominence : float
        Minimum prominence for peak detection (relative to gradient range).
    peak_width : int
        Minimum peak width in grid points.
    bisection_tol : float
        Tolerance for cliff refinement bisection.
    max_bisection_iters : int
        Maximum bisection iterations.
    """

    # Cliff type constants
    POLICY_SWITCH = "policy_switch"
    STATE_COLLAPSE = "state_collapse"
    INFORMATION_CLIFF = "information_cliff"

    def __init__(
        self,
        n_trajectories: int = 100,
        peak_prominence: float = 0.1,
        peak_width: int = 1,
        bisection_tol: float = 0.005,
        max_bisection_iters: int = 30,
    ) -> None:
        self.n_trajectories = n_trajectories
        self.peak_prominence = peak_prominence
        self.peak_width = peak_width
        self.bisection_tol = bisection_tol
        self.max_bisection_iters = max_bisection_iters

    def detect(
        self,
        mdp: MDP,
        beta_range: tuple[float, float],
        resolution: int = 100,
    ) -> list[CliffLocation]:
        """Detect cliffs in the cost-vs-β curve.

        Parameters
        ----------
        mdp : MDP
            The MDP to analyze.
        beta_range : tuple[float, float]
            ``(β_min, β_max)`` range.
        resolution : int
            Number of grid points.

        Returns
        -------
        list[CliffLocation]
            Detected cliff locations, sorted by severity (descending).
        """
        beta_lo, beta_hi = beta_range
        betas = np.linspace(beta_lo, beta_hi, resolution)

        logger.info(
            "Detecting cliffs: β∈[%.2f, %.2f], resolution=%d",
            beta_lo, beta_hi, resolution,
        )

        # 1. Compute cost curve
        costs = np.zeros(resolution)
        for i, beta in enumerate(betas):
            try:
                costs[i] = _solve_and_cost(mdp, float(beta), self.n_trajectories)
            except Exception:
                costs[i] = float("nan")

        # Interpolate NaN values
        nan_mask = np.isnan(costs)
        if nan_mask.any() and not nan_mask.all():
            valid = np.where(~nan_mask)[0]
            from scipy.interpolate import interp1d
            f = interp1d(betas[valid], costs[valid], fill_value="extrapolate")
            costs[nan_mask] = f(betas[nan_mask])

        # 2. Compute gradient
        gradient = self._compute_gradient(costs, betas)

        # 3. Find peaks in |gradient|
        peak_indices = self._find_peaks(gradient, threshold=None)

        # 4. Refine and classify each cliff
        cliffs: list[CliffLocation] = []
        for idx in peak_indices:
            beta_approx = float(betas[idx])
            window = float(betas[1] - betas[0]) * 3

            # Refine cliff location
            cliff = self._refine_cliff(mdp, beta_approx, window)

            # Classify and score
            cliff.cliff_type = self._classify_cliff(cliff, mdp)
            cliff.severity = self._cliff_severity(cliff)
            cliff.gradient = float(np.abs(gradient[idx]))

            cliffs.append(cliff)

        # Sort by severity descending
        cliffs.sort(key=lambda c: c.severity, reverse=True)

        logger.info("Detected %d cliffs", len(cliffs))
        return cliffs

    def _compute_gradient(
        self,
        cost_curve: np.ndarray,
        betas: np.ndarray,
    ) -> np.ndarray:
        """Compute the gradient |dC/dβ| via central finite differences.

        Uses ``numpy.gradient`` which computes second-order central
        differences in the interior and first-order one-sided differences
        at the boundaries.

        Parameters
        ----------
        cost_curve : np.ndarray
        betas : np.ndarray

        Returns
        -------
        np.ndarray
            Absolute gradient at each β.
        """
        raw_gradient = np.gradient(cost_curve, betas)
        return np.abs(raw_gradient)

    def _find_peaks(
        self,
        gradient: np.ndarray,
        threshold: Optional[float] = None,
    ) -> list[int]:
        """Find peaks in the absolute gradient using scipy.

        Parameters
        ----------
        gradient : np.ndarray
            Absolute gradient values.
        threshold : float, optional
            Minimum peak height.  Defaults to ``peak_prominence`` × range.

        Returns
        -------
        list[int]
            Indices of detected peaks.
        """
        grad_range = float(np.max(gradient) - np.min(gradient))
        if grad_range < 1e-12:
            return []

        prominence = self.peak_prominence * grad_range
        if threshold is None:
            threshold = float(np.median(gradient)) + prominence

        peaks, properties = sp_signal.find_peaks(
            gradient,
            height=threshold,
            prominence=prominence,
            width=self.peak_width,
        )

        return peaks.tolist()

    def _refine_cliff(
        self,
        mdp: MDP,
        beta_approx: float,
        window: float,
    ) -> CliffLocation:
        """Refine cliff location to sub-grid precision via bisection.

        Searches for the β value where the cost change rate is maximal
        within the window ``[β_approx - window/2, β_approx + window/2]``.

        Parameters
        ----------
        mdp : MDP
        beta_approx : float
            Approximate cliff β from the grid.
        window : float
            Search window width.

        Returns
        -------
        CliffLocation
            Refined cliff with exact β*, cost_before, cost_after.
        """
        lo = max(beta_approx - window / 2, 0.01)
        hi = beta_approx + window / 2

        # Evaluate costs at boundaries for the initial cliff characterization
        cost_lo = _solve_and_cost(mdp, lo, self.n_trajectories)
        cost_hi = _solve_and_cost(mdp, hi, self.n_trajectories)

        # Bisection to find the steepest point
        best_beta = beta_approx
        best_gradient = 0.0

        for _ in range(self.max_bisection_iters):
            if hi - lo < self.bisection_tol:
                break

            mid = (lo + hi) / 2.0
            q1 = (lo + mid) / 2.0
            q3 = (mid + hi) / 2.0

            cost_q1 = _solve_and_cost(mdp, q1, self.n_trajectories)
            cost_mid = _solve_and_cost(mdp, mid, self.n_trajectories)
            cost_q3 = _solve_and_cost(mdp, q3, self.n_trajectories)

            # Gradient estimates at q1, mid, q3
            grad_lo = abs(cost_mid - cost_q1) / max(mid - q1, 1e-12)
            grad_hi = abs(cost_q3 - cost_mid) / max(q3 - mid, 1e-12)

            if grad_lo > grad_hi:
                hi = mid
                cost_hi = cost_mid
                if grad_lo > best_gradient:
                    best_gradient = grad_lo
                    best_beta = q1
            else:
                lo = mid
                cost_lo = cost_mid
                if grad_hi > best_gradient:
                    best_gradient = grad_hi
                    best_beta = q3

        # Final cost evaluation near the cliff
        eps = self.bisection_tol
        cost_before = _solve_and_cost(
            mdp, max(best_beta - eps, 0.01), self.n_trajectories
        )
        cost_after = _solve_and_cost(mdp, best_beta + eps, self.n_trajectories)

        # Identify affected states
        affected = self._find_affected_states(mdp, best_beta, eps)

        return CliffLocation(
            beta_star=best_beta,
            cost_before=cost_before,
            cost_after=cost_after,
            affected_states=affected,
            gradient=best_gradient,
        )

    def _find_affected_states(
        self,
        mdp: MDP,
        beta_star: float,
        epsilon: float,
    ) -> list[str]:
        """Find states where the policy changes across the cliff.

        Compares the greedy action at β* − ε and β* + ε for each state.

        Parameters
        ----------
        mdp : MDP
        beta_star : float
        epsilon : float

        Returns
        -------
        list[str]
            State IDs where the greedy action differs.
        """
        beta_lo = max(beta_star - epsilon, 0.01)
        beta_hi = beta_star + epsilon

        policy_lo = _solve_policy(mdp, beta_lo)
        policy_hi = _solve_policy(mdp, beta_hi)

        affected: list[str] = []
        for sid in mdp.states:
            probs_lo = policy_lo.action_probs(sid)
            probs_hi = policy_hi.action_probs(sid)
            if not probs_lo or not probs_hi:
                continue
            greedy_lo = max(probs_lo, key=probs_lo.get)
            greedy_hi = max(probs_hi, key=probs_hi.get)
            if greedy_lo != greedy_hi:
                affected.append(sid)

        return affected

    def _classify_cliff(
        self,
        location: CliffLocation,
        mdp: MDP,
    ) -> str:
        """Classify the cliff type based on affected states and cost pattern.

        - **policy_switch**: the greedy action changes at some states.
        - **state_collapse**: many states become indistinguishable
          (entropy drops sharply).
        - **information_cliff**: the KL divergence from prior changes
          sharply.

        Parameters
        ----------
        location : CliffLocation
        mdp : MDP

        Returns
        -------
        str
            Cliff type classification.
        """
        n_affected = len(location.affected_states)
        n_states = mdp.n_states

        if n_affected == 0:
            return self.INFORMATION_CLIFF

        # If many states change policy → likely information cliff
        fraction_affected = n_affected / max(n_states, 1)
        if fraction_affected > 0.5:
            return self.INFORMATION_CLIFF

        # Check entropy change for affected states
        eps = self.bisection_tol
        policy_lo = _solve_policy(mdp, max(location.beta_star - eps, 0.01))
        policy_hi = _solve_policy(mdp, location.beta_star + eps)

        entropy_drops = 0
        for sid in location.affected_states:
            h_lo = policy_lo.entropy(sid)
            h_hi = policy_hi.entropy(sid)
            if h_hi < h_lo * 0.5:
                entropy_drops += 1

        if entropy_drops > n_affected * 0.5:
            return self.STATE_COLLAPSE

        return self.POLICY_SWITCH

    @staticmethod
    def _cliff_severity(location: CliffLocation) -> float:
        """Compute a normalized severity score for a cliff.

        Combines the absolute cost jump with the gradient steepness:

        .. math::
            S = \\tanh(|\\Delta C| \\cdot |dC/d\\beta|)

        Bounded in [0, 1] where 0 = negligible and 1 = severe.

        Parameters
        ----------
        location : CliffLocation

        Returns
        -------
        float
        """
        jump = abs(location.cost_after - location.cost_before)
        gradient = abs(location.gradient)

        raw_severity = jump * (1.0 + gradient)
        return float(np.tanh(raw_severity))
