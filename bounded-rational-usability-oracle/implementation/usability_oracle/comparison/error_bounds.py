"""
usability_oracle.comparison.error_bounds — Formal error bound computation.

Provides :class:`ErrorBoundComputer` for computing rigorous error bounds
on usability cost estimates.  Three sources of error are considered:

1. **Abstraction error**: from bisimulation-based state-space reduction.
2. **Sampling error**: from Monte Carlo estimation of expected costs.
3. **Model error**: from cognitive model parameter uncertainty.

The total error bound is the sum of individual bounds, yielding a
guarantee of the form:

    |Ĉ - C*| ≤ ε_abstraction + ε_sampling + ε_model

with probability at least 1 − α.

References
----------
- Hoeffding, W. (1963). Probability inequalities for sums of bounded
  random variables. *J. American Statistical Association*, 58(301).
- Chebyshev, P. L. (1867). Des valeurs moyennes. *J. de mathématiques
  pures et appliquées*, 2(12), 177–184.
- Dean, T. & Givan, R. (1997). Model minimization in Markov decision
  processes. *AAAI*, 106–111.
- Givan, R., Dean, T., & Greig, M. (2003). Equivalence notions and model
  minimization in Markov decision processes. *Artificial Intelligence*, 147.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import stats as sp_stats

from usability_oracle.mdp.models import MDP
from usability_oracle.comparison.models import Partition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ErrorBoundResult
# ---------------------------------------------------------------------------


@dataclass
class ErrorBoundResult:
    """Detailed error bound breakdown.

    Attributes
    ----------
    abstraction_error : float
        Upper bound on error from state-space abstraction.
    sampling_error : float
        Upper bound on error from finite-sample estimation.
    model_error : float
        Upper bound on error from model parameter uncertainty.
    total_error : float
        Sum of all error components.
    confidence : float
        1 − α confidence level.
    required_samples : int
        Minimum samples needed for the target error.
    """

    abstraction_error: float = 0.0
    sampling_error: float = 0.0
    model_error: float = 0.0
    total_error: float = 0.0
    confidence: float = 0.95
    required_samples: int = 0


# ---------------------------------------------------------------------------
# ErrorBoundComputer
# ---------------------------------------------------------------------------


class ErrorBoundComputer:
    """Computes formal error bounds for usability cost estimates.

    Provides Hoeffding, Chebyshev, and CLT-based concentration
    inequalities for sampling error, plus abstraction error from
    bisimulation quotients.

    Parameters
    ----------
    bound_method : str
        Default concentration inequality: ``"hoeffding"``, ``"chebyshev"``,
        or ``"clt"``.
    cost_range : float
        Known upper bound on the range of individual cost values (used by
        Hoeffding).  If costs are in [0, R], set ``cost_range = R``.
    """

    def __init__(
        self,
        bound_method: str = "hoeffding",
        cost_range: float = 100.0,
    ) -> None:
        self.bound_method = bound_method
        self.cost_range = cost_range

    # ------------------------------------------------------------------
    # Abstraction error
    # ------------------------------------------------------------------

    def compute_abstraction_error(
        self,
        original: MDP,
        abstract: MDP,
        partition: Partition,
    ) -> float:
        """Compute the abstraction error bound from state-space reduction.

        The abstraction error is bounded by the maximum within-block cost
        variation.  For a partition :math:`\\mathcal{P}` of the state space:

        .. math::
            \\varepsilon_{abs} \\leq \\frac{\\gamma}{1 - \\gamma}
            \\max_{B \\in \\mathcal{P}} \\max_{s, s' \\in B}
            |c(s) - c(s')|

        where :math:`c(s)` is the one-step expected cost from state *s*
        under any policy, and :math:`\\gamma` is the discount factor.

        This follows from Theorem 3 of Givan, Dean & Greig (2003).

        Parameters
        ----------
        original : MDP
            Full (un-abstracted) MDP.
        abstract : MDP
            Abstracted MDP.
        partition : Partition
            The partition used for abstraction.

        Returns
        -------
        float
            Upper bound on the value-function error due to abstraction.
        """
        gamma = original.discount

        max_block_variation = 0.0
        for block in partition.blocks:
            if len(block.state_ids) <= 1:
                continue
            costs_in_block: list[float] = []
            for sid in block.state_ids:
                if sid not in original.states:
                    continue
                # Compute expected one-step cost from this state
                actions = original.get_actions(sid)
                if not actions:
                    costs_in_block.append(0.0)
                    continue
                min_cost = float("inf")
                for a in actions:
                    transitions = original.get_transitions(sid, a)
                    expected_cost = sum(p * c for _, p, c in transitions)
                    min_cost = min(min_cost, expected_cost)
                costs_in_block.append(min_cost)

            if len(costs_in_block) >= 2:
                variation = max(costs_in_block) - min(costs_in_block)
                max_block_variation = max(max_block_variation, variation)

        if gamma >= 1.0:
            # Undiscounted: bound grows linearly with horizon
            abstraction_bound = max_block_variation * original.n_states
        else:
            abstraction_bound = (gamma / (1 - gamma)) * max_block_variation

        logger.debug(
            "Abstraction error bound: %.6f (max block variation: %.6f)",
            abstraction_bound, max_block_variation,
        )
        return abstraction_bound

    # ------------------------------------------------------------------
    # Sampling error
    # ------------------------------------------------------------------

    def compute_sampling_error(
        self,
        n_trajectories: int,
        variance: float,
        alpha: float = 0.05,
    ) -> float:
        """Compute the sampling error bound.

        Dispatches to the appropriate concentration inequality based on
        ``self.bound_method``.

        Parameters
        ----------
        n_trajectories : int
            Number of Monte Carlo trajectories.
        variance : float
            Estimated variance of trajectory costs.
        alpha : float
            Failure probability (the bound holds with probability 1 − α).

        Returns
        -------
        float
            Upper bound on |Ĉ − E[C]|.
        """
        if n_trajectories <= 0:
            return float("inf")

        if self.bound_method == "hoeffding":
            return self._hoeffding_bound(n_trajectories, self.cost_range, alpha)
        elif self.bound_method == "chebyshev":
            return self._chebyshev_bound(n_trajectories, variance, alpha)
        elif self.bound_method == "clt":
            return self._clt_bound(n_trajectories, variance, alpha)
        else:
            raise ValueError(f"Unknown bound method: {self.bound_method!r}")

    # ------------------------------------------------------------------
    # Total error
    # ------------------------------------------------------------------

    def compute_total_error(
        self,
        abstraction_err: float,
        sampling_err: float,
        model_err: float = 0.0,
    ) -> float:
        """Compute the total error bound as the sum of components.

        .. math::
            \\varepsilon_{total} = \\varepsilon_{abs} + \\varepsilon_{samp}
            + \\varepsilon_{model}

        Parameters
        ----------
        abstraction_err : float
        sampling_err : float
        model_err : float

        Returns
        -------
        float
        """
        total = abstraction_err + sampling_err + model_err
        logger.debug(
            "Total error: %.6f = %.6f (abs) + %.6f (samp) + %.6f (model)",
            total, abstraction_err, sampling_err, model_err,
        )
        return total

    # ------------------------------------------------------------------
    # Required sample size
    # ------------------------------------------------------------------

    def compute_required_samples(
        self,
        target_error: float,
        variance: float,
        alpha: float = 0.05,
    ) -> int:
        """Compute the minimum number of samples for a target error bound.

        Inverts the concentration inequality to find the smallest *n*
        such that the sampling error ≤ ``target_error``.

        Parameters
        ----------
        target_error : float
            Desired upper bound on sampling error.
        variance : float
            Estimated variance of trajectory costs.
        alpha : float
            Failure probability.

        Returns
        -------
        int
            Minimum number of trajectories required.
        """
        if target_error <= 0:
            return int(1e9)

        if self.bound_method == "hoeffding":
            # ε = R * sqrt(log(2/α) / (2n))
            # n = R² log(2/α) / (2 ε²)
            n = (self.cost_range ** 2 * math.log(2.0 / alpha)) / (
                2.0 * target_error ** 2
            )
        elif self.bound_method == "chebyshev":
            # ε = sqrt(σ² / (n α))
            # n = σ² / (ε² α)
            n = variance / (target_error ** 2 * alpha)
        elif self.bound_method == "clt":
            # ε = z_{α/2} σ / sqrt(n)
            # n = (z σ / ε)²
            z = sp_stats.norm.ppf(1 - alpha / 2)
            sigma = math.sqrt(max(variance, 0.0))
            n = (z * sigma / target_error) ** 2
        else:
            raise ValueError(f"Unknown bound method: {self.bound_method!r}")

        return max(1, math.ceil(n))

    # ------------------------------------------------------------------
    # Concentration inequalities
    # ------------------------------------------------------------------

    @staticmethod
    def _hoeffding_bound(n: int, range_val: float, alpha: float) -> float:
        """Hoeffding's inequality bound.

        For i.i.d. random variables :math:`X_1, \\ldots, X_n` with
        :math:`X_i \\in [a_i, b_i]`, Hoeffding's inequality gives:

        .. math::
            P\\bigl(|\\bar{X} - E[\\bar{X}]| \\geq t\\bigr)
            \\leq 2 \\exp\\!\\left(-\\frac{2 n^2 t^2}
                                          {\\sum_i (b_i - a_i)^2}\\right)

        For identical ranges [0, R] and solving for t at confidence 1 − α:

        .. math::
            t = R \\sqrt{\\frac{\\ln(2/\\alpha)}{2n}}

        Parameters
        ----------
        n : int
            Number of samples.
        range_val : float
            Range of each random variable (max − min).
        alpha : float
            Failure probability.

        Returns
        -------
        float
            Hoeffding bound ε.
        """
        if n <= 0 or range_val <= 0:
            return float("inf")
        return range_val * math.sqrt(math.log(2.0 / alpha) / (2.0 * n))

    @staticmethod
    def _chebyshev_bound(n: int, variance: float, alpha: float) -> float:
        """Chebyshev's inequality bound.

        Chebyshev's inequality for the sample mean:

        .. math::
            P\\bigl(|\\bar{X} - \\mu| \\geq t\\bigr) \\leq \\frac{\\sigma^2}{n t^2}

        Solving for t at confidence 1 − α:

        .. math::
            t = \\sqrt{\\frac{\\sigma^2}{n \\alpha}}

        Parameters
        ----------
        n : int
        variance : float
            Population variance σ².
        alpha : float

        Returns
        -------
        float
            Chebyshev bound ε.
        """
        if n <= 0 or alpha <= 0:
            return float("inf")
        return math.sqrt(variance / (n * alpha))

    @staticmethod
    def _clt_bound(n: int, variance: float, alpha: float) -> float:
        """Central Limit Theorem–based bound.

        For large *n*, the sample mean is approximately normal:

        .. math::
            \\bar{X} \\sim \\mathcal{N}\\!\\left(\\mu,\\, \\frac{\\sigma^2}{n}\\right)

        The bound is:

        .. math::
            \\varepsilon = z_{\\alpha/2} \\cdot \\frac{\\sigma}{\\sqrt{n}}

        Parameters
        ----------
        n : int
        variance : float
        alpha : float

        Returns
        -------
        float
            CLT bound ε.
        """
        if n <= 0:
            return float("inf")
        z = sp_stats.norm.ppf(1 - alpha / 2)
        sigma = math.sqrt(max(variance, 0.0))
        return z * sigma / math.sqrt(n)

    # ------------------------------------------------------------------
    # Full error analysis
    # ------------------------------------------------------------------

    def full_analysis(
        self,
        original: MDP,
        abstract: MDP,
        partition: Partition,
        n_trajectories: int,
        cost_variance: float,
        model_error: float = 0.0,
        alpha: float = 0.05,
        target_error: Optional[float] = None,
    ) -> ErrorBoundResult:
        """Run a complete error analysis.

        Computes all three error components and optionally the required
        sample size for a target error.

        Parameters
        ----------
        original : MDP
        abstract : MDP
        partition : Partition
        n_trajectories : int
        cost_variance : float
        model_error : float
        alpha : float
        target_error : float, optional

        Returns
        -------
        ErrorBoundResult
        """
        abs_err = self.compute_abstraction_error(original, abstract, partition)
        samp_err = self.compute_sampling_error(n_trajectories, cost_variance, alpha)
        total = self.compute_total_error(abs_err, samp_err, model_error)

        req_samples = 0
        if target_error is not None:
            residual = max(target_error - abs_err - model_error, 1e-12)
            req_samples = self.compute_required_samples(
                residual, cost_variance, alpha
            )

        return ErrorBoundResult(
            abstraction_error=abs_err,
            sampling_error=samp_err,
            model_error=model_error,
            total_error=total,
            confidence=1.0 - alpha,
            required_samples=req_samples,
        )
