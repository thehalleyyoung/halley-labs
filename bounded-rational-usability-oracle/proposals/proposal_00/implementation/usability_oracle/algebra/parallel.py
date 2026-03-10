"""
usability_oracle.algebra.parallel — Parallel composition operator ⊗.

Implements the composition of two cognitive cost elements that occur
*concurrently*: the user performs task A and task B at the same time
(e.g., reading while listening, or visual search while maintaining a
working-memory item).

Theoretical Basis
-----------------
The parallel composition model is grounded in Wickens' **Multiple Resource
Theory** (MRT) [Wickens 2002, 2008]:

    *"Humans possess multiple pools of attentional resources defined along
    dichotomous dimensions: processing stages (perceptual/cognitive vs.
    response), processing codes (spatial vs. verbal), and perceptual
    modalities (visual vs. auditory)."*

When two tasks draw from the *same* resource pool, interference is high;
when they draw from *different* pools, tasks proceed nearly independently.

Mathematical Model
------------------
Given cost elements ``a = (μ_a, σ²_a, κ_a, λ_a)`` and
``b = (μ_b, σ²_b, κ_b, λ_b)`` with interference ``η ∈ [0, 1]``:

.. math::

    μ_{a⊗b}  &= \\max(μ_a, μ_b) + η · \\min(μ_a, μ_b)
    σ²_{a⊗b} &= \\max(σ²_a, σ²_b) + η² · \\min(σ²_a, σ²_b)
    κ_{a⊗b}  &= κ_{\\arg\\max(σ²)}   \\text{(dominated by higher-variance channel)}
    λ_{a⊗b}  &= λ_a + λ_b + η · λ_a · λ_b

Interference ``η = 0`` is perfect time-sharing; ``η = 1`` collapses to
fully sequential execution.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from usability_oracle.algebra.models import CostElement


# ---------------------------------------------------------------------------
# MRT interference matrix
# ---------------------------------------------------------------------------

# Channel codes (Wickens 2002 taxonomy)
CHANNELS = [
    "visual_focal",
    "visual_ambient",
    "auditory_verbal",
    "auditory_spatial",
    "cognitive_spatial",
    "cognitive_verbal",
    "response_manual",
    "response_vocal",
]

# Interference factors between channel pairs.
# 1.0 = complete interference (same resource pool)
# 0.0 = no interference (orthogonal pools)
# Values based on Wickens' MRT dimensional overlap heuristics.
INTERFERENCE_MATRIX: Dict[Tuple[str, str], float] = {}

def _build_interference_matrix() -> None:
    """Populate the global interference matrix."""
    # Same channel → full interference
    for ch in CHANNELS:
        INTERFERENCE_MATRIX[(ch, ch)] = 1.0

    # Same modality, different sub-type → moderate interference
    _set_symmetric("visual_focal", "visual_ambient", 0.6)

    _set_symmetric("auditory_verbal", "auditory_spatial", 0.5)

    # Same processing code, different stage → moderate interference
    _set_symmetric("cognitive_spatial", "visual_focal", 0.5)
    _set_symmetric("cognitive_spatial", "visual_ambient", 0.4)
    _set_symmetric("cognitive_verbal", "auditory_verbal", 0.5)
    _set_symmetric("cognitive_verbal", "auditory_spatial", 0.3)

    # Cross-modal → low interference
    _set_symmetric("visual_focal", "auditory_verbal", 0.2)
    _set_symmetric("visual_focal", "auditory_spatial", 0.15)
    _set_symmetric("visual_ambient", "auditory_verbal", 0.15)
    _set_symmetric("visual_ambient", "auditory_spatial", 0.1)

    # Response channels
    _set_symmetric("response_manual", "response_vocal", 0.3)
    _set_symmetric("response_manual", "cognitive_spatial", 0.4)
    _set_symmetric("response_manual", "cognitive_verbal", 0.3)
    _set_symmetric("response_vocal", "cognitive_verbal", 0.5)
    _set_symmetric("response_vocal", "cognitive_spatial", 0.2)

    # Response vs. perceptual
    _set_symmetric("response_manual", "visual_focal", 0.35)
    _set_symmetric("response_manual", "visual_ambient", 0.2)
    _set_symmetric("response_manual", "auditory_verbal", 0.15)
    _set_symmetric("response_manual", "auditory_spatial", 0.1)
    _set_symmetric("response_vocal", "visual_focal", 0.2)
    _set_symmetric("response_vocal", "visual_ambient", 0.1)
    _set_symmetric("response_vocal", "auditory_verbal", 0.4)
    _set_symmetric("response_vocal", "auditory_spatial", 0.2)

    # Cognitive cross-code → moderate
    _set_symmetric("cognitive_spatial", "cognitive_verbal", 0.4)

    # Fill remaining pairs with a default low interference
    for i, ch_a in enumerate(CHANNELS):
        for ch_b in CHANNELS[i:]:
            if (ch_a, ch_b) not in INTERFERENCE_MATRIX:
                INTERFERENCE_MATRIX[(ch_a, ch_b)] = 0.1
                INTERFERENCE_MATRIX[(ch_b, ch_a)] = 0.1


def _set_symmetric(a: str, b: str, value: float) -> None:
    INTERFERENCE_MATRIX[(a, b)] = value
    INTERFERENCE_MATRIX[(b, a)] = value


_build_interference_matrix()


# ---------------------------------------------------------------------------
# ParallelComposer
# ---------------------------------------------------------------------------


class ParallelComposer:
    r"""Compose cost elements in parallel (⊗ operator).

    Based on Wickens' Multiple Resource Theory: concurrent tasks draw on
    shared or independent cognitive resource pools, producing interference
    that increases the effective cost.

    Usage::

        composer = ParallelComposer()
        c = composer.compose(a, b, interference=0.3)

        # MRT-based interference lookup:
        eta = ParallelComposer.interference_factor("visual_focal", "auditory_verbal")
        c = composer.compose(a, b, interference=eta)
    """

    # -- single composition --------------------------------------------------

    def compose(
        self,
        a: CostElement,
        b: CostElement,
        interference: float = 0.0,
    ) -> CostElement:
        r"""Compose two cost elements in parallel.

        Parameters
        ----------
        a, b : CostElement
            The two concurrent cost elements.
        interference : float
            Interference factor ``η ∈ [0, 1]``.  0 = perfect time-sharing,
            1 = fully serial execution.

        Returns
        -------
        CostElement
            The composed cost element ``a ⊗ b``.

        Mathematical formulation
        ------------------------
        .. math::

            μ_{a⊗b}  = \max(μ_a, μ_b) + η · \min(μ_a, μ_b)

        Rationale: with perfect time-sharing (η=0) the total time equals
        the slower task; with full interference (η=1) the total time is
        the sum (i.e. effectively serial).

        .. math::

            σ²_{a⊗b} = \max(σ²_a, σ²_b) + η² · \min(σ²_a, σ²_b)

        Variance of the parallel cost is dominated by the higher-variance
        channel, with a second-order contribution from the other channel
        scaled by ``η²``.

        .. math::

            κ_{a⊗b} = κ_{\arg\max(σ²)}

        The skewness is dominated by the higher-variance channel because
        the max operator in the mean essentially selects that distribution.

        .. math::

            λ_{a⊗b} = λ_a + λ_b + η · λ_a · λ_b

        Tail risk is *superadditive*: the probability of catastrophic failure
        when doing two things at once is greater than the sum of individual
        risks, modulated by interference.

        Soundness
        ---------
        * ``μ_{a⊗b} ≥ max(μ_a, μ_b)`` ✓ (monotonicity)
        * ``a ⊗ 0 = a`` ✓ (identity: η·min(μ_a, 0) = 0)
        * ``a ⊗ b = b ⊗ a`` ✓ (commutativity)
        """
        self._validate_interference(interference)

        mu_max = max(a.mu, b.mu)
        mu_min = min(a.mu, b.mu)
        mu = mu_max + interference * mu_min

        var_max = max(a.sigma_sq, b.sigma_sq)
        var_min = min(a.sigma_sq, b.sigma_sq)
        sigma_sq = var_max + interference * interference * var_min

        # Skewness: dominated by the higher-variance channel
        if a.sigma_sq >= b.sigma_sq:
            kappa = a.kappa
        else:
            kappa = b.kappa

        # Tail risk: superadditive
        lambda_ = a.lambda_ + b.lambda_ + interference * a.lambda_ * b.lambda_
        lambda_ = min(lambda_, 1.0)

        return CostElement(mu=mu, sigma_sq=sigma_sq, kappa=kappa, lambda_=lambda_)

    # -- n-ary parallel composition ------------------------------------------

    def compose_group(
        self,
        elements: List[CostElement],
        interference: float = 0.0,
    ) -> CostElement:
        """Compose a group of elements in parallel (associative fold).

        Parameters
        ----------
        elements : list[CostElement]
            The cost elements to compose concurrently.
        interference : float
            Uniform interference factor between all pairs.

        Returns
        -------
        CostElement
        """
        if not elements:
            return CostElement.zero()
        result = elements[0]
        for elem in elements[1:]:
            result = self.compose(result, elem, interference=interference)
        return result

    def compose_with_channels(
        self,
        elements: List[CostElement],
        channels: List[str],
    ) -> CostElement:
        """Compose elements with MRT-based interference lookup.

        Parameters
        ----------
        elements : list[CostElement]
            Cost elements for concurrent tasks.
        channels : list[str]
            MRT channel labels, one per element.

        Returns
        -------
        CostElement
        """
        if len(elements) != len(channels):
            raise ValueError(
                f"Number of elements ({len(elements)}) must match "
                f"number of channels ({len(channels)})."
            )
        if not elements:
            return CostElement.zero()

        result = elements[0]
        for i in range(1, len(elements)):
            # Use max pairwise interference between the accumulated channels
            # and the new channel.
            eta = max(
                self.interference_factor(channels[j], channels[i])
                for j in range(i)
            )
            result = self.compose(result, elements[i], interference=eta)
        return result

    # -- MRT interference lookup ---------------------------------------------

    @staticmethod
    def interference_factor(channel_a: str, channel_b: str) -> float:
        """Look up the MRT-based interference factor between two channels.

        Parameters
        ----------
        channel_a, channel_b : str
            Channel identifiers from the Wickens taxonomy.
            Valid values: ``visual_focal``, ``visual_ambient``,
            ``auditory_verbal``, ``auditory_spatial``,
            ``cognitive_spatial``, ``cognitive_verbal``,
            ``response_manual``, ``response_vocal``.

        Returns
        -------
        float
            Interference factor ``η ∈ [0, 1]``.

        Raises
        ------
        KeyError
            If either channel is not recognised.
        """
        key = (channel_a, channel_b)
        if key in INTERFERENCE_MATRIX:
            return INTERFERENCE_MATRIX[key]
        # Try reverse order
        key_r = (channel_b, channel_a)
        if key_r in INTERFERENCE_MATRIX:
            return INTERFERENCE_MATRIX[key_r]
        raise KeyError(
            f"Unknown channel pair ({channel_a!r}, {channel_b!r}). "
            f"Valid channels: {CHANNELS}"
        )

    # -- capacity analysis ---------------------------------------------------

    @staticmethod
    def capacity_coefficient(
        rt_a: float, rt_b: float, rt_ab: float
    ) -> float:
        """Compute the capacity coefficient C(t).

        .. math::

            C(t) = \\frac{\\log(S_{AB}(t))}{\\log(S_A(t)) + \\log(S_B(t))}

        where ``S(t) = 1 - F(t)`` is the survivor function evaluated at the
        observed reaction time.

        Simplified version using mean RTs:
        * ``C > 1`` → super-capacity (parallel speedup)
        * ``C = 1`` → unlimited capacity
        * ``C < 1`` → limited capacity (interference)

        Parameters
        ----------
        rt_a : float
            Mean reaction time for task A alone.
        rt_b : float
            Mean reaction time for task B alone.
        rt_ab : float
            Mean reaction time for tasks A and B combined.

        Returns
        -------
        float
            Capacity coefficient estimate.
        """
        # Use a simple exponential survivor model: S(t) = exp(-t/μ)
        # log(S(t)) = -t/μ
        # C(t) = (-t/μ_AB) / (-t/μ_A + -t/μ_B) = (1/μ_AB) / (1/μ_A + 1/μ_B)
        if rt_a <= 0 or rt_b <= 0 or rt_ab <= 0:
            return 1.0
        return (1.0 / rt_ab) / (1.0 / rt_a + 1.0 / rt_b)

    # -- interference estimation from data -----------------------------------

    @staticmethod
    def estimate_interference(
        cost_a: CostElement,
        cost_b: CostElement,
        cost_ab: CostElement,
    ) -> float:
        """Estimate the interference factor from observed single and dual
        task costs.

        Inverts the composition formula:

        .. math::

            η = \\frac{μ_{AB} - \\max(μ_A, μ_B)}{\\min(μ_A, μ_B)}

        Returns
        -------
        float
            Estimated interference, clamped to ``[0, 1]``.
        """
        mu_max = max(cost_a.mu, cost_b.mu)
        mu_min = min(cost_a.mu, cost_b.mu)
        if mu_min < 1e-12:
            return 0.0
        eta = (cost_ab.mu - mu_max) / mu_min
        return max(0.0, min(1.0, eta))

    # -- validation ----------------------------------------------------------

    @staticmethod
    def _validate_interference(interference: float) -> None:
        if not (0.0 <= interference <= 1.0):
            raise ValueError(
                f"Interference must be in [0, 1], got {interference}."
            )
