"""
usability_oracle.information_theory.protocols — Information-theoretic primitives.

Structural interfaces for channel capacity computation, rate-distortion
solving, and general information measures used in the bounded-rationality
free-energy formulation.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, Tuple, runtime_checkable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from usability_oracle.information_theory.types import (
        Channel,
        ChannelCapacity,
        InformationState,
        MutualInfoResult,
        RateDistortionPoint,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ChannelComputer — compute channel capacity
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ChannelComputer(Protocol):
    """Compute the capacity of a discrete memoryless channel.

    Implementations should use the Blahut–Arimoto algorithm or similar
    iterative methods.
    """

    def compute_capacity(
        self,
        channel: Channel,
        *,
        tolerance: float = 1e-12,
        max_iterations: int = 1000,
    ) -> ChannelCapacity:
        """Compute channel capacity C = max_{p(x)} I(X; Y).

        Parameters
        ----------
        channel : Channel
            Discrete memoryless channel specification.
        tolerance : float
            Convergence tolerance for the iterative algorithm.
        max_iterations : int
            Maximum number of iterations.

        Returns
        -------
        ChannelCapacity
            Channel capacity with optimal input distribution.
        """
        ...

    def mutual_information(
        self,
        channel: Channel,
        input_distribution: Sequence[float],
    ) -> float:
        """Compute I(X; Y) for a given input distribution.

        Parameters
        ----------
        channel : Channel
            Channel specification.
        input_distribution : Sequence[float]
            Input distribution p(x).

        Returns
        -------
        float
            Mutual information in bits.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# RateDistortionSolver — compute the rate-distortion function
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class RateDistortionSolver(Protocol):
    """Compute points on the rate-distortion curve R(D).

    Central to bounded rationality: the agent's policy lies on the
    R(D) curve, with the rationality parameter β controlling the
    operating point.
    """

    def compute_rd_point(
        self,
        source_distribution: Sequence[float],
        distortion_matrix: Sequence[Sequence[float]],
        beta: float,
        *,
        tolerance: float = 1e-12,
        max_iterations: int = 1000,
    ) -> RateDistortionPoint:
        """Compute a single R(D) point for a given β.

        Uses the Blahut–Arimoto algorithm for rate-distortion.

        Parameters
        ----------
        source_distribution : Sequence[float]
            Source distribution p(x).
        distortion_matrix : Sequence[Sequence[float]]
            Distortion d(x, x̂) for each (source, reproduction) pair.
        beta : float
            Inverse temperature / Lagrange multiplier.
        tolerance : float
            Convergence tolerance.
        max_iterations : int
            Maximum iterations.

        Returns
        -------
        RateDistortionPoint
            Point on the R(D) curve.
        """
        ...

    def compute_rd_curve(
        self,
        source_distribution: Sequence[float],
        distortion_matrix: Sequence[Sequence[float]],
        beta_range: Tuple[float, float],
        n_points: int = 50,
    ) -> Sequence[RateDistortionPoint]:
        """Compute the full R(D) curve over a range of β values.

        Parameters
        ----------
        source_distribution : Sequence[float]
            Source distribution p(x).
        distortion_matrix : Sequence[Sequence[float]]
            Distortion matrix.
        beta_range : Tuple[float, float]
            (β_min, β_max) range.
        n_points : int
            Number of points to compute.

        Returns
        -------
        Sequence[RateDistortionPoint]
            Points on the R(D) curve, ordered by increasing β.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# InformationMeasure — general information-theoretic computations
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class InformationMeasure(Protocol):
    """General-purpose information-theoretic measure computations.

    Provides entropy, KL divergence, mutual information, and
    related quantities used throughout the oracle.
    """

    def entropy(self, distribution: Sequence[float]) -> float:
        """Shannon entropy H(X) in bits.

        Parameters
        ----------
        distribution : Sequence[float]
            Probability distribution (must sum to 1).

        Returns
        -------
        float
            Entropy in bits.
        """
        ...

    def kl_divergence(
        self,
        p: Sequence[float],
        q: Sequence[float],
    ) -> float:
        """KL divergence D_KL(p ‖ q) in bits.

        Parameters
        ----------
        p : Sequence[float]
            "True" distribution.
        q : Sequence[float]
            "Approximate" distribution.

        Returns
        -------
        float
            KL divergence (non-negative, potentially infinite).
        """
        ...

    def mutual_information(
        self,
        joint: Sequence[Sequence[float]],
    ) -> MutualInfoResult:
        """Compute mutual information from a joint distribution p(x, y).

        Parameters
        ----------
        joint : Sequence[Sequence[float]]
            Joint probability table (rows=X, columns=Y).

        Returns
        -------
        MutualInfoResult
            Full mutual information decomposition.
        """
        ...

    def free_energy(
        self,
        policy: Sequence[float],
        prior: Sequence[float],
        rewards: Sequence[float],
        beta: float,
    ) -> float:
        """Compute free energy F(π) = E_π[R] - (1/β) D_KL(π ‖ p₀).

        Parameters
        ----------
        policy : Sequence[float]
            Agent policy distribution π.
        prior : Sequence[float]
            Prior distribution p₀.
        rewards : Sequence[float]
            Reward for each action.
        beta : float
            Rationality parameter (inverse temperature).

        Returns
        -------
        float
            Free energy value.
        """
        ...


__all__ = [
    "ChannelComputer",
    "InformationMeasure",
    "RateDistortionSolver",
]
