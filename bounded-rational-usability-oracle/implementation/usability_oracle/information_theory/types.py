"""
usability_oracle.information_theory.types — Deep information-theoretic primitives.

Value types for channel capacity computation, rate-distortion theory, and
mutual information analysis used in the free-energy formulation
F(π) = E_π[R] - (1/β) D_KL(π ‖ p₀).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.core.types import Interval


# ═══════════════════════════════════════════════════════════════════════════
# Channel — discrete memoryless channel specification
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Channel:
    """Discrete memoryless channel p(y|x).

    The transition matrix has shape (|X|, |Y|) where rows are input symbols
    and columns are output symbols.  Each row sums to 1.

    Attributes
    ----------
    name : str
        Human-readable channel identifier.
    input_alphabet_size : int
        Number of input symbols |X|.
    output_alphabet_size : int
        Number of output symbols |Y|.
    transition_matrix : tuple[tuple[float, ...], ...]
        Row-stochastic transition matrix p(y|x).
    input_labels : tuple[str, ...]
        Optional labels for input symbols.
    output_labels : tuple[str, ...]
        Optional labels for output symbols.
    """

    name: str
    input_alphabet_size: int
    output_alphabet_size: int
    transition_matrix: Tuple[Tuple[float, ...], ...]
    input_labels: Tuple[str, ...] = ()
    output_labels: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.transition_matrix) != self.input_alphabet_size:
            raise ValueError(
                f"Transition matrix has {len(self.transition_matrix)} rows, "
                f"expected {self.input_alphabet_size}"
            )
        for i, row in enumerate(self.transition_matrix):
            if len(row) != self.output_alphabet_size:
                raise ValueError(
                    f"Row {i} has {len(row)} columns, expected {self.output_alphabet_size}"
                )

    def to_numpy(self) -> np.ndarray:
        """Return the transition matrix as a numpy array."""
        return np.array(self.transition_matrix)

    @classmethod
    def from_numpy(cls, name: str, matrix: np.ndarray) -> Channel:
        """Construct from a numpy transition matrix."""
        rows = tuple(tuple(float(x) for x in row) for row in matrix)
        return cls(
            name=name,
            input_alphabet_size=matrix.shape[0],
            output_alphabet_size=matrix.shape[1],
            transition_matrix=rows,
        )


# ═══════════════════════════════════════════════════════════════════════════
# ChannelCapacity — result of capacity computation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ChannelCapacity:
    """Result of computing channel capacity via Blahut-Arimoto or similar.

    Attributes
    ----------
    capacity_bits : float
        Channel capacity C in bits per channel use.
    optimal_input_distribution : tuple[float, ...]
        The input distribution p*(x) that achieves capacity.
    iterations : int
        Number of iterations used by the algorithm.
    converged : bool
        Whether the algorithm converged within tolerance.
    tolerance : float
        Convergence tolerance used.
    lower_bound : float
        Lower bound on capacity (from dual).
    upper_bound : float
        Upper bound on capacity (from primal).
    """

    capacity_bits: float
    optimal_input_distribution: Tuple[float, ...] = ()
    iterations: int = 0
    converged: bool = True
    tolerance: float = 1e-12
    lower_bound: float = 0.0
    upper_bound: float = float("inf")

    @property
    def capacity_nats(self) -> float:
        """Capacity in nats (multiply bits by ln 2)."""
        return self.capacity_bits * 0.6931471805599453

    @property
    def gap(self) -> float:
        """Gap between upper and lower bounds."""
        return self.upper_bound - self.lower_bound


# ═══════════════════════════════════════════════════════════════════════════
# RateDistortionPoint — point on the R(D) curve
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class RateDistortionPoint:
    """A single point on the rate-distortion curve R(D).

    Central to bounded-rational decision-making: the free-energy formulation
    trades off expected distortion (task cost) against information rate
    (KL divergence from the prior policy).

    Attributes
    ----------
    distortion : float
        Expected distortion D = E[d(x, x̂)] at this operating point.
    rate_bits : float
        Minimum rate R(D) in bits to achieve this distortion.
    beta : float
        Lagrange multiplier (inverse temperature) β = -dR/dD.
        Higher β ⟹ more "rational" (lower distortion, higher rate).
    optimal_encoding : tuple[tuple[float, ...], ...]
        The optimal test channel q*(x̂|x) achieving this point.
    """

    distortion: float
    rate_bits: float
    beta: float = 0.0
    optimal_encoding: Tuple[Tuple[float, ...], ...] = ()

    @property
    def rate_nats(self) -> float:
        return self.rate_bits * 0.6931471805599453

    @property
    def free_energy(self) -> float:
        """Free energy F = D + (1/β) R  (convention: minimize F)."""
        if self.beta <= 0:
            return float("inf")
        return self.distortion + self.rate_bits / self.beta


# ═══════════════════════════════════════════════════════════════════════════
# InformationState — information-theoretic state of a cognitive agent
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class InformationState:
    """Snapshot of the information state of a bounded-rational agent.

    Tracks the agent's belief distribution, accumulated information cost,
    and free-energy along a task trajectory.

    Attributes
    ----------
    belief : tuple[float, ...]
        Current belief distribution over world states.
    accumulated_cost_bits : float
        Total information cost incurred so far (in bits).
    free_energy : float
        Current free energy F(π) = E_π[R] - (1/β) D_KL(π ‖ p₀).
    beta : float
        Rationality parameter (inverse temperature).
    entropy_bits : float
        Shannon entropy H(belief) of the current belief.
    step : int
        Time step in the task trajectory.
    """

    belief: Tuple[float, ...] = ()
    accumulated_cost_bits: float = 0.0
    free_energy: float = 0.0
    beta: float = 1.0
    entropy_bits: float = 0.0
    step: int = 0

    @property
    def belief_size(self) -> int:
        return len(self.belief)

    @property
    def max_belief(self) -> float:
        """Maximum probability in the belief distribution."""
        return max(self.belief) if self.belief else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# MutualInfoResult — mutual information computation output
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class MutualInfoResult:
    """Result of computing mutual information I(X; Y).

    Attributes
    ----------
    mutual_info_bits : float
        I(X; Y) in bits.
    entropy_x_bits : float
        Marginal entropy H(X) in bits.
    entropy_y_bits : float
        Marginal entropy H(Y) in bits.
    conditional_entropy_y_given_x_bits : float
        H(Y|X) in bits.
    conditional_entropy_x_given_y_bits : float
        H(X|Y) in bits.
    joint_entropy_bits : float
        H(X, Y) in bits.
    normalized : float
        Normalized mutual information I(X;Y) / min(H(X), H(Y)).
    """

    mutual_info_bits: float
    entropy_x_bits: float = 0.0
    entropy_y_bits: float = 0.0
    conditional_entropy_y_given_x_bits: float = 0.0
    conditional_entropy_x_given_y_bits: float = 0.0
    joint_entropy_bits: float = 0.0
    normalized: float = 0.0

    @property
    def mutual_info_nats(self) -> float:
        return self.mutual_info_bits * 0.6931471805599453

    @property
    def variation_of_information(self) -> float:
        """VI = H(X|Y) + H(Y|X) — metric on random variable space."""
        return (self.conditional_entropy_x_given_y_bits
                + self.conditional_entropy_y_given_x_bits)


__all__ = [
    "Channel",
    "ChannelCapacity",
    "InformationState",
    "MutualInfoResult",
    "RateDistortionPoint",
]
