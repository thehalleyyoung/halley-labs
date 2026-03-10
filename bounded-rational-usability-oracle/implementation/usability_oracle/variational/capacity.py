"""
usability_oracle.variational.capacity — Channel capacity estimation.

Estimates the effective information-processing capacity of human cognitive
and motor channels, following the information-theoretic framework of
Cover & Thomas and HCI-specific models (Fitts, Hick, etc.).

Provides:

* :class:`CapacityEstimatorImpl` — full capacity estimation engine
* :func:`estimate_fitts_capacity` — motor channel (Fitts/MacKenzie 1992)
* :func:`estimate_hick_capacity` — choice reaction channel
* :func:`estimate_visual_capacity` — visual search channel
* :func:`estimate_memory_capacity` — working-memory channel
* :func:`compose_capacities` — serial/parallel composition
* :func:`blahut_arimoto` — Blahut–Arimoto algorithm for channel capacity
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from usability_oracle.variational.types import (
    CapacityProfile,
    ConvergenceInfo,
    ConvergenceStatus,
    FreeEnergyResult,
    KLDivergenceResult,
    VariationalConfig,
)
from usability_oracle.variational.kl_divergence import compute_policy_kl

logger = logging.getLogger(__name__)

_EPS = 1e-30


# ═══════════════════════════════════════════════════════════════════════════
# Standalone capacity estimation functions
# ═══════════════════════════════════════════════════════════════════════════

def estimate_fitts_capacity(
    distance: float,
    width: float,
    throughput: float = 4.9,
) -> float:
    r"""Estimate motor-channel capacity via Fitts' law (MacKenzie 1992).

    The index of difficulty is:

    .. math::

        \mathrm{ID} = \log_2\!\Bigl(\frac{D}{W} + 1\Bigr)

    and the throughput (bits/sec) follows from the Shannon formulation.

    Parameters
    ----------
    distance : float
        Movement distance D (pixels or mm).
    width : float
        Target width W (same units as *distance*).
    throughput : float
        Empirical throughput IP in bits/sec.  Default 4.9 bits/s is a
        typical value from ISO 9241-411 studies (MacKenzie, 1992).

    Returns
    -------
    float
        Estimated motor capacity in bits/sec.

    Raises
    ------
    ValueError
        If *distance* < 0 or *width* ≤ 0.

    References
    ----------
    MacKenzie, I. S. (1992). Fitts' law as a research and design tool in
    human-computer interaction. *Human-Computer Interaction*, 7(1), 91–139.
    """
    if distance < 0:
        raise ValueError(f"distance must be ≥ 0, got {distance}")
    if width <= 0:
        raise ValueError(f"width must be > 0, got {width}")

    index_of_difficulty = math.log2(distance / width + 1.0)
    # Capacity = throughput (bits/sec); ID gives bits/movement
    # Movement time = ID / throughput, so bits/sec = throughput
    # For a single movement, the channel carries ID bits in ID/throughput seconds
    return throughput * index_of_difficulty / max(index_of_difficulty, _EPS)


def estimate_hick_capacity(
    n_alternatives: int,
    stimulus_probs: Optional[np.ndarray] = None,
) -> float:
    r"""Estimate choice-channel capacity via Hick–Hyman law.

    .. math::

        H = -\sum_i p_i \log_2 p_i

    For equally probable stimuli, H = log₂(n).  The capacity is H bits
    per decision (typically ~2.5–7 bits/sec in practice).

    Parameters
    ----------
    n_alternatives : int
        Number of stimulus–response alternatives.
    stimulus_probs : np.ndarray, optional
        Stimulus probabilities.  Uniform if ``None``.

    Returns
    -------
    float
        Choice entropy in bits (capacity per decision).

    Raises
    ------
    ValueError
        If *n_alternatives* < 1.
    """
    if n_alternatives < 1:
        raise ValueError(f"n_alternatives must be ≥ 1, got {n_alternatives}")

    if stimulus_probs is not None:
        p = np.asarray(stimulus_probs, dtype=np.float64).ravel()
        if p.shape[0] != n_alternatives:
            raise ValueError("stimulus_probs length must match n_alternatives")
        p = np.maximum(p, _EPS)
        p /= p.sum()
        return float(-np.sum(p * np.log2(p)))

    return math.log2(n_alternatives) if n_alternatives > 1 else 0.0


def estimate_visual_capacity(
    eccentricity: float,
    crowding: float = 0.5,
    n_items: int = 1,
) -> float:
    r"""Estimate visual-search channel capacity.

    Models the visual channel as limited by:

    * **Eccentricity**: acuity drops as 1/(1 + k·e) where e is eccentricity
      in degrees of visual angle.
    * **Crowding**: nearby flankers reduce discriminability by factor
      (1 − crowding).
    * **Set size**: effective bits per fixation ≈ log₂(n_items) scaled by
      acuity and crowding.

    Parameters
    ----------
    eccentricity : float
        Eccentricity in degrees of visual angle (0 = fovea).
    crowding : float
        Crowding factor in [0, 1).  0 = no crowding, higher = worse.
    n_items : int
        Number of items in the visual field.

    Returns
    -------
    float
        Estimated visual capacity in bits per fixation.
    """
    if eccentricity < 0:
        raise ValueError(f"eccentricity must be ≥ 0, got {eccentricity}")
    if not (0 <= crowding < 1):
        raise ValueError(f"crowding must be in [0, 1), got {crowding}")
    if n_items < 1:
        raise ValueError(f"n_items must be ≥ 1, got {n_items}")

    # Acuity scaling: Bouma's law for critical spacing
    acuity_factor = 1.0 / (1.0 + 0.3 * eccentricity)

    # Crowding penalty
    crowding_factor = 1.0 - crowding

    # Base information: log₂(n) bits needed to identify one among n items
    base_bits = math.log2(n_items) if n_items > 1 else 1.0

    return base_bits * acuity_factor * crowding_factor


def estimate_memory_capacity(
    n_chunks: int = 4,
    decay_rate: float = 0.0,
    rehearsal: float = 0.0,
) -> float:
    r"""Estimate working-memory channel capacity.

    Based on Cowan's (2001) 4 ± 1 chunk model with exponential decay
    and rehearsal recovery:

    .. math::

        C_{\mathrm{WM}} = n_{\mathrm{chunks}} \cdot e^{-\lambda\,(1 - r)}

    where λ is the decay rate and r is the rehearsal factor.  The result
    is in bits (log₂ of effective number of chunks).

    Parameters
    ----------
    n_chunks : int
        Nominal number of WM chunks (default 4, per Cowan 2001).
    decay_rate : float
        Temporal decay rate λ (≥ 0).  0 = no decay.
    rehearsal : float
        Rehearsal factor in [0, 1].  1 = full rehearsal (no decay loss).

    Returns
    -------
    float
        Effective WM capacity in bits.

    References
    ----------
    Cowan, N. (2001). The magical number 4 in short-term memory.
    *Behavioral and Brain Sciences*, 24(1), 87–114.
    """
    if n_chunks < 0:
        raise ValueError(f"n_chunks must be ≥ 0, got {n_chunks}")
    if decay_rate < 0:
        raise ValueError(f"decay_rate must be ≥ 0, got {decay_rate}")
    if not (0 <= rehearsal <= 1):
        raise ValueError(f"rehearsal must be in [0, 1], got {rehearsal}")

    if n_chunks == 0:
        return 0.0

    effective_chunks = n_chunks * math.exp(-decay_rate * (1.0 - rehearsal))
    return math.log2(max(effective_chunks, 1.0))


def compose_capacities(
    capacities: Sequence[float],
    mode: str = "serial",
) -> float:
    r"""Compose channel capacities for serial or parallel arrangements.

    * **Serial** (bottleneck): the overall capacity is limited by the
      slowest channel:  C = min(C₁, C₂, …).
    * **Parallel** (independent): capacities add:  C = C₁ + C₂ + ….

    Parameters
    ----------
    capacities : sequence of float
        Individual channel capacities (bits/sec or bits).
    mode : str
        ``"serial"`` or ``"parallel"``.

    Returns
    -------
    float
        Composed capacity.

    Raises
    ------
    ValueError
        If *mode* is unknown or *capacities* is empty.
    """
    if not capacities:
        raise ValueError("capacities must be non-empty")

    caps = [float(c) for c in capacities]

    if mode == "serial":
        return min(caps)
    elif mode == "parallel":
        return sum(caps)
    else:
        raise ValueError(f"Unknown mode '{mode}'; expected 'serial' or 'parallel'")


# ═══════════════════════════════════════════════════════════════════════════
# Blahut–Arimoto algorithm
# ═══════════════════════════════════════════════════════════════════════════

def blahut_arimoto(
    channel_matrix: np.ndarray,
    tolerance: float = 1e-8,
    max_iter: int = 1000,
) -> Tuple[float, np.ndarray]:
    r"""Compute channel capacity via the Blahut–Arimoto algorithm.

    Given a discrete memoryless channel with transition matrix
    W(y|x) of shape ``(|X|, |Y|)``, iteratively computes:

    .. math::

        C = \max_{p(x)}\; I(X; Y)

    by alternating between:

    1. **Q-step**: q(x|y) ∝ p(x) W(y|x)
    2. **P-step**: p(x) ∝ exp(Σ_y W(y|x) log q(x|y))

    Parameters
    ----------
    channel_matrix : np.ndarray
        Channel transition matrix W(y|x), shape ``(n_inputs, n_outputs)``.
        Each row must sum to 1 (or will be normalised).
    tolerance : float
        Convergence criterion on |C_new − C_old|.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    capacity : float
        Channel capacity in **bits**.
    input_dist : np.ndarray
        Capacity-achieving input distribution p*(x).

    References
    ----------
    Blahut, R. E. (1972). Computation of channel capacity and rate-distortion
    functions. *IEEE Transactions on Information Theory*, 18(4), 460–473.
    """
    W = np.asarray(channel_matrix, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"channel_matrix must be 2-D, got ndim={W.ndim}")

    n_x, n_y = W.shape

    # Normalise rows
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, _EPS)
    W = W / row_sums

    # Initialise uniform input distribution
    p_x = np.ones(n_x) / n_x

    capacity_prev = -np.inf

    for it in range(max_iter):
        # Joint p(x,y) = p(x) * W(y|x)
        joint = p_x[:, np.newaxis] * W  # (n_x, n_y)

        # Output distribution p(y) = sum_x p(x) W(y|x)
        p_y = joint.sum(axis=0)
        p_y = np.maximum(p_y, _EPS)

        # Compute I(X;Y) = sum_x,y p(x) W(y|x) log(W(y|x) / p(y))
        log_ratio = np.log(W / p_y[np.newaxis, :])
        # Avoid -inf * 0
        mi_terms = np.where(W > 0, W * log_ratio, 0.0)
        # c(x) = sum_y W(y|x) log(W(y|x) / p(y))
        c_x = mi_terms.sum(axis=1)

        # Capacity bounds
        capacity = float(np.dot(p_x, c_x)) / math.log(2.0)

        if abs(capacity - capacity_prev) < tolerance:
            logger.debug("Blahut-Arimoto converged in %d iterations", it + 1)
            break
        capacity_prev = capacity

        # Update input distribution: p(x) ∝ p(x) * exp(c(x))
        log_p = np.log(np.maximum(p_x, _EPS)) + c_x
        log_p -= logsumexp(log_p)
        p_x = np.exp(log_p)

    return capacity, p_x


# ═══════════════════════════════════════════════════════════════════════════
# CapacityEstimatorImpl class (implements CapacityEstimator protocol)
# ═══════════════════════════════════════════════════════════════════════════

class CapacityEstimatorImpl:
    """Concrete implementation of the :class:`CapacityEstimator` protocol.

    Sweeps rationality parameter β and computes the rate–distortion
    profile for a given cost matrix and reference policy.
    """

    def estimate_profile(
        self,
        cost_matrix: Dict[str, Dict[str, float]],
        reference_policy: Dict[str, Dict[str, float]],
        beta_range: Sequence[float],
        config: VariationalConfig,
    ) -> CapacityProfile:
        """Sweep β values and compute capacity profile.

        See :meth:`CapacityEstimator.estimate_profile` for full documentation.
        """
        from usability_oracle.variational.free_energy import FreeEnergyComputer

        beta_vals: List[float] = []
        cap_bits: List[float] = []
        exp_costs: List[float] = []
        kl_nats: List[float] = []

        for beta in beta_range:
            cfg = VariationalConfig(
                beta=beta,
                max_iterations=config.max_iterations,
                tolerance=config.tolerance,
                learning_rate=config.learning_rate,
                objective=config.objective,
                use_natural_gradient=config.use_natural_gradient,
                line_search=config.line_search,
                regularisation_lambda=config.regularisation_lambda,
                num_inner_iterations=config.num_inner_iterations,
                seed=config.seed,
            )
            computer = FreeEnergyComputer(cfg)
            result = computer.compute(cost_matrix, reference_policy)

            beta_vals.append(beta)
            kl_val = result.kl_divergence.total_kl
            kl_nats.append(kl_val)
            cap_bits.append(kl_val / math.log(2.0) if np.isfinite(kl_val) else float("inf"))
            exp_costs.append(result.expected_cost)

        # Identify Pareto front (lower cost, lower KL)
        pareto_indices = self._pareto_front(exp_costs, kl_nats)

        return CapacityProfile(
            beta_values=tuple(beta_vals),
            capacity_bits=tuple(cap_bits),
            expected_costs=tuple(exp_costs),
            kl_nats=tuple(kl_nats),
            pareto_front_indices=tuple(pareto_indices),
        )

    def compute_kl(
        self,
        policy: Dict[str, Dict[str, float]],
        reference: Dict[str, Dict[str, float]],
        state_distribution: Optional[Dict[str, float]] = None,
    ) -> KLDivergenceResult:
        """Compute KL(π ‖ π₀) with per-state breakdown.

        See :meth:`CapacityEstimator.compute_kl` for full documentation.
        """
        return compute_policy_kl(policy, reference, state_distribution)

    @staticmethod
    def _pareto_front(
        costs: List[float],
        rates: List[float],
    ) -> List[int]:
        """Find Pareto-optimal points (lower cost, lower rate is better)."""
        n = len(costs)
        if n == 0:
            return []

        indices = list(range(n))
        # Sort by cost
        indices.sort(key=lambda i: costs[i])

        pareto: List[int] = []
        min_rate = float("inf")
        for i in indices:
            if rates[i] < min_rate:
                pareto.append(i)
                min_rate = rates[i]

        pareto.sort()
        return pareto
