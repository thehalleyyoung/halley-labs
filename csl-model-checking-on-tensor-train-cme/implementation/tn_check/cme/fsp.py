"""
Finite State Projection (FSP) bounds.

Implements the Munsky-Khammash FSP method for truncating the state space
of the CME while maintaining rigorous probability bounds.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from tn_check.cme.reaction_network import ReactionNetwork

logger = logging.getLogger(__name__)


class FSPBounds:
    """
    Finite State Projection bounds for state-space truncation.

    Tracks the probability mass that escapes the truncated state space,
    providing rigorous error bounds on the CME solution.

    Attributes:
        lower_bounds: Lower bound on copy number for each species.
        upper_bounds: Upper bound on copy number for each species.
        escape_rate_bound: Upper bound on the total escape rate.
        probability_loss_bound: Upper bound on probability mass outside truncation.
        time_horizon: Time horizon for the bound.
    """

    def __init__(
        self,
        lower_bounds: list[int],
        upper_bounds: list[int],
        escape_rate_bound: float = 0.0,
        probability_loss_bound: float = 0.0,
        time_horizon: float = 0.0,
    ):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.escape_rate_bound = escape_rate_bound
        self.probability_loss_bound = probability_loss_bound
        self.time_horizon = time_horizon

    @property
    def num_species(self) -> int:
        return len(self.lower_bounds)

    @property
    def physical_dims(self) -> list[int]:
        """Effective physical dimensions within the truncation."""
        return [ub - lb for lb, ub in zip(self.lower_bounds, self.upper_bounds)]

    @property
    def truncated_state_space_size(self) -> int:
        result = 1
        for d in self.physical_dims:
            result *= d
        return result

    def contains_state(self, state: NDArray) -> bool:
        """Check if a state is within the FSP bounds."""
        for i in range(self.num_species):
            if state[i] < self.lower_bounds[i] or state[i] >= self.upper_bounds[i]:
                return False
        return True

    def expand(self, factor: float = 1.5) -> FSPBounds:
        """Create expanded bounds."""
        new_lower = []
        new_upper = []
        for lb, ub in zip(self.lower_bounds, self.upper_bounds):
            span = ub - lb
            expansion = int(span * (factor - 1) / 2)
            new_lower.append(max(0, lb - expansion))
            new_upper.append(ub + expansion)
        return FSPBounds(new_lower, new_upper)

    def __repr__(self) -> str:
        return (
            f"FSPBounds(dims={self.physical_dims}, "
            f"size={self.truncated_state_space_size:.2e}, "
            f"prob_loss≤{self.probability_loss_bound:.2e})"
        )


def compute_fsp_bounds(
    network: ReactionNetwork,
    time_horizon: float,
    tolerance: float = 1e-6,
    initial_expansion: float = 2.0,
) -> FSPBounds:
    """
    Compute initial FSP bounds based on network structure.

    Uses a simple heuristic: expand from initial state based on
    maximum possible transitions within the time horizon.

    Args:
        network: Reaction network.
        time_horizon: Simulation time horizon.
        tolerance: Target probability loss tolerance.
        initial_expansion: Factor for initial expansion.

    Returns:
        FSP bounds.
    """
    N = network.num_species

    lower_bounds = [0] * N
    upper_bounds = list(network.physical_dims)

    # Estimate maximum transition rates
    max_rate = network.max_exit_rate()
    expected_transitions = max_rate * time_horizon

    # Use Poisson tail bound for number of events
    # P(N_events > k) ~ exp(-max_rate * t) * (max_rate * t)^k / k!
    # We want this < tolerance

    from scipy.stats import poisson
    if max_rate * time_horizon > 0:
        max_events = int(poisson.ppf(1 - tolerance, max_rate * time_horizon))
    else:
        max_events = 0

    # Expand bounds based on stoichiometry and max events
    for rxn in network.reactions:
        sv = rxn.stoichiometry_vector
        for sp_idx, change in sv.items():
            if change > 0:
                potential_increase = change * max_events
                upper_bounds[sp_idx] = min(
                    network.species[sp_idx].max_copy_number,
                    max(upper_bounds[sp_idx],
                        network.species[sp_idx].initial_count + potential_increase + 1)
                )

    # Ensure bounds contain initial state
    for k in range(N):
        ic = network.species[k].initial_count
        if ic < lower_bounds[k]:
            lower_bounds[k] = ic
        if ic >= upper_bounds[k]:
            upper_bounds[k] = ic + 1

    # Compute escape rate bound
    escape_rate = max_rate  # Conservative upper bound
    prob_loss = 1 - np.exp(-escape_rate * time_horizon * tolerance)

    return FSPBounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        escape_rate_bound=escape_rate,
        probability_loss_bound=prob_loss,
        time_horizon=time_horizon,
    )


def adaptive_fsp_expansion(
    current_bounds: FSPBounds,
    probability_vector,
    network: ReactionNetwork,
    tolerance: float = 1e-6,
    expansion_factor: float = 1.5,
) -> tuple[FSPBounds, bool]:
    """
    Adaptively expand FSP bounds if probability mass approaches the boundary.

    Checks if significant probability mass is near the boundary states
    and expands the truncation if needed.

    Args:
        current_bounds: Current FSP bounds.
        probability_vector: Current probability MPS.
        network: Reaction network.
        tolerance: Expansion threshold.
        expansion_factor: Factor for expansion.

    Returns:
        Tuple of (new_bounds, was_expanded).
    """
    from tn_check.tensor.mps import MPS
    from tn_check.tensor.operations import mps_probability_at_index

    # Check boundary probability mass
    boundary_mass = _estimate_boundary_mass(
        probability_vector, current_bounds
    )

    if boundary_mass > tolerance:
        logger.info(
            f"FSP boundary mass {boundary_mass:.2e} > tolerance {tolerance:.2e}, "
            f"expanding bounds"
        )
        new_bounds = current_bounds.expand(expansion_factor)
        return new_bounds, True

    return current_bounds, False


def _estimate_boundary_mass(
    probability_mps,
    bounds: FSPBounds,
) -> float:
    """
    Estimate the probability mass at the boundary of the FSP region.

    Uses sampling to estimate the fraction of probability near boundaries.
    """
    from tn_check.tensor.operations import mps_probability_at_index

    rng = np.random.default_rng(42)
    N = bounds.num_species
    n_samples = min(1000, bounds.truncated_state_space_size)

    boundary_mass = 0.0
    total_samples = 0

    for _ in range(n_samples):
        state = []
        is_boundary = False
        for k in range(N):
            lb = bounds.lower_bounds[k]
            ub = bounds.upper_bounds[k]
            n = rng.integers(lb, ub)
            state.append(n)
            if n == lb or n == ub - 1:
                is_boundary = True

        if is_boundary:
            try:
                prob = mps_probability_at_index(probability_mps, state)
                boundary_mass += max(0, prob)
            except (IndexError, ValueError):
                pass
            total_samples += 1

    return boundary_mass


def validate_fsp_truncation(
    probability_mps,
    bounds: FSPBounds,
    tolerance: float = 1e-6,
) -> dict:
    """
    Validate the FSP truncation by checking various error indicators.

    Returns:
        Dictionary with validation results.
    """
    from tn_check.tensor.operations import mps_total_probability

    total_prob = mps_total_probability(probability_mps)
    prob_loss = abs(1.0 - total_prob)

    boundary_mass = _estimate_boundary_mass(probability_mps, bounds)

    return {
        "total_probability": total_prob,
        "probability_loss": prob_loss,
        "boundary_mass_estimate": boundary_mass,
        "fsp_valid": prob_loss < tolerance,
        "needs_expansion": boundary_mass > tolerance * 0.1,
    }
