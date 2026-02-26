"""
Initial state construction for CME.

Provides methods to create MPS representations of common initial
probability distributions.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson as poisson_dist
from scipy.stats import binom as binom_dist

from tn_check.tensor.mps import MPS, product_mps, unit_mps
from tn_check.cme.reaction_network import ReactionNetwork

logger = logging.getLogger(__name__)


def deterministic_initial_state(
    network: ReactionNetwork,
    initial_counts: Optional[list[int]] = None,
) -> MPS:
    """
    Create a deterministic initial state (delta distribution).

    p(n) = 1 if n = n_0, 0 otherwise.

    This is a rank-1 MPS.

    Args:
        network: Reaction network.
        initial_counts: Initial copy numbers. Defaults to network's initial state.

    Returns:
        MPS representing the initial delta distribution.
    """
    if initial_counts is None:
        initial_counts = network.initial_state

    phys = network.physical_dims

    return unit_mps(
        num_sites=network.num_species,
        physical_dims=phys,
        indices=initial_counts,
    )


def poisson_initial_state(
    network: ReactionNetwork,
    means: Optional[list[float]] = None,
) -> MPS:
    """
    Create a Poisson initial state where each species independently
    follows a Poisson distribution.

    p(n) = prod_i Poisson(n_i; lambda_i)

    This is a rank-1 MPS since the distribution factorizes.

    Args:
        network: Reaction network.
        means: Mean copy numbers. Defaults to network's initial state.

    Returns:
        MPS representing the product-Poisson distribution.
    """
    if means is None:
        means = [float(ic) for ic in network.initial_state]

    phys = network.physical_dims

    vectors = []
    for k in range(network.num_species):
        d = phys[k]
        lam = means[k]
        probs = np.zeros(d, dtype=np.float64)
        for n in range(d):
            probs[n] = poisson_dist.pmf(n, lam) if lam > 0 else (1.0 if n == 0 else 0.0)

        # Renormalize to account for truncation
        total = probs.sum()
        if total > 0:
            probs /= total

        vectors.append(probs)

    return product_mps(vectors)


def binomial_initial_state(
    network: ReactionNetwork,
    total_counts: list[int],
    probabilities: list[float],
) -> MPS:
    """
    Create a binomial initial state where each species independently
    follows a binomial distribution.

    p(n_i) = Binomial(n_i; N_i, p_i)

    Args:
        network: Reaction network.
        total_counts: N parameter for each species.
        probabilities: p parameter for each species.

    Returns:
        MPS representing the product-binomial distribution.
    """
    phys = network.physical_dims

    vectors = []
    for k in range(network.num_species):
        d = phys[k]
        N_k = total_counts[k]
        p_k = probabilities[k]
        probs = np.zeros(d, dtype=np.float64)
        for n in range(min(d, N_k + 1)):
            probs[n] = binom_dist.pmf(n, N_k, p_k)

        total = probs.sum()
        if total > 0:
            probs /= total

        vectors.append(probs)

    return product_mps(vectors)


def thermal_initial_state(
    network: ReactionNetwork,
    temperature: float = 1.0,
    energy_function: Optional[callable] = None,
) -> MPS:
    """
    Create a thermal (Boltzmann) initial state.

    p(n) proportional to exp(-E(n) / T)

    For the default energy function, E(n) = sum_i n_i (harmonic).

    Args:
        network: Reaction network.
        temperature: Temperature parameter.
        energy_function: Energy function (defaults to harmonic).

    Returns:
        MPS representing the thermal distribution.
    """
    phys = network.physical_dims

    vectors = []
    for k in range(network.num_species):
        d = phys[k]
        probs = np.zeros(d, dtype=np.float64)
        for n in range(d):
            if energy_function is not None:
                state = np.zeros(network.num_species)
                state[k] = n
                energy = energy_function(state)
            else:
                energy = float(n)  # Simple harmonic

            probs[n] = np.exp(-energy / max(temperature, 1e-300))

        total = probs.sum()
        if total > 0:
            probs /= total

        vectors.append(probs)

    return product_mps(vectors)


def mixture_initial_state(
    network: ReactionNetwork,
    states: list[list[int]],
    weights: list[float],
    max_bond_dim: Optional[int] = None,
) -> MPS:
    """
    Create a mixture initial state: p = sum_k w_k * delta_{n_k}.

    Args:
        network: Reaction network.
        states: List of states (copy number configurations).
        weights: Mixture weights (must sum to 1).
        max_bond_dim: Max bond dim for compression.

    Returns:
        MPS representing the mixture distribution.
    """
    from tn_check.tensor.operations import mps_weighted_sum

    phys = network.physical_dims
    mps_list = []
    for state in states:
        mps_list.append(unit_mps(network.num_species, phys, state))

    result = mps_weighted_sum(mps_list, weights, max_bond_dim=max_bond_dim)
    return result


def uniform_initial_state(
    network: ReactionNetwork,
) -> MPS:
    """
    Create a uniform initial state over all copy numbers.

    Args:
        network: Reaction network.

    Returns:
        MPS representing uniform distribution.
    """
    from tn_check.tensor.mps import uniform_mps
    return uniform_mps(network.num_species, network.physical_dims)
