"""
Species ordering strategies for TT bond-dimension minimization.

The reaction network defines a hypergraph: species are nodes, reactions
are hyperedges. Species that co-occur in many reactions are correlated
and should be placed adjacent in the TT chain to minimize bond dimension.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def identity_ordering(n_species: int) -> list[int]:
    """Identity ordering (no reordering)."""
    return list(range(n_species))


def _build_interaction_matrix(
    reaction_network,
) -> NDArray:
    """
    Build species interaction matrix from a reaction network.

    A[i,j] = number of reactions involving both species i and j.
    """
    n = len(reaction_network.species)
    A = np.zeros((n, n), dtype=np.float64)

    for rxn in reaction_network.reactions:
        involved = set()
        for s in rxn.reactant_species:
            involved.add(s)
        for s in rxn.product_species:
            involved.add(s)
        # Also add species from propensity
        if hasattr(rxn.propensity, 'reactant_species'):
            for s in rxn.propensity.reactant_species:
                involved.add(s)
        if hasattr(rxn.propensity, 'species_index'):
            involved.add(rxn.propensity.species_index)

        involved_list = list(involved)
        for i in range(len(involved_list)):
            for j in range(i + 1, len(involved_list)):
                si, sj = involved_list[i], involved_list[j]
                if 0 <= si < n and 0 <= sj < n:
                    A[si, sj] += 1
                    A[sj, si] += 1

    return A


def reverse_cuthill_mckee(
    reaction_network,
) -> list[int]:
    """
    Reverse Cuthill-McKee ordering for bandwidth reduction.

    Reorders species so that strongly interacting species are adjacent,
    minimizing the bandwidth of the interaction matrix. This heuristic
    tends to reduce TT bond dimensions because correlations between
    distant species in the chain require higher bond dimensions.

    Args:
        reaction_network: ReactionNetwork instance.

    Returns:
        Permutation of species indices.
    """
    n = len(reaction_network.species)
    if n <= 2:
        return list(range(n))

    A = _build_interaction_matrix(reaction_network)

    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0:
                adj[i].append(j)

    # Start from node with minimum degree
    degrees = [len(adj[i]) for i in range(n)]
    start = int(np.argmin(degrees))

    # BFS to get Cuthill-McKee ordering
    visited = set()
    order = []
    queue = [start]
    visited.add(start)

    while queue:
        node = queue.pop(0)
        order.append(node)

        # Sort neighbors by degree (Cuthill-McKee refinement)
        neighbors = sorted(
            [v for v in adj[node] if v not in visited],
            key=lambda v: degrees[v],
        )
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                queue.append(v)

    # Add any isolated nodes
    for i in range(n):
        if i not in visited:
            order.append(i)

    # Reverse for RCM
    order.reverse()
    return order


def spectral_ordering(
    reaction_network,
) -> list[int]:
    """
    Spectral ordering using Fiedler vector of the graph Laplacian.

    The Fiedler vector (eigenvector of second-smallest eigenvalue of
    the Laplacian) provides a 1D embedding that minimizes the sum
    of squared distances between adjacent nodes.

    Args:
        reaction_network: ReactionNetwork instance.

    Returns:
        Permutation of species indices.
    """
    n = len(reaction_network.species)
    if n <= 2:
        return list(range(n))

    A = _build_interaction_matrix(reaction_network)
    D = np.diag(A.sum(axis=1))
    L = D - A  # Graph Laplacian

    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Fiedler vector: second smallest eigenvalue
    fiedler = eigenvectors[:, 1]

    # Sort species by Fiedler vector value
    order = list(np.argsort(fiedler))
    return order


def greedy_entanglement_ordering(
    reaction_network,
    n_trials: int = 5,
) -> list[int]:
    """
    Greedy ordering that minimizes estimated entanglement.

    At each step, places the species that has maximum interaction
    with the already-placed species, trying to keep strongly
    correlated species adjacent.

    Args:
        reaction_network: ReactionNetwork instance.
        n_trials: Number of random restarts.

    Returns:
        Best permutation found.
    """
    n = len(reaction_network.species)
    if n <= 2:
        return list(range(n))

    A = _build_interaction_matrix(reaction_network)

    best_order = None
    best_cost = float("inf")

    rng = np.random.default_rng(42)

    for trial in range(n_trials):
        if trial == 0:
            # First trial: start from highest-degree node
            start = int(np.argmax(A.sum(axis=1)))
        else:
            start = rng.integers(0, n)

        placed = [start]
        remaining = set(range(n)) - {start}

        while remaining:
            # Find species with max interaction to last placed
            last = placed[-1]
            best_next = max(remaining, key=lambda s: A[last, s])
            placed.append(best_next)
            remaining.remove(best_next)

        # Compute cost: sum of A[i,j] * |pos(i) - pos(j)| (bandwidth)
        pos = {s: i for i, s in enumerate(placed)}
        cost = sum(
            A[i, j] * abs(pos[i] - pos[j])
            for i in range(n) for j in range(i + 1, n)
        )

        if cost < best_cost:
            best_cost = cost
            best_order = placed

    return best_order
