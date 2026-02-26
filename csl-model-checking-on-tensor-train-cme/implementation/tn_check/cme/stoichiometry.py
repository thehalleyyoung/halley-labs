"""
Stoichiometry matrix computation and conservation law detection.

The stoichiometry matrix S has entry S[i,j] = net change in species i
when reaction j fires. Conservation laws are found as left null vectors
of S: c^T S = 0 implies c^T n(t) = c^T n(0) for all t.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import null_space

from tn_check.cme.reaction_network import ReactionNetwork

logger = logging.getLogger(__name__)


class StoichiometryMatrix:
    """
    Stoichiometry matrix and derived quantities.

    Attributes:
        matrix: S[i,j] = net change in species i from reaction j.
        reactant_matrix: R[i,j] = stoichiometric coefficient of species i as reactant in j.
        product_matrix: P[i,j] = stoichiometric coefficient of species i as product in j.
        conservation_vectors: Rows of the left null space of S.
        conservation_constants: Constants c^T n(0) for each conservation law.
    """

    def __init__(
        self,
        matrix: NDArray,
        reactant_matrix: NDArray,
        product_matrix: NDArray,
        species_names: list[str],
        reaction_names: list[str],
    ):
        self.matrix = matrix
        self.reactant_matrix = reactant_matrix
        self.product_matrix = product_matrix
        self.species_names = species_names
        self.reaction_names = reaction_names
        self.conservation_vectors: Optional[NDArray] = None
        self.conservation_constants: Optional[NDArray] = None
        self._rank: Optional[int] = None

    @property
    def num_species(self) -> int:
        return self.matrix.shape[0]

    @property
    def num_reactions(self) -> int:
        return self.matrix.shape[1]

    @property
    def rank(self) -> int:
        if self._rank is None:
            self._rank = int(np.linalg.matrix_rank(self.matrix))
        return self._rank

    @property
    def num_conservation_laws(self) -> int:
        return self.num_species - self.rank

    @property
    def deficiency(self) -> int:
        """
        Network deficiency (delta = #complexes - #linkage_classes - rank(S)).

        Simplified computation: just return num_species - rank for now.
        """
        return self.num_species - self.rank

    def find_conservation_laws(
        self,
        initial_state: Optional[NDArray] = None,
    ) -> tuple[NDArray, Optional[NDArray]]:
        """
        Find conservation laws as left null vectors of S.

        A conservation law c satisfies c^T S = 0, meaning c^T n(t) = const.

        Args:
            initial_state: Initial copy numbers (to compute conservation constants).

        Returns:
            Tuple of (conservation_vectors, conservation_constants).
        """
        ns = null_space(self.matrix.T)

        if ns.shape[1] == 0:
            self.conservation_vectors = np.empty((0, self.num_species))
            self.conservation_constants = np.empty(0)
            return self.conservation_vectors, self.conservation_constants

        # Each column of ns is a conservation vector
        self.conservation_vectors = ns.T  # shape: (num_laws, num_species)

        # Normalize so entries are approximately integer
        for i in range(self.conservation_vectors.shape[0]):
            vec = self.conservation_vectors[i]
            # Scale by smallest nonzero entry
            nonzero = np.abs(vec[np.abs(vec) > 1e-10])
            if len(nonzero) > 0:
                scale = np.min(nonzero)
                self.conservation_vectors[i] = vec / scale

            # Round near-integers
            rounded = np.round(self.conservation_vectors[i])
            if np.allclose(self.conservation_vectors[i], rounded, atol=1e-6):
                self.conservation_vectors[i] = rounded

        if initial_state is not None:
            initial_state = np.asarray(initial_state, dtype=np.float64)
            self.conservation_constants = self.conservation_vectors @ initial_state
        else:
            self.conservation_constants = None

        logger.info(
            f"Found {self.conservation_vectors.shape[0]} conservation laws"
        )

        return self.conservation_vectors, self.conservation_constants

    def get_reachable_bounds(
        self,
        initial_state: NDArray,
        max_copy_number: Optional[int] = None,
    ) -> list[tuple[int, int]]:
        """
        Compute tighter copy-number bounds using conservation laws.

        Args:
            initial_state: Initial state.
            max_copy_number: Global upper bound.

        Returns:
            List of (lower, upper) bounds for each species.
        """
        N = self.num_species
        initial_state = np.asarray(initial_state, dtype=np.float64)

        bounds = []
        for i in range(N):
            lb = 0
            ub = max_copy_number if max_copy_number is not None else 1000
            bounds.append((lb, ub))

        if self.conservation_vectors is not None and self.conservation_constants is not None:
            for c_vec, c_const in zip(
                self.conservation_vectors, self.conservation_constants
            ):
                for i in range(N):
                    if abs(c_vec[i]) > 1e-10:
                        # c_vec @ n = c_const
                        # If c_vec[i] > 0: n[i] <= c_const / c_vec[i]
                        #   (assuming other terms >= 0)
                        if c_vec[i] > 0:
                            upper = int(np.floor(c_const / c_vec[i]))
                            bounds[i] = (bounds[i][0], min(bounds[i][1], upper))

        return bounds

    def __repr__(self) -> str:
        return (
            f"StoichiometryMatrix({self.num_species}x{self.num_reactions}, "
            f"rank={self.rank}, conservation_laws={self.num_conservation_laws})"
        )


def compute_stoichiometry(network: ReactionNetwork) -> StoichiometryMatrix:
    """
    Compute the stoichiometry matrix for a reaction network.

    Args:
        network: Reaction network.

    Returns:
        StoichiometryMatrix object.
    """
    N = network.num_species
    M = network.num_reactions

    S = np.zeros((N, M), dtype=np.float64)
    R = np.zeros((N, M), dtype=np.float64)
    P = np.zeros((N, M), dtype=np.float64)

    for j, rxn in enumerate(network.reactions):
        for sp_idx, stoich in zip(rxn.reactant_species, rxn.reactant_stoichiometry):
            R[sp_idx, j] = stoich
            S[sp_idx, j] -= stoich

        for sp_idx, stoich in zip(rxn.product_species, rxn.product_stoichiometry):
            P[sp_idx, j] = stoich
            S[sp_idx, j] += stoich

    species_names = [sp.name for sp in network.species]
    reaction_names = [rxn.name for rxn in network.reactions]

    return StoichiometryMatrix(S, R, P, species_names, reaction_names)


def find_conservation_laws(
    network: ReactionNetwork,
) -> tuple[NDArray, NDArray]:
    """
    Find conservation laws for a reaction network.

    Returns:
        Tuple of (conservation_vectors, conservation_constants).
    """
    stoich = compute_stoichiometry(network)
    initial_state = np.array(network.initial_state, dtype=np.float64)
    return stoich.find_conservation_laws(initial_state)


def compute_reachable_bounds(
    network: ReactionNetwork,
    global_max: Optional[int] = None,
) -> list[tuple[int, int]]:
    """
    Compute tighter copy-number bounds using conservation laws.

    Args:
        network: Reaction network.
        global_max: Global upper bound on copy numbers.

    Returns:
        List of (lower, upper) bounds for each species.
    """
    stoich = compute_stoichiometry(network)
    initial_state = np.array(network.initial_state, dtype=np.float64)
    stoich.find_conservation_laws(initial_state)

    if global_max is None:
        global_max = max(sp.max_copy_number for sp in network.species)

    return stoich.get_reachable_bounds(initial_state, max_copy_number=global_max)
