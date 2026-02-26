"""
CME-to-MPO compiler.

Compiles a reaction network into an MPO representation of the CME generator.
The generator Q decomposes as a sum of Kronecker products:

    Q = sum_j tensor_product_i A_i^(j)

where each factor A_i^(j) is at most pentadiagonal because each elementary
reaction touches at most a few species.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from tn_check.tensor.mps import MPS
from tn_check.tensor.mpo import (
    MPO,
    identity_mpo,
    number_mpo,
    shift_mpo,
    propensity_diagonal_mpo,
)
from tn_check.tensor.algebra import sum_mpo
from tn_check.tensor.operations import mpo_addition
from tn_check.cme.reaction_network import (
    ReactionNetwork,
    Reaction,
    KineticsType,
    MassActionPropensity,
)

logger = logging.getLogger(__name__)


class CMECompiler:
    """
    Compiler that transforms a ReactionNetwork into an MPO generator.

    The CME generator Q has the structure:
        Q = sum over reactions j of (propensity_diag_j * shift_j - propensity_diag_j)
    where:
        - propensity_diag_j is the diagonal operator with a_j(n) on diagonal
        - shift_j applies the stoichiometry change
        - The second term subtracts the diagonal (exit rate) contribution

    Each term has exact Kronecker product structure when the propensity
    factors over species (always true for mass-action, approximated for others).
    """

    def __init__(
        self,
        network: ReactionNetwork,
        max_bond_dim: Optional[int] = None,
        compression_tolerance: float = 1e-12,
        use_conservation_laws: bool = True,
    ):
        self.network = network
        self.max_bond_dim = max_bond_dim
        self.compression_tolerance = compression_tolerance
        self.use_conservation_laws = use_conservation_laws
        self._compiled_mpo: Optional[MPO] = None
        self._reaction_mpos: list[MPO] = []
        self._diagonal_correction: Optional[MPO] = None
        self._max_exit_rate: float = 0.0

    @property
    def physical_dims(self) -> list[int]:
        return self.network.physical_dims

    @property
    def num_sites(self) -> int:
        return self.network.num_species

    def compile(self) -> MPO:
        """
        Compile the full CME generator as an MPO.

        Returns:
            MPO representing the infinitesimal generator Q.
        """
        logger.info(
            f"Compiling CME generator for network '{self.network.name}' "
            f"({self.network.num_species} species, {self.network.num_reactions} reactions)"
        )

        reaction_mpos = []
        for rxn in self.network.reactions:
            mpo = compile_reaction_to_mpo(rxn, self.network)
            reaction_mpos.append(mpo)

        self._reaction_mpos = reaction_mpos

        if not reaction_mpos:
            return identity_mpo(self.num_sites, self.physical_dims)

        # Sum all reaction contributions
        result = sum_mpo(
            reaction_mpos,
            compress=self.max_bond_dim is not None,
            max_bond_dim=self.max_bond_dim,
            tolerance=self.compression_tolerance,
        )

        # Add diagonal correction to ensure zero column sums
        result = self._add_diagonal_correction(result)

        self._compiled_mpo = result

        logger.info(
            f"CME generator compiled: bond dims = {result.bond_dims}, "
            f"max D = {result.max_bond_dim}, "
            f"total params = {result.total_params}"
        )

        return result

    def _add_diagonal_correction(self, mpo: MPO) -> MPO:
        """
        Add diagonal correction to ensure zero column sums.

        The CME generator must satisfy sum_i Q[i,j] = 0 for all j.
        The off-diagonal terms come from the reaction shift operators,
        and we need the diagonal to be -sum of propensities.
        """
        # Build the diagonal exit-rate operator
        N = self.num_sites
        phys = self.physical_dims

        # For each state, the exit rate is the sum of all propensities
        # Build this as an MPO with the negative sum of propensities on diagonal

        diag_mpos = []
        for rxn in self.network.reactions:
            diag_mpo = _build_propensity_diagonal_mpo(rxn, self.network)
            diag_mpos.append(diag_mpo)

        if diag_mpos:
            total_diag = sum_mpo(diag_mpos)
            total_diag.scale(-1.0)  # Negative exit rate
            self._diagonal_correction = total_diag
            mpo = mpo_addition(mpo, total_diag)

        return mpo

    def get_max_exit_rate(self) -> float:
        """
        Compute/estimate the maximum exit rate for uniformization.

        Returns:
            Maximum exit rate q = max_n sum_j a_j(n).
        """
        if self._max_exit_rate > 0:
            return self._max_exit_rate

        self._max_exit_rate = self.network.max_exit_rate()
        return self._max_exit_rate

    def get_uniformization_rate(self, safety_factor: float = 1.01) -> float:
        """Get the uniformization rate q >= max exit rate."""
        return self.get_max_exit_rate() * safety_factor

    def get_transition_probability_mpo(self) -> MPO:
        """
        Get the transition probability matrix P = I + Q/q for uniformization.

        Returns:
            MPO for P.
        """
        q = self.get_uniformization_rate()
        if self._compiled_mpo is None:
            self.compile()

        from tn_check.tensor.operations import mpo_scalar_multiply

        P = mpo_scalar_multiply(self._compiled_mpo, 1.0 / q)
        I = identity_mpo(self.num_sites, self.physical_dims)
        P = mpo_addition(P, I)

        return P


def compile_reaction_to_mpo(
    reaction: Reaction,
    network: ReactionNetwork,
) -> MPO:
    """
    Compile a single reaction into an MPO contribution to the generator.

    For reaction j with propensity a_j(n) and stoichiometry vector v_j:
        Q_j[m, n] = a_j(n) * delta_{m, n + v_j}   for m != n

    In MPO form, this is the product of:
    1. The shift operator (stoichiometry change): |n + v_j><n|
    2. The propensity diagonal: diag(a_j(n))

    Each factor acts on at most one or two species, giving a Kronecker
    product of single-site operators with identities on uninvolved species.

    Args:
        reaction: The reaction to compile.
        network: The reaction network (for species info).

    Returns:
        MPO contribution for this reaction.
    """
    N = network.num_species
    phys = network.physical_dims

    # Build the off-diagonal part: propensity * shift
    # For each species, determine the local operator

    species_factors = _compute_species_factors(reaction, network)

    cores = []
    for k in range(N):
        d = phys[k]
        prop_factor, shift_amount = species_factors[k]

        # Build the local operator matrix: (d x d)
        # O[m, n] = prop_factor[n] * delta_{m, n + shift}
        local_op = np.zeros((d, d), dtype=np.float64)
        for n in range(d):
            m = n + shift_amount
            if 0 <= m < d:
                local_op[m, n] = prop_factor[n]

        # Wrap as MPO core: (1, d, d, 1)
        core = local_op.reshape(1, d, d, 1)
        cores.append(core)

    return MPO(cores, copy_cores=False)


def _compute_species_factors(
    reaction: Reaction,
    network: ReactionNetwork,
) -> list[tuple[NDArray, int]]:
    """
    Compute per-species factors for a reaction.

    Returns a list of (propensity_factor, shift_amount) for each species.
    - propensity_factor[n] is the contribution of species k at copy number n
    - shift_amount is the net stoichiometry change for species k
    """
    N = network.num_species
    phys = network.physical_dims

    # Stoichiometry vector
    stoich = reaction.stoichiometry_vector

    # Propensity factors
    species_indices = list(range(N))
    prop_factors = reaction.propensity.per_species_factors(
        species_indices, phys
    )

    # Distribute rate constant into first involved species
    rate_distributed = False

    result = []
    for k in range(N):
        shift = stoich.get(k, 0)
        factor = prop_factors[k].copy()

        # For mass-action: distribute rate constant
        if not rate_distributed and isinstance(reaction.propensity, MassActionPropensity):
            if k in reaction.propensity.reactant_species or k == 0:
                factor *= reaction.propensity.rate_constant
                rate_distributed = True

        result.append((factor, shift))

    # If rate constant not yet distributed, apply to first species
    if not rate_distributed and isinstance(reaction.propensity, MassActionPropensity):
        factor, shift = result[0]
        factor = factor * reaction.propensity.rate_constant
        result[0] = (factor, shift)

    return result


def _build_propensity_diagonal_mpo(
    reaction: Reaction,
    network: ReactionNetwork,
) -> MPO:
    """
    Build the diagonal MPO for a reaction's propensity function.

    diag(a_j(n)) = tensor product of per-species diagonal factors.
    """
    N = network.num_species
    phys = network.physical_dims

    species_indices = list(range(N))
    prop_factors = reaction.propensity.per_species_factors(
        species_indices, phys
    )

    cores = []
    rate_applied = False
    for k in range(N):
        d = phys[k]
        factor = prop_factors[k]

        if not rate_applied and isinstance(reaction.propensity, MassActionPropensity):
            if k in reaction.propensity.reactant_species or k == 0:
                factor = factor * reaction.propensity.rate_constant
                rate_applied = True

        core = np.zeros((1, d, d, 1), dtype=np.float64)
        for n in range(d):
            if n < len(factor):
                core[0, n, n, 0] = factor[n]

        cores.append(core)

    if not rate_applied and isinstance(reaction.propensity, MassActionPropensity):
        cores[0] = cores[0] * reaction.propensity.rate_constant

    return MPO(cores, copy_cores=False)


def compile_network_to_mpo(
    network: ReactionNetwork,
    max_bond_dim: Optional[int] = None,
    compression_tolerance: float = 1e-12,
) -> MPO:
    """
    Convenience function to compile a reaction network to an MPO.

    Args:
        network: Reaction network.
        max_bond_dim: Maximum bond dimension.
        compression_tolerance: Compression tolerance.

    Returns:
        MPO generator.
    """
    compiler = CMECompiler(
        network,
        max_bond_dim=max_bond_dim,
        compression_tolerance=compression_tolerance,
    )
    return compiler.compile()


def compile_propensity_to_diagonal(
    network: ReactionNetwork,
    reaction_index: int,
) -> MPO:
    """
    Compile a single reaction's propensity as a diagonal MPO.

    Args:
        network: Reaction network.
        reaction_index: Index of the reaction.

    Returns:
        Diagonal MPO with propensity values.
    """
    rxn = network.reactions[reaction_index]
    return _build_propensity_diagonal_mpo(rxn, network)


def build_uniformization_mpo(
    network: ReactionNetwork,
    max_bond_dim: Optional[int] = None,
) -> tuple[MPO, float]:
    """
    Build the uniformized transition probability MPO.

    P = I + Q / q where q = max exit rate.

    Returns:
        Tuple of (transition probability MPO, uniformization rate q).
    """
    compiler = CMECompiler(network, max_bond_dim=max_bond_dim)
    compiler.compile()
    q = compiler.get_uniformization_rate()
    P = compiler.get_transition_probability_mpo()
    return P, q


class CMEGeneratorAnalyzer:
    """Analysis tools for the compiled CME generator."""

    def __init__(self, mpo: MPO, network: ReactionNetwork):
        self.mpo = mpo
        self.network = network

    def check_generator_properties(self) -> dict:
        """Check key properties of the generator."""
        results = {}

        # Column sum check
        col_sum_error = self.mpo.check_column_sum_zero()
        results["max_column_sum_deviation"] = col_sum_error
        results["column_sums_zero"] = col_sum_error < 1e-8

        # Metzler check (for small systems)
        total_size = np.prod(self.network.physical_dims)
        if total_size <= 10000:
            results["is_metzler"] = self.mpo.check_metzler()
        else:
            results["is_metzler"] = None  # Too large to check

        results["bond_dims"] = self.mpo.bond_dims
        results["max_bond_dim"] = self.mpo.max_bond_dim
        results["total_params"] = self.mpo.total_params

        return results

    def spectral_gap_estimate(self, num_eigenvalues: int = 5) -> NDArray:
        """
        Estimate the leading eigenvalues of the generator.

        Only feasible for small systems.
        """
        from tn_check.tensor.operations import mpo_to_dense

        total_size = np.prod(self.network.physical_dims)
        if total_size > 50000:
            logger.warning("System too large for direct eigenvalue computation")
            return np.array([])

        Q = mpo_to_dense(self.mpo)

        if total_size < 5000:
            eigenvalues = np.linalg.eigvals(Q)
            # Sort by real part (descending)
            idx = np.argsort(-np.real(eigenvalues))
            return eigenvalues[idx[:num_eigenvalues]]
        else:
            from scipy.sparse.linalg import eigs
            from scipy.sparse import csr_matrix
            Q_sparse = csr_matrix(Q)
            eigenvalues, _ = eigs(Q_sparse, k=num_eigenvalues, which="LR")
            return eigenvalues

    def stiffness_ratio(self) -> float:
        """
        Estimate the stiffness ratio max|lambda| / min|nonzero lambda|.

        High stiffness requires implicit methods or uniformization.
        """
        evals = self.spectral_gap_estimate(num_eigenvalues=10)
        if len(evals) < 2:
            return 1.0

        real_parts = np.abs(np.real(evals))
        real_parts = real_parts[real_parts > 1e-15]

        if len(real_parts) < 2:
            return 1.0

        return float(np.max(real_parts) / np.min(real_parts))
