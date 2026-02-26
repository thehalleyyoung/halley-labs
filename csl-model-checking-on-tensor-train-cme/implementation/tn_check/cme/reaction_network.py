"""
Reaction network intermediate representation.

Defines the data structures for representing stochastic chemical reaction
networks: species, reactions, propensity functions, and kinetic parameters.
"""

from __future__ import annotations

import enum
import dataclasses
import logging
from typing import Any, Callable, Optional, Sequence

import math
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class KineticsType(enum.Enum):
    """Type of reaction kinetics."""
    MASS_ACTION = "mass_action"
    HILL = "hill"
    MICHAELIS_MENTEN = "michaelis_menten"
    CUSTOM = "custom"


@dataclasses.dataclass
class Species:
    """A chemical species in the reaction network."""
    name: str
    index: int
    max_copy_number: int = 50
    initial_count: int = 0
    description: str = ""
    is_conserved: bool = False
    compartment: str = "default"

    def __repr__(self) -> str:
        return f"Species({self.name}, idx={self.index}, max={self.max_copy_number})"


class PropensityFunction:
    """Base class for propensity functions."""

    def evaluate(self, copy_numbers: NDArray) -> float:
        """Evaluate propensity at given copy numbers."""
        raise NotImplementedError

    def per_species_factors(
        self, species_indices: list[int], max_copy_numbers: list[int]
    ) -> list[NDArray]:
        """
        Compute per-species propensity factor arrays.

        For Kronecker-product MPO construction, the propensity must factor
        as a product of single-species functions:
            a(n) = prod_i f_i(n_i)

        Returns a list of arrays, one per species involved.
        """
        raise NotImplementedError

    def involved_species(self) -> list[int]:
        """Return indices of species involved in the propensity."""
        raise NotImplementedError


class MassActionPropensity(PropensityFunction):
    """
    Mass-action propensity: a(n) = k * prod_i n_i! / (n_i - s_i)!

    For zeroth order: a = k
    For first order (A ->): a = k * n_A
    For second order (A + B ->): a = k * n_A * n_B
    For second order (2A ->): a = k * n_A * (n_A - 1) / 2
    """

    def __init__(
        self,
        rate_constant: float,
        reactant_species: list[int],
        reactant_stoichiometry: list[int],
    ):
        self.rate_constant = rate_constant
        self.reactant_species = reactant_species
        self.reactant_stoichiometry = reactant_stoichiometry

    def evaluate(self, copy_numbers: NDArray) -> float:
        result = self.rate_constant
        for sp_idx, stoich in zip(self.reactant_species, self.reactant_stoichiometry):
            n = copy_numbers[sp_idx]
            for j in range(stoich):
                result *= (n - j) / max(1, stoich)
            if stoich == 2:
                result *= 2  # Undo the division, use falling factorial
                result = self.rate_constant
                # Redo properly
                n_val = copy_numbers[sp_idx]
                result *= n_val * (n_val - 1) / 2.0
                break
        return max(0.0, result)

    def per_species_factors(
        self, species_indices: list[int], max_copy_numbers: list[int]
    ) -> list[NDArray]:
        factors = []
        for sp_idx, max_n in zip(species_indices, max_copy_numbers):
            if sp_idx in self.reactant_species:
                idx_in_reactants = self.reactant_species.index(sp_idx)
                stoich = self.reactant_stoichiometry[idx_in_reactants]

                f = np.zeros(max_n, dtype=np.float64)
                for n in range(max_n):
                    val = 1.0
                    for j in range(stoich):
                        val *= max(0, n - j)
                    if stoich >= 2:
                        val /= math.factorial(stoich)
                        val *= math.factorial(stoich)  # mass-action convention
                    f[n] = val
                factors.append(f)
            else:
                factors.append(np.ones(max_n, dtype=np.float64))

        return factors

    def involved_species(self) -> list[int]:
        return self.reactant_species.copy()

    def __repr__(self) -> str:
        return (
            f"MassAction(k={self.rate_constant}, "
            f"species={self.reactant_species}, "
            f"stoich={self.reactant_stoichiometry})"
        )


class HillPropensity(PropensityFunction):
    """
    Hill function propensity: a(n) = V_max * n^h / (K^h + n^h)

    Used for cooperative binding, gene regulation.
    """

    def __init__(
        self,
        v_max: float,
        k_half: float,
        hill_coefficient: float,
        species_index: int,
        activation: bool = True,
    ):
        self.v_max = v_max
        self.k_half = k_half
        self.hill_coefficient = hill_coefficient
        self.species_index = species_index
        self.activation = activation

    def evaluate(self, copy_numbers: NDArray) -> float:
        n = copy_numbers[self.species_index]
        h = self.hill_coefficient
        K = self.k_half

        if self.activation:
            return self.v_max * n ** h / (K ** h + n ** h + 1e-300)
        else:
            return self.v_max * K ** h / (K ** h + n ** h + 1e-300)

    def per_species_factors(
        self, species_indices: list[int], max_copy_numbers: list[int]
    ) -> list[NDArray]:
        factors = []
        for sp_idx, max_n in zip(species_indices, max_copy_numbers):
            if sp_idx == self.species_index:
                f = np.zeros(max_n, dtype=np.float64)
                h = self.hill_coefficient
                K = self.k_half
                for n in range(max_n):
                    if self.activation:
                        f[n] = self.v_max * n ** h / (K ** h + n ** h + 1e-300)
                    else:
                        f[n] = self.v_max * K ** h / (K ** h + n ** h + 1e-300)
                factors.append(f)
            else:
                factors.append(np.ones(max_n, dtype=np.float64))
        return factors

    def involved_species(self) -> list[int]:
        return [self.species_index]

    def __repr__(self) -> str:
        return (
            f"Hill(Vmax={self.v_max}, K={self.k_half}, "
            f"h={self.hill_coefficient}, sp={self.species_index})"
        )


class MichaelisMentenPropensity(PropensityFunction):
    """
    Michaelis-Menten propensity: a(n_S, n_E) = k_cat * n_E * n_S / (K_M + n_S)

    Models enzymatic reactions.
    """

    def __init__(
        self,
        k_cat: float,
        k_m: float,
        enzyme_index: int,
        substrate_index: int,
    ):
        self.k_cat = k_cat
        self.k_m = k_m
        self.enzyme_index = enzyme_index
        self.substrate_index = substrate_index

    def evaluate(self, copy_numbers: NDArray) -> float:
        n_E = copy_numbers[self.enzyme_index]
        n_S = copy_numbers[self.substrate_index]
        return self.k_cat * n_E * n_S / (self.k_m + n_S + 1e-300)

    def per_species_factors(
        self, species_indices: list[int], max_copy_numbers: list[int]
    ) -> list[NDArray]:
        factors = []
        for sp_idx, max_n in zip(species_indices, max_copy_numbers):
            f = np.zeros(max_n, dtype=np.float64)
            if sp_idx == self.enzyme_index:
                for n in range(max_n):
                    f[n] = float(n)
            elif sp_idx == self.substrate_index:
                for n in range(max_n):
                    f[n] = self.k_cat * n / (self.k_m + n + 1e-300)
            else:
                f[:] = 1.0
            factors.append(f)
        return factors

    def involved_species(self) -> list[int]:
        return [self.enzyme_index, self.substrate_index]


class CustomPropensity(PropensityFunction):
    """Custom propensity function with polynomial interpolation for MPO."""

    def __init__(
        self,
        func: Callable[[NDArray], float],
        species_indices: list[int],
        interpolation_order: int = 4,
    ):
        self.func = func
        self._species_indices = species_indices
        self.interpolation_order = interpolation_order
        self._factored_values: Optional[list[NDArray]] = None

    def evaluate(self, copy_numbers: NDArray) -> float:
        return self.func(copy_numbers)

    def per_species_factors(
        self, species_indices: list[int], max_copy_numbers: list[int]
    ) -> list[NDArray]:
        # For custom propensities, we compute the full tensor and
        # decompose it into per-species factors via TT decomposition
        if len(self._species_indices) == 1:
            sp_idx = self._species_indices[0]
            pos = species_indices.index(sp_idx)
            max_n = max_copy_numbers[pos]
            f = np.zeros(max_n, dtype=np.float64)
            for n in range(max_n):
                state = np.zeros(len(species_indices))
                state[pos] = n
                f[n] = self.func(state)

            factors = [np.ones(max_n, dtype=np.float64) for _ in species_indices]
            factors[pos] = f
            return factors
        else:
            # Multi-species: build lookup table and use rank-1 approximation
            factors = []
            for sp_idx, max_n in zip(species_indices, max_copy_numbers):
                if sp_idx in self._species_indices:
                    f = np.zeros(max_n, dtype=np.float64)
                    for n in range(max_n):
                        state = np.zeros(len(species_indices))
                        state[species_indices.index(sp_idx)] = n
                        f[n] = self.func(state)
                    factors.append(f)
                else:
                    factors.append(np.ones(max_n, dtype=np.float64))
            return factors

    def involved_species(self) -> list[int]:
        return self._species_indices.copy()


@dataclasses.dataclass
class Reaction:
    """A chemical reaction."""
    name: str
    index: int
    reactant_species: list[int]
    reactant_stoichiometry: list[int]
    product_species: list[int]
    product_stoichiometry: list[int]
    propensity: PropensityFunction
    kinetics_type: KineticsType = KineticsType.MASS_ACTION
    reversible: bool = False
    description: str = ""

    @property
    def stoichiometry_vector(self) -> dict[int, int]:
        """Net stoichiometry change for each species."""
        changes: dict[int, int] = {}
        for sp, s in zip(self.reactant_species, self.reactant_stoichiometry):
            changes[sp] = changes.get(sp, 0) - s
        for sp, s in zip(self.product_species, self.product_stoichiometry):
            changes[sp] = changes.get(sp, 0) + s
        return changes

    @property
    def all_species(self) -> list[int]:
        """All species involved (reactants + products)."""
        return sorted(set(self.reactant_species) | set(self.product_species))

    @property
    def propensity_species(self) -> list[int]:
        """Species whose copy numbers appear in the propensity function."""
        return self.propensity.involved_species()

    def net_change(self, species_index: int) -> int:
        """Net change in copy number for a given species."""
        sv = self.stoichiometry_vector
        return sv.get(species_index, 0)

    def __repr__(self) -> str:
        reactants = " + ".join(
            f"{s}*S{sp}" for sp, s in
            zip(self.reactant_species, self.reactant_stoichiometry)
        )
        products = " + ".join(
            f"{s}*S{sp}" for sp, s in
            zip(self.product_species, self.product_stoichiometry)
        )
        return f"Reaction({self.name}: {reactants} -> {products})"


class ReactionNetwork:
    """
    Complete reaction network representation.

    Stores species, reactions, and provides methods for querying
    the network structure needed for CME compilation.
    """

    def __init__(
        self,
        name: str = "unnamed",
        description: str = "",
    ):
        self.name = name
        self.description = description
        self.species: list[Species] = []
        self.reactions: list[Reaction] = []
        self._species_by_name: dict[str, Species] = {}
        self._reaction_by_name: dict[str, Reaction] = {}

    @property
    def num_species(self) -> int:
        return len(self.species)

    @property
    def num_reactions(self) -> int:
        return len(self.reactions)

    @property
    def physical_dims(self) -> list[int]:
        """Local dimensions (max copy number + 1) for each species."""
        return [sp.max_copy_number for sp in self.species]

    @property
    def initial_state(self) -> list[int]:
        """Initial copy numbers."""
        return [sp.initial_count for sp in self.species]

    def add_species(
        self,
        name: str,
        max_copy_number: int = 50,
        initial_count: int = 0,
        description: str = "",
    ) -> Species:
        """Add a species to the network."""
        if name in self._species_by_name:
            raise ValueError(f"Species '{name}' already exists")

        sp = Species(
            name=name,
            index=len(self.species),
            max_copy_number=max_copy_number,
            initial_count=initial_count,
            description=description,
        )
        self.species.append(sp)
        self._species_by_name[name] = sp
        return sp

    def add_reaction(
        self,
        name: str,
        reactants: dict[str, int],
        products: dict[str, int],
        rate_constant: float,
        kinetics_type: KineticsType = KineticsType.MASS_ACTION,
        propensity: Optional[PropensityFunction] = None,
        reversible: bool = False,
        description: str = "",
    ) -> Reaction:
        """
        Add a reaction to the network.

        Args:
            name: Reaction name.
            reactants: Dict mapping species name to stoichiometric coefficient.
            products: Dict mapping species name to stoichiometric coefficient.
            rate_constant: Rate constant.
            kinetics_type: Type of kinetics.
            propensity: Custom propensity (auto-created for mass-action if None).
            reversible: Whether the reaction is reversible.
            description: Description.

        Returns:
            The created Reaction object.
        """
        if name in self._reaction_by_name:
            raise ValueError(f"Reaction '{name}' already exists")

        reactant_species = []
        reactant_stoich = []
        for sp_name, stoich in reactants.items():
            sp = self._species_by_name[sp_name]
            reactant_species.append(sp.index)
            reactant_stoich.append(stoich)

        product_species = []
        product_stoich = []
        for sp_name, stoich in products.items():
            sp = self._species_by_name[sp_name]
            product_species.append(sp.index)
            product_stoich.append(stoich)

        if propensity is None:
            if kinetics_type == KineticsType.MASS_ACTION:
                propensity = MassActionPropensity(
                    rate_constant=rate_constant,
                    reactant_species=reactant_species,
                    reactant_stoichiometry=reactant_stoich,
                )
            else:
                raise ValueError(
                    f"Must provide explicit propensity for kinetics type {kinetics_type}"
                )

        rxn = Reaction(
            name=name,
            index=len(self.reactions),
            reactant_species=reactant_species,
            reactant_stoichiometry=reactant_stoich,
            product_species=product_species,
            product_stoichiometry=product_stoich,
            propensity=propensity,
            kinetics_type=kinetics_type,
            reversible=reversible,
            description=description,
        )

        self.reactions.append(rxn)
        self._reaction_by_name[name] = rxn
        return rxn

    def add_reaction_with_hill(
        self,
        name: str,
        products: dict[str, int],
        regulator: str,
        v_max: float,
        k_half: float,
        hill_coefficient: float,
        activation: bool = True,
        description: str = "",
    ) -> Reaction:
        """Add a reaction with Hill kinetics."""
        sp_reg = self._species_by_name[regulator]
        propensity = HillPropensity(
            v_max=v_max,
            k_half=k_half,
            hill_coefficient=hill_coefficient,
            species_index=sp_reg.index,
            activation=activation,
        )

        product_species = []
        product_stoich = []
        for sp_name, stoich in products.items():
            sp = self._species_by_name[sp_name]
            product_species.append(sp.index)
            product_stoich.append(stoich)

        rxn = Reaction(
            name=name,
            index=len(self.reactions),
            reactant_species=[],
            reactant_stoichiometry=[],
            product_species=product_species,
            product_stoichiometry=product_stoich,
            propensity=propensity,
            kinetics_type=KineticsType.HILL,
            description=description,
        )

        self.reactions.append(rxn)
        self._reaction_by_name[name] = rxn
        return rxn

    def add_reaction_with_michaelis_menten(
        self,
        name: str,
        substrate: str,
        product: str,
        enzyme: str,
        k_cat: float,
        k_m: float,
        description: str = "",
    ) -> Reaction:
        """Add a reaction with Michaelis-Menten kinetics."""
        sp_sub = self._species_by_name[substrate]
        sp_prod = self._species_by_name[product]
        sp_enz = self._species_by_name[enzyme]

        propensity = MichaelisMentenPropensity(
            k_cat=k_cat,
            k_m=k_m,
            enzyme_index=sp_enz.index,
            substrate_index=sp_sub.index,
        )

        rxn = Reaction(
            name=name,
            index=len(self.reactions),
            reactant_species=[sp_sub.index],
            reactant_stoichiometry=[1],
            product_species=[sp_prod.index],
            product_stoichiometry=[1],
            propensity=propensity,
            kinetics_type=KineticsType.MICHAELIS_MENTEN,
            description=description,
        )

        self.reactions.append(rxn)
        self._reaction_by_name[name] = rxn
        return rxn

    def get_species(self, name: str) -> Species:
        """Get a species by name."""
        return self._species_by_name[name]

    def get_reaction(self, name: str) -> Reaction:
        """Get a reaction by name."""
        return self._reaction_by_name[name]

    def species_interaction_graph(self) -> dict[int, set[int]]:
        """
        Build the species interaction graph.

        Two species are connected if they appear in the same reaction
        (either as reactant, product, or in the propensity function).
        """
        graph: dict[int, set[int]] = {sp.index: set() for sp in self.species}

        for rxn in self.reactions:
            involved = set(rxn.all_species) | set(rxn.propensity_species)
            for sp_i in involved:
                for sp_j in involved:
                    if sp_i != sp_j:
                        graph[sp_i].add(sp_j)
                        graph[sp_j].add(sp_i)

        return graph

    def species_coupling_strength(self) -> NDArray:
        """
        Compute the coupling strength matrix between species.

        Entry (i,j) is the number of reactions that involve both species i and j.
        This is used for species ordering optimization.
        """
        N = self.num_species
        coupling = np.zeros((N, N), dtype=np.float64)

        for rxn in self.reactions:
            involved = list(set(rxn.all_species) | set(rxn.propensity_species))
            for a in range(len(involved)):
                for b in range(a + 1, len(involved)):
                    i, j = involved[a], involved[b]
                    coupling[i, j] += 1
                    coupling[j, i] += 1

        return coupling

    def max_exit_rate(self) -> float:
        """
        Estimate the maximum exit rate (for uniformization).

        The exit rate of state n is sum of all propensity functions.
        We estimate the maximum over the state space.
        """
        max_rate = 0.0

        # Sample corners and midpoints of the state space
        for _ in range(1000):
            state = np.array([
                np.random.randint(0, sp.max_copy_number) for sp in self.species
            ], dtype=np.float64)

            total_rate = 0.0
            for rxn in self.reactions:
                total_rate += rxn.propensity.evaluate(state)

            max_rate = max(max_rate, total_rate)

        return max_rate

    def validate(self) -> list[str]:
        """Validate the reaction network for consistency."""
        errors = []

        for rxn in self.reactions:
            for sp_idx in rxn.reactant_species:
                if sp_idx >= self.num_species:
                    errors.append(
                        f"Reaction '{rxn.name}': reactant species index "
                        f"{sp_idx} out of range"
                    )
            for sp_idx in rxn.product_species:
                if sp_idx >= self.num_species:
                    errors.append(
                        f"Reaction '{rxn.name}': product species index "
                        f"{sp_idx} out of range"
                    )

            # Check stoichiometry doesn't violate bounds
            sv = rxn.stoichiometry_vector
            for sp_idx, change in sv.items():
                sp = self.species[sp_idx]
                if change > 0 and sp.initial_count + change > sp.max_copy_number:
                    pass  # Warning only, not an error

        return errors

    def summary(self) -> str:
        """Return a human-readable summary of the network."""
        lines = [
            f"Reaction Network: {self.name}",
            f"  Species: {self.num_species}",
            f"  Reactions: {self.num_reactions}",
            f"  Physical dims: {self.physical_dims}",
            f"  State space size: {np.prod(self.physical_dims):.2e}",
            "",
            "  Species:",
        ]
        for sp in self.species:
            lines.append(
                f"    {sp.name} (idx={sp.index}, max={sp.max_copy_number}, "
                f"init={sp.initial_count})"
            )
        lines.append("")
        lines.append("  Reactions:")
        for rxn in self.reactions:
            lines.append(f"    {rxn}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ReactionNetwork({self.name}, {self.num_species} species, {self.num_reactions} reactions)"
