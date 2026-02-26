"""Parameterised model-template generators for BioProver.

Provides scalable, parameterised circuit templates for benchmarking and
systematic exploration of circuit design spaces.  Templates generate
:class:`~bioprover.models.bio_model.BioModel` instances with Hill kinetics
and realistic parameter ranges.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bioprover.models.bio_model import BioModel
from bioprover.models.species import Species, SpeciesType
from bioprover.models.reactions import (
    ConstitutiveProduction,
    HillActivation,
    HillRepression,
    LinearDegradation,
    Reaction,
    StoichiometryEntry,
)


# ---------------------------------------------------------------------------
# Realistic parameter ranges (from literature)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParameterRange:
    """Numeric range for randomised parameter generation.

    Attributes:
        low:     Lower bound (inclusive).
        high:    Upper bound (inclusive).
        log_scale: If ``True``, sample uniformly on a log scale.
    """

    low: float
    high: float
    log_scale: bool = False

    def sample(self, rng: Optional[random.Random] = None) -> float:
        """Draw a single random sample from the range."""
        r = rng if rng else random
        if self.log_scale:
            return math.exp(r.uniform(math.log(self.low), math.log(self.high)))
        return r.uniform(self.low, self.high)


# Default realistic ranges for E. coli at 37°C
REALISTIC_RANGES: Dict[str, ParameterRange] = {
    "Vmax": ParameterRange(0.5, 20.0, log_scale=True),
    "K": ParameterRange(1.0, 50.0, log_scale=True),
    "n": ParameterRange(1.0, 4.0),
    "gamma": ParameterRange(0.01, 0.2, log_scale=True),
    "basal": ParameterRange(0.01, 1.0, log_scale=True),
    "initial_conc": ParameterRange(0.0, 10.0),
}


# ---------------------------------------------------------------------------
# TemplateGenerator
# ---------------------------------------------------------------------------

class TemplateGenerator:
    """Factory for parameterised circuit model templates.

    All generated models use Hill kinetics and carry parameters within
    biologically realistic ranges suitable for *E. coli* at 37 °C unless
    overridden.

    Parameters
    ----------
    seed:
        Random seed for reproducible parameter randomisation.
    ranges:
        Optional dict overriding default parameter ranges.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        ranges: Optional[Dict[str, ParameterRange]] = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._ranges: Dict[str, ParameterRange] = {
            **REALISTIC_RANGES, **(ranges or {})
        }

    # -- helpers ------------------------------------------------------------

    def _sample(self, param: str) -> float:
        """Sample a parameter from the configured range."""
        return self._ranges[param].sample(self._rng)

    def _add_hill_repression(
        self,
        model: BioModel,
        product: str,
        repressor: str,
        prefix: str,
        vmax: Optional[float] = None,
        K: Optional[float] = None,
        n: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> None:
        """Add Hill-repression production + linear degradation."""
        vmax = vmax if vmax is not None else self._sample("Vmax")
        K = K if K is not None else self._sample("K")
        n = n if n is not None else self._sample("n")
        gamma = gamma if gamma is not None else self._sample("gamma")

        hill = HillRepression(Vmax=vmax, K=K, n=n)
        hill.repressor_name = repressor
        model.add_reaction(Reaction(
            name=f"{prefix}_prod",
            reactants=[],
            products=[StoichiometryEntry(product)],
            kinetic_law=hill,
            modifiers=[repressor],
        ))
        deg = LinearDegradation(rate=gamma)
        deg.species_name = product
        model.add_reaction(Reaction(
            name=f"{prefix}_deg",
            reactants=[StoichiometryEntry(product)],
            products=[],
            kinetic_law=deg,
        ))

    def _add_hill_activation(
        self,
        model: BioModel,
        product: str,
        activator: str,
        prefix: str,
        vmax: Optional[float] = None,
        K: Optional[float] = None,
        n: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> None:
        """Add Hill-activation production + linear degradation."""
        vmax = vmax if vmax is not None else self._sample("Vmax")
        K = K if K is not None else self._sample("K")
        n = n if n is not None else self._sample("n")
        gamma = gamma if gamma is not None else self._sample("gamma")

        hill = HillActivation(Vmax=vmax, K=K, n=n)
        hill.activator_name = activator
        model.add_reaction(Reaction(
            name=f"{prefix}_prod",
            reactants=[],
            products=[StoichiometryEntry(product)],
            kinetic_law=hill,
            modifiers=[activator],
        ))
        deg = LinearDegradation(rate=gamma)
        deg.species_name = product
        model.add_reaction(Reaction(
            name=f"{prefix}_deg",
            reactants=[StoichiometryEntry(product)],
            products=[],
            kinetic_law=deg,
        ))

    # ====================================================================== #
    #                      Template generators                                #
    # ====================================================================== #

    def n_stage_cascade(
        self,
        n_stages: int = 3,
        *,
        activation: bool = True,
        vmax: Optional[float] = None,
        K: Optional[float] = None,
        hill_n: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> BioModel:
        """Generate an N-stage signalling cascade with Hill kinetics.

        Parameters
        ----------
        n_stages:
            Number of cascade stages (≥ 2).
        activation:
            ``True`` for activation cascade, ``False`` for alternating
            repression cascade.
        vmax, K, hill_n, gamma:
            If supplied, override the random sampling for all stages.

        Returns
        -------
        BioModel
            A model named ``cascade_N`` with species ``S0 .. S{N-1}``.
        """
        if n_stages < 2:
            raise ValueError("Cascade requires at least 2 stages")

        m = BioModel(f"cascade_{n_stages}")
        names = [f"S{i}" for i in range(n_stages)]
        for nm in names:
            ic = self._sample("initial_conc") if nm != names[0] else 5.0
            m.add_species(Species(nm, initial_concentration=ic))

        # Constitutive input S0
        g0 = gamma if gamma is not None else self._sample("gamma")
        m.add_reaction(Reaction(
            name="S0_prod", reactants=[], products=[StoichiometryEntry("S0")],
            kinetic_law=ConstitutiveProduction(rate=5.0 * g0),
        ))
        deg0 = LinearDegradation(rate=g0)
        deg0.species_name = "S0"
        m.add_reaction(Reaction(
            name="S0_deg", reactants=[StoichiometryEntry("S0")], products=[],
            kinetic_law=deg0,
        ))

        for i in range(1, n_stages):
            if activation:
                self._add_hill_activation(
                    m, names[i], names[i - 1], names[i],
                    vmax=vmax, K=K, n=hill_n, gamma=gamma,
                )
            else:
                # Alternate repression / activation at each stage
                if i % 2 == 1:
                    self._add_hill_repression(
                        m, names[i], names[i - 1], names[i],
                        vmax=vmax, K=K, n=hill_n, gamma=gamma,
                    )
                else:
                    self._add_hill_activation(
                        m, names[i], names[i - 1], names[i],
                        vmax=vmax, K=K, n=hill_n, gamma=gamma,
                    )
        return m

    def n_node_repressilator(
        self,
        n_nodes: int = 3,
        *,
        vmax: Optional[float] = None,
        K: Optional[float] = None,
        hill_n: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> BioModel:
        """Generate an N-node repressilator ring oscillator.

        Parameters
        ----------
        n_nodes:
            Number of genes in the ring (must be odd and ≥ 3).

        Returns
        -------
        BioModel
            Model with species ``G0 .. G{N-1}`` in a cyclic repression ring.
        """
        if n_nodes < 3:
            raise ValueError("Repressilator ring requires at least 3 nodes")

        m = BioModel(f"repressilator_{n_nodes}")
        names = [f"G{i}" for i in range(n_nodes)]
        for nm in names:
            m.add_species(Species(nm, initial_concentration=self._sample("initial_conc")))

        for i in range(n_nodes):
            repressor = names[(i - 1) % n_nodes]
            product = names[i]
            self._add_hill_repression(
                m, product, repressor, product,
                vmax=vmax, K=K, n=hill_n, gamma=gamma,
            )
        return m

    def n_toggle_switch(
        self,
        n_genes: int = 2,
        *,
        vmax: Optional[float] = None,
        K: Optional[float] = None,
        hill_n: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> BioModel:
        """Generate a toggle switch with *n_genes* mutual repressors.

        For n_genes=2 this is the classic Gardner toggle switch.  For n>2,
        each gene represses all other genes.

        Returns
        -------
        BioModel
            Model with species ``T0 .. T{N-1}``.
        """
        if n_genes < 2:
            raise ValueError("Toggle switch requires at least 2 genes")

        m = BioModel(f"toggle_switch_{n_genes}")
        names = [f"T{i}" for i in range(n_genes)]
        for i, nm in enumerate(names):
            ic = 10.0 if i == 0 else 0.1
            m.add_species(Species(nm, initial_concentration=ic))

        for i in range(n_genes):
            for j in range(n_genes):
                if i == j:
                    continue
                self._add_hill_repression(
                    m, names[i], names[j], f"{names[i]}_rep_{names[j]}",
                    vmax=vmax, K=K, n=hill_n, gamma=gamma,
                )
        return m

    def fan_out_tree(
        self,
        depth: int = 3,
        branching: int = 2,
        *,
        vmax: Optional[float] = None,
        K: Optional[float] = None,
        hill_n: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> BioModel:
        """Generate a fan-out tree of *depth* levels with *branching* factor.

        The root node is a constitutive input.  Each interior node
        activates *branching* children.

        Returns
        -------
        BioModel
            Model with ``(branching^depth - 1) / (branching - 1)`` species.
        """
        if depth < 1 or branching < 1:
            raise ValueError("depth and branching must be >= 1")

        m = BioModel(f"fanout_d{depth}_b{branching}")
        node_idx = 0
        levels: List[List[str]] = []

        for d in range(depth):
            level_names: List[str] = []
            n_nodes = branching ** d
            for _ in range(n_nodes):
                name = f"N{node_idx}"
                m.add_species(Species(name, initial_concentration=0.0))
                level_names.append(name)
                node_idx += 1
            levels.append(level_names)

        # Root constitutive production
        root = levels[0][0]
        g0 = gamma if gamma is not None else self._sample("gamma")
        m.add_reaction(Reaction(
            name=f"{root}_prod", reactants=[],
            products=[StoichiometryEntry(root)],
            kinetic_law=ConstitutiveProduction(rate=5.0 * g0),
        ))
        deg0 = LinearDegradation(rate=g0)
        deg0.species_name = root
        m.add_reaction(Reaction(
            name=f"{root}_deg", reactants=[StoichiometryEntry(root)],
            products=[], kinetic_law=deg0,
        ))

        for d in range(1, depth):
            for i, child in enumerate(levels[d]):
                parent = levels[d - 1][i // branching]
                self._add_hill_activation(
                    m, child, parent, child,
                    vmax=vmax, K=K, n=hill_n, gamma=gamma,
                )
        return m

    def random_grn(
        self,
        n_genes: int = 5,
        connectivity: float = 0.3,
        *,
        fraction_repression: float = 0.5,
        vmax: Optional[float] = None,
        K: Optional[float] = None,
        hill_n: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> BioModel:
        """Generate a random gene-regulatory network.

        Parameters
        ----------
        n_genes:
            Number of genes.
        connectivity:
            Probability of an edge between any pair [0, 1].
        fraction_repression:
            Fraction of edges that are repressive (remainder are activating).

        Returns
        -------
        BioModel
            Random GRN model with species ``R0 .. R{N-1}``.
        """
        m = BioModel(f"random_grn_{n_genes}_c{connectivity:.2f}")
        names = [f"R{i}" for i in range(n_genes)]
        for nm in names:
            m.add_species(Species(nm, initial_concentration=self._sample("initial_conc")))

        edge_count = 0
        for i in range(n_genes):
            for j in range(n_genes):
                if i == j:
                    continue
                if self._rng.random() < connectivity:
                    is_rep = self._rng.random() < fraction_repression
                    prefix = f"{names[j]}_from_{names[i]}"
                    if is_rep:
                        self._add_hill_repression(
                            m, names[j], names[i], prefix,
                            vmax=vmax, K=K, n=hill_n, gamma=gamma,
                        )
                    else:
                        self._add_hill_activation(
                            m, names[j], names[i], prefix,
                            vmax=vmax, K=K, n=hill_n, gamma=gamma,
                        )
                    edge_count += 1

        # Ensure at least one constitutive input if no self-production exists
        if edge_count == 0:
            g = gamma if gamma is not None else self._sample("gamma")
            m.add_reaction(Reaction(
                name="R0_const", reactants=[],
                products=[StoichiometryEntry("R0")],
                kinetic_law=ConstitutiveProduction(rate=5.0 * g),
            ))
            deg = LinearDegradation(rate=g)
            deg.species_name = "R0"
            m.add_reaction(Reaction(
                name="R0_deg", reactants=[StoichiometryEntry("R0")],
                products=[], kinetic_law=deg,
            ))
        return m

    # -- summary / documentation --------------------------------------------

    @staticmethod
    def available_templates() -> List[Dict[str, str]]:
        """Return documentation for each available template type."""
        return [
            {
                "name": "n_stage_cascade",
                "description": "Linear N-stage signalling cascade with Hill "
                               "kinetics.  Models signal propagation delay.",
                "parameters": "n_stages, activation, vmax, K, hill_n, gamma",
            },
            {
                "name": "n_node_repressilator",
                "description": "N-node cyclic repression ring oscillator.  "
                               "Generalisation of the 3-gene repressilator.",
                "parameters": "n_nodes, vmax, K, hill_n, gamma",
            },
            {
                "name": "n_toggle_switch",
                "description": "Toggle switch with N mutual repressors.  "
                               "For N=2, the classic Gardner toggle.",
                "parameters": "n_genes, vmax, K, hill_n, gamma",
            },
            {
                "name": "fan_out_tree",
                "description": "Fan-out activation tree of depth D and "
                               "branching factor B.",
                "parameters": "depth, branching, vmax, K, hill_n, gamma",
            },
            {
                "name": "random_grn",
                "description": "Random gene-regulatory network with specified "
                               "connectivity and repression fraction.",
                "parameters": "n_genes, connectivity, fraction_repression, "
                              "vmax, K, hill_n, gamma",
            },
        ]
