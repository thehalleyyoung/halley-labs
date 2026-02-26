"""
Library of standard stochastic reaction network models.

Each model builder returns a ReactionNetwork that can be compiled
to an MPO generator via the CME compiler.
"""

from __future__ import annotations

from tn_check.cme.reaction_network import (
    Species, Reaction, ReactionNetwork,
    MassActionPropensity, HillPropensity, PropensityFunction,
    KineticsType,
)


def _build_network(name: str, species_list, reactions_list) -> ReactionNetwork:
    """Helper to build a ReactionNetwork from species and reaction defs."""
    net = ReactionNetwork(name=name)
    for sp in species_list:
        net.species.append(sp)
        net._species_by_name[sp.name] = sp
    for rxn in reactions_list:
        net.reactions.append(rxn)
        net._reaction_by_name[rxn.name] = rxn
    return net


def birth_death(
    birth_rate: float = 1.0,
    death_rate: float = 0.1,
    max_copy: int = 50,
) -> ReactionNetwork:
    """
    Simple birth-death process: ∅ → X, X → ∅.

    Stationary distribution is Poisson(birth_rate / death_rate).
    """
    species = [Species(name="X", index=0, max_copy_number=max_copy)]
    reactions = [
        Reaction(
            name="birth",
            index=0,
            reactant_species=[],
            reactant_stoichiometry=[],
            product_species=[0],
            product_stoichiometry=[1],
            propensity=MassActionPropensity(rate_constant=birth_rate, reactant_species=[], reactant_stoichiometry=[]),
        ),
        Reaction(
            name="death",
            index=1,
            reactant_species=[0],
            reactant_stoichiometry=[1],
            product_species=[],
            product_stoichiometry=[],
            propensity=MassActionPropensity(rate_constant=death_rate, reactant_species=[0], reactant_stoichiometry=[1]),
        ),
    ]
    return _build_network("birth_death", species, reactions)


def toggle_switch(
    alpha1: float = 50.0,
    alpha2: float = 50.0,
    beta: float = 2.5,
    gamma: float = 1.0,
    max_copy: int = 100,
) -> ReactionNetwork:
    """
    Gardner toggle switch (2 species, bistable).

    Two mutually repressing genes:
    - ∅ →[α₁/(1+X₂^β)] X₁  (production of X₁, repressed by X₂)
    - X₁ →[γ] ∅              (degradation of X₁)
    - ∅ →[α₂/(1+X₁^β)] X₂  (production of X₂, repressed by X₁)
    - X₂ →[γ] ∅              (degradation of X₂)
    """
    species = [
        Species(name="X1", index=0, max_copy_number=max_copy),
        Species(name="X2", index=1, max_copy_number=max_copy),
    ]
    reactions = [
        Reaction(
            name="prod_X1",
            index=0,
            reactant_species=[],
            reactant_stoichiometry=[],
            product_species=[0],
            product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=alpha1, k_half=1.0, hill_coefficient=beta,
                species_index=1, activation=False,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        Reaction(
            name="deg_X1",
            index=1,
            reactant_species=[0],
            reactant_stoichiometry=[1],
            product_species=[],
            product_stoichiometry=[],
            propensity=MassActionPropensity(rate_constant=gamma, reactant_species=[0], reactant_stoichiometry=[1]),
        ),
        Reaction(
            name="prod_X2",
            index=2,
            reactant_species=[],
            reactant_stoichiometry=[],
            product_species=[1],
            product_stoichiometry=[1],
            propensity=HillPropensity(
                v_max=alpha2, k_half=1.0, hill_coefficient=beta,
                species_index=0, activation=False,
            ),
            kinetics_type=KineticsType.HILL,
        ),
        Reaction(
            name="deg_X2",
            index=3,
            reactant_species=[1],
            reactant_stoichiometry=[1],
            product_species=[],
            product_stoichiometry=[],
            propensity=MassActionPropensity(rate_constant=gamma, reactant_species=[1], reactant_stoichiometry=[1]),
        ),
    ]
    return _build_network("toggle_switch", species, reactions)


def repressilator(
    alpha: float = 50.0,
    alpha0: float = 0.5,
    beta: float = 2.0,
    gamma: float = 1.0,
    n_genes: int = 3,
    max_copy: int = 80,
) -> ReactionNetwork:
    """
    Repressilator (n_genes species, oscillatory).

    Ring of mutually repressing genes:
    Gene i is repressed by gene (i-1) mod n.
    """
    species = [
        Species(name=f"X{i}", index=i, max_copy_number=max_copy)
        for i in range(n_genes)
    ]
    reactions = []
    idx = 0
    for i in range(n_genes):
        repressor = (i - 1) % n_genes
        reactions.append(
            Reaction(
                name=f"prod_X{i}",
                index=idx,
                reactant_species=[],
                reactant_stoichiometry=[],
                product_species=[i],
                product_stoichiometry=[1],
                propensity=HillPropensity(
                    v_max=alpha, k_half=1.0, hill_coefficient=beta,
                    species_index=repressor, activation=False,
                ),
                kinetics_type=KineticsType.HILL,
            )
        )
        idx += 1
        reactions.append(
            Reaction(
                name=f"deg_X{i}",
                index=idx,
                reactant_species=[i],
                reactant_stoichiometry=[1],
                product_species=[],
                product_stoichiometry=[],
                propensity=MassActionPropensity(
                    rate_constant=gamma, reactant_species=[i], reactant_stoichiometry=[1],
                ),
            )
        )
        idx += 1
    return _build_network(f"repressilator_{n_genes}", species, reactions)


def cascade(
    n_layers: int = 3,
    k_activation: float = 1.0,
    k_deactivation: float = 0.1,
    max_copy: int = 50,
) -> ReactionNetwork:
    """
    Linear signaling cascade (n_layers species, modular).

    Layer i is activated by layer i-1 (mass-action kinetics).
    Models MAPK-like cascade structure.
    """
    species = [
        Species(name=f"L{i}", index=i, max_copy_number=max_copy)
        for i in range(n_layers)
    ]
    reactions = []
    idx = 0

    # First layer: constitutive production
    reactions.append(
        Reaction(
            name="input",
            index=idx,
            reactant_species=[],
            reactant_stoichiometry=[],
            product_species=[0],
            product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_activation, reactant_species=[], reactant_stoichiometry=[],
            ),
        )
    )
    idx += 1

    for i in range(1, n_layers):
        reactions.append(
            Reaction(
                name=f"activate_L{i}",
                index=idx,
                reactant_species=[i - 1],
                reactant_stoichiometry=[1],
                product_species=[i - 1, i],
                product_stoichiometry=[1, 1],
                propensity=MassActionPropensity(
                    rate_constant=k_activation, reactant_species=[i - 1], reactant_stoichiometry=[1],
                ),
            )
        )
        idx += 1

    for i in range(n_layers):
        reactions.append(
            Reaction(
                name=f"deactivate_L{i}",
                index=idx,
                reactant_species=[i],
                reactant_stoichiometry=[1],
                product_species=[],
                product_stoichiometry=[],
                propensity=MassActionPropensity(
                    rate_constant=k_deactivation, reactant_species=[i], reactant_stoichiometry=[1],
                ),
            )
        )
        idx += 1

    return _build_network(f"cascade_{n_layers}", species, reactions)


def schlogl(
    c1: float = 3e-7,
    c2: float = 1e-4,
    c3: float = 1e-3,
    c4: float = 3.5,
    max_copy: int = 200,
) -> ReactionNetwork:
    """
    Schlögl model (1 species, bistable).

    Classic bistable model:
    - 2X →[c1] 3X
    - 3X →[c2] 2X
    - ∅  →[c3] X
    - X  →[c4] ∅
    """
    species = [Species(name="X", index=0, max_copy_number=max_copy)]
    reactions = [
        Reaction(
            name="r1",
            index=0,
            reactant_species=[0],
            reactant_stoichiometry=[2],
            product_species=[0],
            product_stoichiometry=[3],
            propensity=MassActionPropensity(rate_constant=c1, reactant_species=[0], reactant_stoichiometry=[2]),
        ),
        Reaction(
            name="r2",
            index=1,
            reactant_species=[0],
            reactant_stoichiometry=[3],
            product_species=[0],
            product_stoichiometry=[2],
            propensity=MassActionPropensity(rate_constant=c2, reactant_species=[0], reactant_stoichiometry=[3]),
        ),
        Reaction(
            name="r3",
            index=2,
            reactant_species=[],
            reactant_stoichiometry=[],
            product_species=[0],
            product_stoichiometry=[1],
            propensity=MassActionPropensity(rate_constant=c3, reactant_species=[], reactant_stoichiometry=[]),
        ),
        Reaction(
            name="r4",
            index=3,
            reactant_species=[0],
            reactant_stoichiometry=[1],
            product_species=[],
            product_stoichiometry=[],
            propensity=MassActionPropensity(rate_constant=c4, reactant_species=[0], reactant_stoichiometry=[1]),
        ),
    ]
    return _build_network("schlogl", species, reactions)
