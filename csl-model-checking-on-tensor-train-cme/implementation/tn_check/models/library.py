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


def gene_expression(
    k_txn: float = 0.5,
    k_tln: float = 5.0,
    gamma_m: float = 0.1,
    gamma_p: float = 0.005,
    max_copy_mRNA: int = 40,
    max_copy_protein: int = 200,
) -> ReactionNetwork:
    """
    Simple 2-species gene expression: mRNA → Protein.

    Reactions (Thattai & van Oudenaarden, 2001):
    - ∅ →[k_txn] mRNA           (transcription)
    - mRNA →[k_tln] mRNA + P    (translation, catalytic)
    - mRNA →[γ_m] ∅             (mRNA degradation)
    - P →[γ_p] ∅                (protein degradation)

    Steady state: <mRNA> = k_txn/γ_m, <P> = k_txn*k_tln/(γ_m*γ_p).
    """
    species = [
        Species(name="mRNA", index=0, max_copy_number=max_copy_mRNA),
        Species(name="Protein", index=1, max_copy_number=max_copy_protein),
    ]
    reactions = [
        Reaction(
            name="transcription",
            index=0,
            reactant_species=[],
            reactant_stoichiometry=[],
            product_species=[0],
            product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_txn, reactant_species=[], reactant_stoichiometry=[],
            ),
        ),
        Reaction(
            name="translation",
            index=1,
            reactant_species=[0],
            reactant_stoichiometry=[1],
            product_species=[0, 1],
            product_stoichiometry=[1, 1],
            propensity=MassActionPropensity(
                rate_constant=k_tln, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        Reaction(
            name="mRNA_degradation",
            index=2,
            reactant_species=[0],
            reactant_stoichiometry=[1],
            product_species=[],
            product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=gamma_m, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        Reaction(
            name="protein_degradation",
            index=3,
            reactant_species=[1],
            reactant_stoichiometry=[1],
            product_species=[],
            product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=gamma_p, reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("gene_expression", species, reactions)


def exclusive_switch(
    k_on_a: float = 0.05,
    k_on_b: float = 0.05,
    k_off_a: float = 0.005,
    k_off_b: float = 0.005,
    k_prod_a: float = 5.0,
    k_prod_b: float = 5.0,
    gamma_a: float = 0.1,
    gamma_b: float = 0.1,
    max_copy_dna: int = 3,
    max_copy_protein: int = 80,
) -> ReactionNetwork:
    """
    Exclusive genetic switch (3 species: DNA state, Protein A, Protein B).

    Models promoter binding competition (Loinger et al., 2007).
    DNA has states 0=free, 1=bound-A, 2=bound-B.

    Reactions:
    - DNA=0 + A →[k_on_a] DNA=1       (A binds promoter)
    - DNA=1 →[k_off_a] DNA=0 + A      (A unbinds)
    - DNA=0 + B →[k_on_b] DNA=2       (B binds promoter)
    - DNA=2 →[k_off_b] DNA=0 + B      (B unbinds)
    - DNA=1 →[k_prod_a] DNA=1 + A     (A production when bound)
    - DNA=2 →[k_prod_b] DNA=2 + B     (B production when bound)
    - A →[γ_a] ∅                       (A degradation)
    - B →[γ_b] ∅                       (B degradation)

    Simplified to mass-action on copy numbers: production proportional
    to DNA state indicator (approximated via linear propensity on DNA).
    """
    species = [
        Species(name="DNA", index=0, max_copy_number=max_copy_dna),
        Species(name="ProtA", index=1, max_copy_number=max_copy_protein),
        Species(name="ProtB", index=2, max_copy_number=max_copy_protein),
    ]
    reactions = [
        # A binds free promoter: DNA increases, consumes A
        Reaction(
            name="bind_A",
            index=0,
            reactant_species=[1],
            reactant_stoichiometry=[1],
            product_species=[0],
            product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_on_a, reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # A unbinds: DNA decreases, produces A
        Reaction(
            name="unbind_A",
            index=1,
            reactant_species=[0],
            reactant_stoichiometry=[1],
            product_species=[1],
            product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_off_a, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # B binds free promoter
        Reaction(
            name="bind_B",
            index=2,
            reactant_species=[2],
            reactant_stoichiometry=[1],
            product_species=[0],
            product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_on_b, reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
        # B unbinds
        Reaction(
            name="unbind_B",
            index=3,
            reactant_species=[0],
            reactant_stoichiometry=[1],
            product_species=[2],
            product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_off_b, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Protein A production (proportional to DNA occupancy)
        Reaction(
            name="prod_A",
            index=4,
            reactant_species=[0],
            reactant_stoichiometry=[1],
            product_species=[0, 1],
            product_stoichiometry=[1, 1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_a, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Protein B production (proportional to DNA occupancy)
        Reaction(
            name="prod_B",
            index=5,
            reactant_species=[0],
            reactant_stoichiometry=[1],
            product_species=[0, 2],
            product_stoichiometry=[1, 1],
            propensity=MassActionPropensity(
                rate_constant=k_prod_b, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        ),
        # Protein A degradation
        Reaction(
            name="deg_A",
            index=6,
            reactant_species=[1],
            reactant_stoichiometry=[1],
            product_species=[],
            product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=gamma_a, reactant_species=[1], reactant_stoichiometry=[1],
            ),
        ),
        # Protein B degradation
        Reaction(
            name="deg_B",
            index=7,
            reactant_species=[2],
            reactant_stoichiometry=[1],
            product_species=[],
            product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=gamma_b, reactant_species=[2], reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("exclusive_switch", species, reactions)


def sir_epidemic(
    beta: float = 0.005,
    gamma: float = 0.1,
    max_S: int = 50,
    max_I: int = 50,
    max_R: int = 50,
    S0: int = 40,
    I0: int = 5,
) -> ReactionNetwork:
    """
    Stochastic SIR epidemic model.

    Reactions:
    - S + I →[β] 2I   (infection: S decreases, I increases)
    - I →[γ] R         (recovery)

    Standard epidemiological model in stochastic form.
    """
    species = [
        Species(name="S", index=0, max_copy_number=max_S, initial_count=S0),
        Species(name="I", index=1, max_copy_number=max_I, initial_count=I0),
        Species(name="R", index=2, max_copy_number=max_R, initial_count=0),
    ]
    reactions = [
        # Infection: S + I → 2I  (net: S-1, I+1)
        Reaction(
            name="infection",
            index=0,
            reactant_species=[0, 1],
            reactant_stoichiometry=[1, 1],
            product_species=[1],
            product_stoichiometry=[2],
            propensity=MassActionPropensity(
                rate_constant=beta,
                reactant_species=[0, 1],
                reactant_stoichiometry=[1, 1],
            ),
        ),
        # Recovery: I → R
        Reaction(
            name="recovery",
            index=1,
            reactant_species=[1],
            reactant_stoichiometry=[1],
            product_species=[2],
            product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=gamma,
                reactant_species=[1],
                reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("sir_epidemic", species, reactions)


def michaelis_menten_enzyme(
    k_f: float = 0.01,
    k_r: float = 0.1,
    k_cat: float = 0.5,
    max_E: int = 20,
    max_S: int = 40,
    max_ES: int = 20,
    max_P: int = 40,
    E0: int = 10,
    S0: int = 20,
) -> ReactionNetwork:
    """
    Full Michaelis-Menten enzymatic reaction (4 species).

    Reactions:
    - E + S →[k_f] ES        (binding)
    - ES →[k_r] E + S        (unbinding)
    - ES →[k_cat] E + P      (catalysis)

    Conservation laws: E + ES = E_total, S + ES + P = S_total.
    """
    species = [
        Species(name="E", index=0, max_copy_number=max_E, initial_count=E0),
        Species(name="S", index=1, max_copy_number=max_S, initial_count=S0),
        Species(name="ES", index=2, max_copy_number=max_ES, initial_count=0),
        Species(name="P", index=3, max_copy_number=max_P, initial_count=0),
    ]
    reactions = [
        # E + S → ES
        Reaction(
            name="binding",
            index=0,
            reactant_species=[0, 1],
            reactant_stoichiometry=[1, 1],
            product_species=[2],
            product_stoichiometry=[1],
            propensity=MassActionPropensity(
                rate_constant=k_f,
                reactant_species=[0, 1],
                reactant_stoichiometry=[1, 1],
            ),
        ),
        # ES → E + S
        Reaction(
            name="unbinding",
            index=1,
            reactant_species=[2],
            reactant_stoichiometry=[1],
            product_species=[0, 1],
            product_stoichiometry=[1, 1],
            propensity=MassActionPropensity(
                rate_constant=k_r,
                reactant_species=[2],
                reactant_stoichiometry=[1],
            ),
        ),
        # ES → E + P
        Reaction(
            name="catalysis",
            index=2,
            reactant_species=[2],
            reactant_stoichiometry=[1],
            product_species=[0, 3],
            product_stoichiometry=[1, 1],
            propensity=MassActionPropensity(
                rate_constant=k_cat,
                reactant_species=[2],
                reactant_stoichiometry=[1],
            ),
        ),
    ]
    return _build_network("michaelis_menten_enzyme", species, reactions)


def multi_species_cascade(
    n_species: int = 5,
    k_activation: float = 1.0,
    k_degradation: float = 0.1,
    hill_coeff: float = 2.0,
    k_half: float = 10.0,
    max_copy: int = 40,
) -> ReactionNetwork:
    """
    Parameterizable n-species signaling cascade with Hill function regulation.

    Scales from 2 to 20+ species for scalability testing.

    Structure:
    - Species 0: constitutive production + degradation (input signal)
    - Species i (i>0): produced via Hill activation by species i-1, degraded linearly

    Reactions for each layer i:
    - ∅ →[Hill(X_{i-1})] X_i   (activated production, Hill kinetics)
    - X_i →[k_deg] ∅           (linear degradation)

    For species 0:
    - ∅ →[k_act] X_0           (constitutive production)
    - X_0 →[k_deg] ∅           (degradation)
    """
    if n_species < 2:
        raise ValueError("multi_species_cascade requires at least 2 species")

    species = [
        Species(name=f"X{i}", index=i, max_copy_number=max_copy)
        for i in range(n_species)
    ]
    reactions = []
    idx = 0

    # Species 0: constitutive production
    reactions.append(
        Reaction(
            name="prod_X0",
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

    # Species 0: degradation
    reactions.append(
        Reaction(
            name="deg_X0",
            index=idx,
            reactant_species=[0],
            reactant_stoichiometry=[1],
            product_species=[],
            product_stoichiometry=[],
            propensity=MassActionPropensity(
                rate_constant=k_degradation, reactant_species=[0], reactant_stoichiometry=[1],
            ),
        )
    )
    idx += 1

    # Downstream species: Hill-activated production + degradation
    for i in range(1, n_species):
        reactions.append(
            Reaction(
                name=f"prod_X{i}",
                index=idx,
                reactant_species=[],
                reactant_stoichiometry=[],
                product_species=[i],
                product_stoichiometry=[1],
                propensity=HillPropensity(
                    v_max=k_activation,
                    k_half=k_half,
                    hill_coefficient=hill_coeff,
                    species_index=i - 1,
                    activation=True,
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
                    rate_constant=k_degradation, reactant_species=[i], reactant_stoichiometry=[1],
                ),
            )
        )
        idx += 1

    return _build_network(f"multi_species_cascade_{n_species}", species, reactions)
