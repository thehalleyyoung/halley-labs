"""Circuit motif library for BioProver.

Provides a curated collection of ~15-20 canonical synthetic-biology circuit
motifs (toggle switch, repressilator, feed-forward loops, logic gates, etc.).
Each motif carries a BioModel template with real ODE models and literature-
sourced parameter values, a default Bio-STL specification, and known
analytical properties.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
from bioprover.temporal.stl_ast import (
    Always,
    ComparisonOp,
    Eventually,
    Expression,
    Interval,
    Predicate,
    STLAnd,
    STLFormula,
    STLNot,
    STLOr,
    make_var_expr,
)


# ---------------------------------------------------------------------------
# CircuitMotif
# ---------------------------------------------------------------------------

@dataclass
class CircuitMotif:
    """A reusable circuit motif with an ODE model and default specification.

    Attributes:
        name:           Short identifier.
        description:    Biological function / purpose.
        category:       Functional category (e.g. ``"memory"``, ``"logic"``).
        model_builder:  Callable that returns a fresh :class:`BioModel`.
        default_spec:   Optional default Bio-STL specification.
        known_properties: Analytical properties (e.g. monotonicity).
        literature_ref: Primary literature reference string.
        default_params: Default parameter dict for the motif builder.
    """

    name: str
    description: str
    category: str = ""
    model_builder: Optional[Callable[..., BioModel]] = None
    default_spec: Optional[STLFormula] = None
    known_properties: Dict[str, Any] = field(default_factory=dict)
    literature_ref: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)

    def build_model(self, **overrides: Any) -> BioModel:
        """Construct a :class:`BioModel` using default (or overridden) params."""
        if self.model_builder is None:
            raise RuntimeError(f"Motif '{self.name}' has no model builder")
        params = {**self.default_params, **overrides}
        return self.model_builder(**params)


# ======================================================================== #
#                        Model-builder functions                            #
# ======================================================================== #

def _add_production_repression(
    model: BioModel,
    product: str,
    repressor: str,
    rxn_prefix: str,
    vmax: float,
    K: float,
    n: float,
    gamma: float,
) -> None:
    """Helper: add Hill-repression production + linear degradation."""
    hill = HillRepression(Vmax=vmax, K=K, n=n)
    hill.repressor_name = repressor
    model.add_reaction(Reaction(
        name=f"{rxn_prefix}_prod",
        reactants=[],
        products=[StoichiometryEntry(product)],
        kinetic_law=hill,
        modifiers=[repressor],
    ))
    deg = LinearDegradation(rate=gamma)
    deg.species_name = product
    model.add_reaction(Reaction(
        name=f"{rxn_prefix}_deg",
        reactants=[StoichiometryEntry(product)],
        products=[],
        kinetic_law=deg,
    ))


def _add_production_activation(
    model: BioModel,
    product: str,
    activator: str,
    rxn_prefix: str,
    vmax: float,
    K: float,
    n: float,
    gamma: float,
) -> None:
    """Helper: add Hill-activation production + linear degradation."""
    hill = HillActivation(Vmax=vmax, K=K, n=n)
    hill.activator_name = activator
    model.add_reaction(Reaction(
        name=f"{rxn_prefix}_prod",
        reactants=[],
        products=[StoichiometryEntry(product)],
        kinetic_law=hill,
        modifiers=[activator],
    ))
    deg = LinearDegradation(rate=gamma)
    deg.species_name = product
    model.add_reaction(Reaction(
        name=f"{rxn_prefix}_deg",
        reactants=[StoichiometryEntry(product)],
        products=[],
        kinetic_law=deg,
    ))


# ---------------------------------------------------------------------------
# Toggle switch  (Gardner, Cantor & Collins, Nature 2000)
# ---------------------------------------------------------------------------

def _build_toggle_switch(
    alpha1: float = 5.0,
    alpha2: float = 5.0,
    K1: float = 10.0,
    K2: float = 10.0,
    n1: float = 2.0,
    n2: float = 2.0,
    gamma1: float = 0.0693,
    gamma2: float = 0.0693,
) -> BioModel:
    """Gardner toggle switch: two mutually repressing genes.

    dx/dt = α₁ · K₁ⁿ¹/(K₁ⁿ¹ + yⁿ¹) − γ₁·x
    dy/dt = α₂ · K₂ⁿ²/(K₂ⁿ² + xⁿ²) − γ₂·y
    """
    m = BioModel("toggle_switch")
    m.add_species(Species("x", initial_concentration=10.0))
    m.add_species(Species("y", initial_concentration=0.1))
    _add_production_repression(m, "x", "y", "x", alpha1, K1, n1, gamma1)
    _add_production_repression(m, "y", "x", "y", alpha2, K2, n2, gamma2)
    return m


# ---------------------------------------------------------------------------
# Repressilator  (Elowitz & Leibler, Nature 2000)
# ---------------------------------------------------------------------------

def _build_repressilator(
    alpha: float = 5.0,
    K: float = 10.0,
    n: float = 2.0,
    gamma: float = 0.0693,
) -> BioModel:
    """Three-gene ring oscillator: TetR ⊣ LacI ⊣ CI ⊣ TetR.

    Parameters from Elowitz & Leibler (2000) supplementary.
    """
    m = BioModel("repressilator")
    for sp in ["TetR", "LacI", "CI"]:
        m.add_species(Species(sp, initial_concentration=5.0))
    # TetR represses LacI, LacI represses CI, CI represses TetR
    _add_production_repression(m, "LacI", "TetR", "lacI", alpha, K, n, gamma)
    _add_production_repression(m, "CI", "LacI", "cI", alpha, K, n, gamma)
    _add_production_repression(m, "TetR", "CI", "tetR", alpha, K, n, gamma)
    return m


# ---------------------------------------------------------------------------
# Feed-forward loops (FFL) — all 8 types (Alon, Nat Rev Genet 2007)
# ---------------------------------------------------------------------------

class FFLType(Enum):
    """Eight canonical FFL types (coherent C / incoherent I, subtypes 1-4)."""
    C1 = auto()   # X→Y→Z, X→Z  (all activation)
    C2 = auto()   # X⊣Y⊣Z, X→Z
    C3 = auto()   # X→Y⊣Z, X⊣Z
    C4 = auto()   # X⊣Y→Z, X⊣Z
    I1 = auto()   # X→Y→Z, X⊣Z  (incoherent type 1)
    I2 = auto()   # X⊣Y⊣Z, X⊣Z
    I3 = auto()   # X→Y⊣Z, X→Z
    I4 = auto()   # X⊣Y→Z, X→Z


def _build_ffl(
    ffl_type: FFLType = FFLType.C1,
    alpha: float = 5.0,
    K: float = 10.0,
    n: float = 2.0,
    gamma: float = 0.0693,
) -> BioModel:
    """Construct a 3-node feed-forward loop of the specified type.

    Node X is the input, Y the intermediate, Z the output.
    """
    m = BioModel(f"ffl_{ffl_type.name}")
    m.add_species(Species("X", initial_concentration=0.0))
    m.add_species(Species("Y", initial_concentration=0.0))
    m.add_species(Species("Z", initial_concentration=0.0))

    # X is an external input — constitutive production (clamped at start)
    m.add_reaction(Reaction(
        name="X_prod",
        reactants=[],
        products=[StoichiometryEntry("X")],
        kinetic_law=ConstitutiveProduction(rate=alpha * gamma),
    ))
    deg_x = LinearDegradation(rate=gamma)
    deg_x.species_name = "X"
    m.add_reaction(Reaction(
        name="X_deg",
        reactants=[StoichiometryEntry("X")],
        products=[],
        kinetic_law=deg_x,
    ))

    # X→Y or X⊣Y
    x_activates_y = ffl_type in (FFLType.C1, FFLType.C3, FFLType.I1, FFLType.I3)
    if x_activates_y:
        _add_production_activation(m, "Y", "X", "Y", alpha, K, n, gamma)
    else:
        _add_production_repression(m, "Y", "X", "Y", alpha, K, n, gamma)

    # Y→Z or Y⊣Z
    y_activates_z = ffl_type in (FFLType.C1, FFLType.C2, FFLType.I1, FFLType.I2,
                                  FFLType.C4, FFLType.I4)
    # Correct: C1(Y→Z), C2(Y⊣Z→ wait: review)
    # C1: X→Y→Z  => Y activates Z
    # C2: X⊣Y⊣Z  => Y represses Z
    # C3: X→Y⊣Z  => Y represses Z
    # C4: X⊣Y→Z  => Y activates Z
    # I1: X→Y→Z  => Y activates Z
    # I2: X⊣Y⊣Z  => Y represses Z
    # I3: X→Y⊣Z  => Y represses Z
    # I4: X⊣Y→Z  => Y activates Z
    y_activates_z = ffl_type in (FFLType.C1, FFLType.C4, FFLType.I1, FFLType.I4)
    if y_activates_z:
        _add_production_activation(m, "Z", "Y", "Z_from_Y", alpha, K, n, gamma)
    else:
        _add_production_repression(m, "Z", "Y", "Z_from_Y", alpha, K, n, gamma)

    # X→Z or X⊣Z  (direct arm)
    # C1: X→Z, C2: X→Z, C3: X⊣Z, C4: X⊣Z
    # I1: X⊣Z, I2: X⊣Z, I3: X→Z, I4: X→Z
    x_activates_z = ffl_type in (FFLType.C1, FFLType.C2, FFLType.I3, FFLType.I4)
    if x_activates_z:
        _add_production_activation(m, "Z", "X", "Z_from_X", alpha, K, n, gamma)
    else:
        _add_production_repression(m, "Z", "X", "Z_from_X", alpha, K, n, gamma)

    return m


# ---------------------------------------------------------------------------
# Negative autoregulation
# ---------------------------------------------------------------------------

def _build_negative_autoregulation(
    alpha: float = 10.0,
    K: float = 5.0,
    n: float = 2.0,
    gamma: float = 0.0693,
) -> BioModel:
    """Gene that represses its own transcription.

    dx/dt = α · Kⁿ/(Kⁿ + xⁿ) − γ·x
    Rosenfeld et al., J Mol Biol 2002.
    """
    m = BioModel("negative_autoregulation")
    m.add_species(Species("X", initial_concentration=0.1))
    _add_production_repression(m, "X", "X", "X", alpha, K, n, gamma)
    return m


# ---------------------------------------------------------------------------
# Positive autoregulation
# ---------------------------------------------------------------------------

def _build_positive_autoregulation(
    alpha: float = 5.0,
    K: float = 5.0,
    n: float = 2.0,
    gamma: float = 0.0693,
    basal: float = 0.5,
) -> BioModel:
    """Gene that activates its own transcription (plus basal production).

    dx/dt = α · xⁿ/(Kⁿ + xⁿ) + basal − γ·x
    """
    m = BioModel("positive_autoregulation")
    m.add_species(Species("X", initial_concentration=0.1))
    _add_production_activation(m, "X", "X", "X", alpha, K, n, gamma)
    m.add_reaction(Reaction(
        name="X_basal",
        reactants=[],
        products=[StoichiometryEntry("X")],
        kinetic_law=ConstitutiveProduction(rate=basal),
    ))
    return m


# ---------------------------------------------------------------------------
# Genetic logic gates (NOT, AND, NAND, OR, NOR)
# ---------------------------------------------------------------------------

def _build_not_gate(
    alpha: float = 5.0,
    K: float = 10.0,
    n: float = 2.0,
    gamma: float = 0.0693,
) -> BioModel:
    """Inverter: output is high when input is low."""
    m = BioModel("not_gate")
    m.add_species(Species("input", initial_concentration=0.0))
    m.add_species(Species("output", initial_concentration=0.0))
    _add_production_repression(m, "output", "input", "out", alpha, K, n, gamma)
    return m


def _build_and_gate(
    alpha: float = 5.0,
    K: float = 10.0,
    n: float = 2.0,
    gamma: float = 0.0693,
) -> BioModel:
    """AND gate via layered activators: both A and B must be present.

    Uses a cascade: A activates intermediate M, M AND B gate Z
    through a simplified product-based Hill kinetics.
    """
    m = BioModel("and_gate")
    for sp in ["A", "B", "M", "Z"]:
        m.add_species(Species(sp, initial_concentration=0.0))
    _add_production_activation(m, "M", "A", "M", alpha, K, n, gamma)
    _add_production_activation(m, "Z", "M", "Z_from_M", alpha, K, n, gamma)
    # B also required: add activation from B to Z
    _add_production_activation(m, "Z", "B", "Z_from_B", alpha, K, n, gamma)
    return m


def _build_nand_gate(
    alpha: float = 5.0,
    K: float = 10.0,
    n: float = 2.0,
    gamma: float = 0.0693,
) -> BioModel:
    """NAND gate: output low only when both inputs high.

    Implemented as two repressors in series (each can independently
    maintain output).
    """
    m = BioModel("nand_gate")
    for sp in ["A", "B", "Z"]:
        m.add_species(Species(sp, initial_concentration=0.0))
    _add_production_repression(m, "Z", "A", "Z_rep_A", alpha, K, n, gamma)
    _add_production_repression(m, "Z", "B", "Z_rep_B", alpha, K, n, gamma)
    return m


def _build_or_gate(
    alpha: float = 5.0,
    K: float = 10.0,
    n: float = 2.0,
    gamma: float = 0.0693,
) -> BioModel:
    """OR gate: output high when either input is high."""
    m = BioModel("or_gate")
    for sp in ["A", "B", "Z"]:
        m.add_species(Species(sp, initial_concentration=0.0))
    _add_production_activation(m, "Z", "A", "Z_from_A", alpha, K, n, gamma)
    _add_production_activation(m, "Z", "B", "Z_from_B", alpha, K, n, gamma)
    return m


def _build_nor_gate(
    alpha: float = 5.0,
    K: float = 10.0,
    n: float = 2.5,
    gamma: float = 0.0693,
) -> BioModel:
    """NOR gate: output high only when both inputs low.

    Two repressors in parallel feeding a single promoter
    (Cello-style, Nielsen et al. Science 2016).
    """
    m = BioModel("nor_gate")
    for sp in ["A", "B", "Z"]:
        m.add_species(Species(sp, initial_concentration=0.0))
    _add_production_repression(m, "Z", "A", "Z_rep_A", alpha, K, n, gamma)
    _add_production_repression(m, "Z", "B", "Z_rep_B", alpha, K, n, gamma)
    return m


# ---------------------------------------------------------------------------
# Band-pass filter
# ---------------------------------------------------------------------------

def _build_bandpass_filter(
    alpha_act: float = 8.0,
    K_act: float = 5.0,
    n_act: float = 2.0,
    alpha_rep: float = 8.0,
    K_rep: float = 20.0,
    n_rep: float = 4.0,
    gamma: float = 0.0693,
) -> BioModel:
    """Band-pass filter: output high only for intermediate input levels.

    Implemented as parallel activation (low K) and repression (high K)
    arms feeding the same output.  Basu et al., Nature 2005.
    """
    m = BioModel("bandpass_filter")
    m.add_species(Species("input", initial_concentration=0.0))
    m.add_species(Species("activator_arm", initial_concentration=0.0))
    m.add_species(Species("output", initial_concentration=0.0))

    # Low-threshold activation arm
    _add_production_activation(
        m, "activator_arm", "input", "act_arm", alpha_act, K_act, n_act, gamma
    )
    # Output activated by arm, repressed directly by high input
    _add_production_activation(
        m, "output", "activator_arm", "out_act", alpha_act, K_act, n_act, gamma
    )
    _add_production_repression(
        m, "output", "input", "out_rep", alpha_rep, K_rep, n_rep, gamma
    )
    return m


# ---------------------------------------------------------------------------
# Pulse generator (incoherent FFL type 1)
# ---------------------------------------------------------------------------

def _build_pulse_generator(
    alpha: float = 5.0,
    K_act: float = 5.0,
    K_rep: float = 15.0,
    n: float = 2.0,
    gamma: float = 0.0693,
) -> BioModel:
    """Pulse generator via incoherent FFL type 1.

    X activates both Y and Z directly; Y represses Z.  Z shows a
    transient pulse after step increase in X.
    Mangan & Alon, PNAS 2003.
    """
    m = BioModel("pulse_generator")
    m.add_species(Species("X", initial_concentration=0.0))
    m.add_species(Species("Y", initial_concentration=0.0))
    m.add_species(Species("Z", initial_concentration=0.0))

    _add_production_activation(m, "Y", "X", "Y", alpha, K_act, n, gamma)
    _add_production_activation(m, "Z", "X", "Z_act", alpha, K_act, n, gamma)
    _add_production_repression(m, "Z", "Y", "Z_rep", alpha, K_rep, n, gamma)
    return m


# ---------------------------------------------------------------------------
# Tunable oscillator (Goodwin oscillator with degradation)
# ---------------------------------------------------------------------------

def _build_tunable_oscillator(
    alpha: float = 10.0,
    K: float = 5.0,
    n: float = 3.0,
    gamma_x: float = 0.1,
    gamma_y: float = 0.05,
    gamma_z: float = 0.05,
    k_xy: float = 1.0,
    k_yz: float = 1.0,
) -> BioModel:
    """Goodwin-type oscillator with three-stage cascade and negative feedback.

    mRNA (X) → protein (Y) → modified protein (Z) ⊣ X.
    Period tunable via degradation rates.
    Goodwin, Adv Enzyme Regul 1965; Griffith, J Theor Biol 1968.
    """
    m = BioModel("tunable_oscillator")
    m.add_species(Species("X", initial_concentration=5.0))
    m.add_species(Species("Y", initial_concentration=2.0))
    m.add_species(Species("Z", initial_concentration=1.0))

    # Z represses X
    _add_production_repression(m, "X", "Z", "X", alpha, K, n, gamma_x)
    # X linearly produces Y
    hill_xy = HillActivation(Vmax=k_xy, K=1.0, n=1.0)
    hill_xy.activator_name = "X"
    m.add_reaction(Reaction(
        name="Y_prod",
        reactants=[],
        products=[StoichiometryEntry("Y")],
        kinetic_law=hill_xy,
        modifiers=["X"],
    ))
    deg_y = LinearDegradation(rate=gamma_y)
    deg_y.species_name = "Y"
    m.add_reaction(Reaction(
        name="Y_deg",
        reactants=[StoichiometryEntry("Y")],
        products=[],
        kinetic_law=deg_y,
    ))
    # Y linearly produces Z
    hill_yz = HillActivation(Vmax=k_yz, K=1.0, n=1.0)
    hill_yz.activator_name = "Y"
    m.add_reaction(Reaction(
        name="Z_prod",
        reactants=[],
        products=[StoichiometryEntry("Z")],
        kinetic_law=hill_yz,
        modifiers=["Y"],
    ))
    deg_z = LinearDegradation(rate=gamma_z)
    deg_z.species_name = "Z"
    m.add_reaction(Reaction(
        name="Z_deg",
        reactants=[StoichiometryEntry("Z")],
        products=[],
        kinetic_law=deg_z,
    ))
    return m


# ---------------------------------------------------------------------------
# Bistable memory element (toggle switch with positive feedback)
# ---------------------------------------------------------------------------

def _build_bistable_memory(
    alpha: float = 6.0,
    K: float = 8.0,
    n: float = 2.5,
    gamma: float = 0.0693,
    basal: float = 0.2,
) -> BioModel:
    """Bistable memory: toggle switch with additional positive autoregulation.

    Two mutually repressing genes, each with positive autoregulation,
    giving robust bistability.
    """
    m = BioModel("bistable_memory")
    m.add_species(Species("A", initial_concentration=10.0))
    m.add_species(Species("B", initial_concentration=0.1))
    _add_production_repression(m, "A", "B", "A_rep", alpha, K, n, gamma)
    _add_production_repression(m, "B", "A", "B_rep", alpha, K, n, gamma)
    # Positive autoregulation on A
    _add_production_activation(m, "A", "A", "A_auto", alpha * 0.3, K, n, 0.0)
    _add_production_activation(m, "B", "B", "B_auto", alpha * 0.3, K, n, 0.0)
    # Basal leak
    m.add_reaction(Reaction(
        name="A_basal", reactants=[], products=[StoichiometryEntry("A")],
        kinetic_law=ConstitutiveProduction(rate=basal),
    ))
    m.add_reaction(Reaction(
        name="B_basal", reactants=[], products=[StoichiometryEntry("B")],
        kinetic_law=ConstitutiveProduction(rate=basal),
    ))
    return m


# ---------------------------------------------------------------------------
# N-stage cascade
# ---------------------------------------------------------------------------

def _build_cascade(
    depth: int = 3,
    alpha: float = 5.0,
    K: float = 10.0,
    n: float = 2.0,
    gamma: float = 0.0693,
    activation: bool = True,
) -> BioModel:
    """Linear signalling cascade of *depth* stages.

    Stage 0 is the input (constitutive); each subsequent stage is
    activated (or repressed) by the previous one.
    """
    m = BioModel(f"cascade_depth{depth}")
    names = [f"S{i}" for i in range(depth)]
    for nm in names:
        m.add_species(Species(nm, initial_concentration=0.0))
    # Constitutive input
    m.add_reaction(Reaction(
        name="S0_prod", reactants=[], products=[StoichiometryEntry("S0")],
        kinetic_law=ConstitutiveProduction(rate=alpha * gamma),
    ))
    deg0 = LinearDegradation(rate=gamma)
    deg0.species_name = "S0"
    m.add_reaction(Reaction(
        name="S0_deg", reactants=[StoichiometryEntry("S0")], products=[],
        kinetic_law=deg0,
    ))
    for i in range(1, depth):
        if activation:
            _add_production_activation(
                m, names[i], names[i - 1], names[i], alpha, K, n, gamma
            )
        else:
            _add_production_repression(
                m, names[i], names[i - 1], names[i], alpha, K, n, gamma
            )
    return m


# ======================================================================== #
#                           MotifLibrary                                    #
# ======================================================================== #

class MotifLibrary:
    """Registry of canonical circuit motifs.

    Provides lookup, search, and composition utilities.
    """

    def __init__(self, *, load_builtins: bool = True) -> None:
        self._motifs: Dict[str, CircuitMotif] = {}
        if load_builtins:
            self._register_builtins()

    # -- CRUD ---------------------------------------------------------------

    def register(self, motif: CircuitMotif) -> None:
        """Register a motif.  Overwrites any existing motif with the same name."""
        self._motifs[motif.name] = motif

    def get(self, name: str) -> Optional[CircuitMotif]:
        """Retrieve a motif by name."""
        return self._motifs.get(name)

    @property
    def names(self) -> List[str]:
        return sorted(self._motifs)

    @property
    def all_motifs(self) -> List[CircuitMotif]:
        return list(self._motifs.values())

    def __len__(self) -> int:
        return len(self._motifs)

    def __contains__(self, name: str) -> bool:
        return name in self._motifs

    # -- search -------------------------------------------------------------

    def search_by_category(self, category: str) -> List[CircuitMotif]:
        """Return motifs matching *category* (case-insensitive substring)."""
        cat = category.lower()
        return [m for m in self._motifs.values() if cat in m.category.lower()]

    def search_by_function(self, keyword: str) -> List[CircuitMotif]:
        """Return motifs whose description contains *keyword*."""
        kw = keyword.lower()
        return [
            m for m in self._motifs.values()
            if kw in m.description.lower() or kw in m.name.lower()
        ]

    def search_by_property(self, prop_name: str) -> List[CircuitMotif]:
        """Return motifs that declare a known property named *prop_name*."""
        return [
            m for m in self._motifs.values()
            if prop_name in m.known_properties
        ]

    # -- composition --------------------------------------------------------

    @staticmethod
    def compose(
        upstream: CircuitMotif,
        downstream: CircuitMotif,
        connection_species: str,
        upstream_output: str,
        downstream_input: str,
    ) -> BioModel:
        """Connect two motifs by identifying an output of *upstream* with an
        input of *downstream*.

        Builds both models, renames *downstream_input* to *upstream_output*
        in the downstream model, then merges species and reactions.
        """
        m_up = upstream.build_model()
        m_down = downstream.build_model()

        composed = BioModel(f"{upstream.name}_{downstream.name}_composed")

        # Add upstream species and reactions
        for sp in m_up.species:
            composed.add_species(copy.deepcopy(sp))
        for rxn in m_up.reactions:
            composed.add_reaction(copy.deepcopy(rxn))

        # Add downstream, renaming the input species
        rename = {downstream_input: upstream_output}
        for sp in m_down.species:
            new_sp = copy.deepcopy(sp)
            if sp.name in rename:
                continue  # already exists from upstream
            if sp.name in [s.name for s in composed.species]:
                continue
            composed.add_species(new_sp)
        for rxn in m_down.reactions:
            new_rxn = copy.deepcopy(rxn)
            # Rename in stoichiometry
            for entry in new_rxn.reactants:
                if entry.species_name in rename:
                    object.__setattr__(entry, "species_name", rename[entry.species_name])
            for entry in new_rxn.products:
                if entry.species_name in rename:
                    object.__setattr__(entry, "species_name", rename[entry.species_name])
            new_rxn.modifiers = [
                rename.get(mod, mod) for mod in new_rxn.modifiers
            ]
            # Avoid reaction name collision
            if new_rxn.name in [r.name for r in composed.reactions]:
                new_rxn.name = f"ds_{new_rxn.name}"
            composed.add_reaction(new_rxn)

        return composed

    # -- built-in motifs ----------------------------------------------------

    def _register_builtins(self) -> None:
        """Register the canonical synthetic-biology motifs."""

        # 1. Toggle switch
        ts_spec = Always(
            STLOr(
                Predicate(make_var_expr("x"), ComparisonOp.GT, 5.0),
                Predicate(make_var_expr("y"), ComparisonOp.GT, 5.0),
            ),
            Interval(50, 200),
        )
        self.register(CircuitMotif(
            name="toggle_switch",
            description="Bistable genetic toggle switch with two mutually "
                        "repressing genes; exhibits two stable steady states",
            category="memory",
            model_builder=_build_toggle_switch,
            default_spec=ts_spec,
            known_properties={
                "bistability": True,
                "num_steady_states": 3,
                "symmetry": "symmetric when α₁=α₂",
            },
            literature_ref="Gardner, Cantor & Collins, Nature 403:339 (2000)",
            default_params=dict(alpha1=5.0, alpha2=5.0, K1=10.0, K2=10.0,
                                n1=2.0, n2=2.0, gamma1=0.0693, gamma2=0.0693),
        ))

        # 2. Repressilator
        rep_spec = Always(
            Eventually(
                Predicate(make_var_expr("TetR"), ComparisonOp.GT, 3.0),
                Interval(0, 50),
            ),
            Interval(0, 200),
        )
        self.register(CircuitMotif(
            name="repressilator",
            description="Three-gene ring oscillator producing sustained "
                        "oscillations in protein concentrations",
            category="oscillator",
            model_builder=_build_repressilator,
            default_spec=rep_spec,
            known_properties={
                "oscillation": True,
                "period_range_min": (30, 100),
                "requires_cooperativity": "n > 1",
            },
            literature_ref="Elowitz & Leibler, Nature 403:335 (2000)",
            default_params=dict(alpha=5.0, K=10.0, n=2.0, gamma=0.0693),
        ))

        # 3-10. Feed-forward loops (all 8 types)
        ffl_descriptions = {
            FFLType.C1: "Coherent type 1: sign-sensitive delay element",
            FFLType.C2: "Coherent type 2: X⊣Y⊣Z, X→Z",
            FFLType.C3: "Coherent type 3: X→Y⊣Z, X⊣Z",
            FFLType.C4: "Coherent type 4: X⊣Y→Z, X⊣Z",
            FFLType.I1: "Incoherent type 1: pulse generator (most common IFFL)",
            FFLType.I2: "Incoherent type 2: X⊣Y⊣Z, X⊣Z",
            FFLType.I3: "Incoherent type 3: X→Y⊣Z, X→Z",
            FFLType.I4: "Incoherent type 4: X⊣Y→Z, X→Z",
        }
        ffl_properties = {
            FFLType.C1: {"sign_sensitive_delay": True, "monotone": True},
            FFLType.C2: {"sign_sensitive_delay": True},
            FFLType.C3: {"sign_sensitive_delay": True},
            FFLType.C4: {"sign_sensitive_delay": True},
            FFLType.I1: {"pulse_generation": True, "adaptation": True},
            FFLType.I2: {"pulse_generation": True},
            FFLType.I3: {"fold_change_detection": True},
            FFLType.I4: {"fold_change_detection": True},
        }
        for ft in FFLType:
            self.register(CircuitMotif(
                name=f"ffl_{ft.name}",
                description=f"Feed-forward loop {ffl_descriptions[ft]}",
                category="network_motif",
                model_builder=lambda ffl_type=ft, **kw: _build_ffl(ffl_type=ffl_type, **kw),
                known_properties=ffl_properties.get(ft, {}),
                literature_ref="Alon, Nat Rev Genet 8:450 (2007)",
                default_params=dict(ffl_type=ft, alpha=5.0, K=10.0,
                                    n=2.0, gamma=0.0693),
            ))

        # 11. Negative autoregulation
        nar_spec = Eventually(
            STLAnd(
                Predicate(make_var_expr("X"), ComparisonOp.GE, 4.0),
                Predicate(make_var_expr("X"), ComparisonOp.LE, 6.0),
            ),
            Interval(0, 30),
        )
        self.register(CircuitMotif(
            name="negative_autoregulation",
            description="Gene that represses its own expression; speeds "
                        "response time and reduces noise",
            category="autoregulation",
            model_builder=_build_negative_autoregulation,
            default_spec=nar_spec,
            known_properties={
                "speeds_response": True,
                "reduces_noise": True,
                "monotone": True,
            },
            literature_ref="Rosenfeld et al., J Mol Biol 323:785 (2002)",
            default_params=dict(alpha=10.0, K=5.0, n=2.0, gamma=0.0693),
        ))

        # 12. Positive autoregulation
        self.register(CircuitMotif(
            name="positive_autoregulation",
            description="Gene that activates its own expression; slows "
                        "response but can create bistability",
            category="autoregulation",
            model_builder=_build_positive_autoregulation,
            known_properties={
                "slows_response": True,
                "can_be_bistable": "when n >= 2 and basal is low",
            },
            literature_ref="Becskei et al., EMBO J 20:2528 (2001)",
            default_params=dict(alpha=5.0, K=5.0, n=2.0, gamma=0.0693,
                                basal=0.5),
        ))

        # 13. NOT gate
        self.register(CircuitMotif(
            name="not_gate",
            description="Genetic inverter: output high when input low",
            category="logic",
            model_builder=_build_not_gate,
            known_properties={"boolean_function": "NOT", "monotone": True},
            literature_ref="Weiss et al., Nat Comput 2:47 (2003)",
            default_params=dict(alpha=5.0, K=10.0, n=2.0, gamma=0.0693),
        ))

        # 14. AND gate
        self.register(CircuitMotif(
            name="and_gate",
            description="Genetic AND gate: output high when both inputs high",
            category="logic",
            model_builder=_build_and_gate,
            known_properties={"boolean_function": "AND"},
            literature_ref="Anderson, Voigt & Arkin, Mol Syst Biol 3:133 (2007)",
            default_params=dict(alpha=5.0, K=10.0, n=2.0, gamma=0.0693),
        ))

        # 15. NAND gate
        self.register(CircuitMotif(
            name="nand_gate",
            description="Genetic NAND gate: output low only when both inputs high",
            category="logic",
            model_builder=_build_nand_gate,
            known_properties={"boolean_function": "NAND", "universal": True},
            literature_ref="Tamsir, Tabor & Voigt, Nature 469:212 (2011)",
            default_params=dict(alpha=5.0, K=10.0, n=2.0, gamma=0.0693),
        ))

        # 16. OR gate
        self.register(CircuitMotif(
            name="or_gate",
            description="Genetic OR gate: output high when either input high",
            category="logic",
            model_builder=_build_or_gate,
            known_properties={"boolean_function": "OR"},
            default_params=dict(alpha=5.0, K=10.0, n=2.0, gamma=0.0693),
        ))

        # 17. NOR gate
        self.register(CircuitMotif(
            name="nor_gate",
            description="Genetic NOR gate: output high only when both inputs low",
            category="logic",
            model_builder=_build_nor_gate,
            known_properties={"boolean_function": "NOR", "universal": True},
            literature_ref="Nielsen et al., Science 352:aac7341 (2016)",
            default_params=dict(alpha=5.0, K=10.0, n=2.5, gamma=0.0693),
        ))

        # 18. Band-pass filter
        self.register(CircuitMotif(
            name="bandpass_filter",
            description="Band-pass filter: output high only for intermediate "
                        "input concentrations",
            category="signal_processing",
            model_builder=_build_bandpass_filter,
            known_properties={"non_monotone": True, "band_pass": True},
            literature_ref="Basu et al., Nature 434:1130 (2005)",
            default_params=dict(alpha_act=8.0, K_act=5.0, n_act=2.0,
                                alpha_rep=8.0, K_rep=20.0, n_rep=4.0,
                                gamma=0.0693),
        ))

        # 19. Pulse generator (IFFL type 1)
        pulse_spec = STLAnd(
            Eventually(
                Predicate(make_var_expr("Z"), ComparisonOp.GT, 3.0),
                Interval(0, 30),
            ),
            Eventually(
                Predicate(make_var_expr("Z"), ComparisonOp.LT, 1.0),
                Interval(30, 100),
            ),
        )
        self.register(CircuitMotif(
            name="pulse_generator",
            description="Transient pulse via incoherent FFL type 1; output "
                        "spikes then returns to baseline",
            category="signal_processing",
            model_builder=_build_pulse_generator,
            default_spec=pulse_spec,
            known_properties={
                "pulse_generation": True,
                "adaptation": True,
                "pulse_duration_depends_on": "Y production delay",
            },
            literature_ref="Mangan & Alon, PNAS 100:11980 (2003)",
            default_params=dict(alpha=5.0, K_act=5.0, K_rep=15.0,
                                n=2.0, gamma=0.0693),
        ))

        # 20. Tunable oscillator (Goodwin)
        self.register(CircuitMotif(
            name="tunable_oscillator",
            description="Goodwin-type oscillator with three-stage negative "
                        "feedback; period tunable via degradation rates",
            category="oscillator",
            model_builder=_build_tunable_oscillator,
            known_properties={
                "oscillation": True,
                "requires_cooperativity": "n >= 8 for Goodwin, relaxed with cascading",
                "tunable_period": True,
            },
            literature_ref="Goodwin, Adv Enzyme Regul 3:425 (1965)",
            default_params=dict(alpha=10.0, K=5.0, n=3.0,
                                gamma_x=0.1, gamma_y=0.05, gamma_z=0.05,
                                k_xy=1.0, k_yz=1.0),
        ))

        # 21. Bistable memory element
        self.register(CircuitMotif(
            name="bistable_memory",
            description="Robust bistable memory with mutual repression and "
                        "positive autoregulation on each gene",
            category="memory",
            model_builder=_build_bistable_memory,
            known_properties={
                "bistability": True,
                "hysteresis": True,
                "robust_to_noise": True,
            },
            literature_ref="Isaacs et al., Nat Biotechnol 21:1069 (2003)",
            default_params=dict(alpha=6.0, K=8.0, n=2.5, gamma=0.0693,
                                basal=0.2),
        ))

        # 22. Cascade (depth-parameterised)
        self.register(CircuitMotif(
            name="cascade",
            description="Linear signalling cascade of parameterised depth; "
                        "models signal propagation delay and attenuation",
            category="signal_processing",
            model_builder=_build_cascade,
            known_properties={
                "monotone": True,
                "delay_increases_with_depth": True,
            },
            default_params=dict(depth=3, alpha=5.0, K=10.0, n=2.0,
                                gamma=0.0693, activation=True),
        ))
