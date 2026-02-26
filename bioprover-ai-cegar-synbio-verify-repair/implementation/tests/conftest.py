"""Common fixtures for BioProver test suite."""

import os
import random
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

IMPL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIOPROVER_DIR = os.path.join(IMPL_DIR, "bioprover")

# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Pin random seeds for reproducibility in every test."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    yield seed


# ---------------------------------------------------------------------------
# Temporary directory
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory that is cleaned up after the test."""
    return tmp_path


@pytest.fixture
def tmp_file(tmp_path):
    """Provide a temporary file path (not yet created)."""
    return tmp_path / "output.tmp"


# ---------------------------------------------------------------------------
# Interval fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unit_interval():
    """The interval [0, 1]."""
    from bioprover.solver.interval import Interval
    return Interval(0.0, 1.0)


@pytest.fixture
def symmetric_interval():
    """The interval [-1, 1]."""
    from bioprover.solver.interval import Interval
    return Interval(-1.0, 1.0)


@pytest.fixture
def positive_interval():
    """The interval [1, 3]."""
    from bioprover.solver.interval import Interval
    return Interval(1.0, 3.0)


@pytest.fixture
def thin_interval():
    """A thin (degenerate) interval [2, 2]."""
    from bioprover.solver.interval import Interval
    return Interval(2.0, 2.0)


@pytest.fixture
def interval_vector_2d():
    """A 2-D interval vector [[0,1], [2,3]]."""
    from bioprover.solver.interval import Interval, IntervalVector
    return IntervalVector([Interval(0.0, 1.0), Interval(2.0, 3.0)])


@pytest.fixture
def identity_interval_matrix():
    """A 2x2 interval identity matrix."""
    from bioprover.solver.interval import IntervalMatrix
    return IntervalMatrix.identity(2)


# ---------------------------------------------------------------------------
# Expression fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def expr_vars():
    """Common expression variables x, y, z."""
    from bioprover.encoding.expression import Var
    return Var("x"), Var("y"), Var("z")


@pytest.fixture
def simple_expr():
    """Expression: x + y * 2."""
    from bioprover.encoding.expression import Var, Const
    x, y = Var("x"), Var("y")
    return x + y * Const(2.0)


# ---------------------------------------------------------------------------
# Species & Reaction fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def species_A():
    from bioprover.models.species import Species, SpeciesType
    return Species("A", initial_concentration=1.0, species_type=SpeciesType.PROTEIN)


@pytest.fixture
def species_B():
    from bioprover.models.species import Species, SpeciesType
    return Species("B", initial_concentration=0.5, species_type=SpeciesType.PROTEIN)


# ---------------------------------------------------------------------------
# BioModel fixtures
# ---------------------------------------------------------------------------


def _build_toggle_switch():
    """Build a genetic toggle switch model with mutual repression."""
    from bioprover.models.bio_model import BioModel
    from bioprover.models.species import Species, SpeciesType
    from bioprover.models.reactions import (
        Reaction, StoichiometryEntry, HillRepression, LinearDegradation,
    )

    model = BioModel("toggle_switch")

    sp_u = Species("u", initial_concentration=2.0, species_type=SpeciesType.PROTEIN)
    sp_v = Species("v", initial_concentration=0.5, species_type=SpeciesType.PROTEIN)
    model.add_species(sp_u)
    model.add_species(sp_v)

    # u represses v production, v represses u production
    hill_u = HillRepression(Vmax=3.0, K=1.0, n=2.0)
    hill_u.repressor_name = "v"
    rxn_u = Reaction(
        "u_production",
        reactants=[],
        products=[StoichiometryEntry("u")],
        kinetic_law=hill_u,
        modifiers=["v"],
    )

    hill_v = HillRepression(Vmax=3.0, K=1.0, n=2.0)
    hill_v.repressor_name = "u"
    rxn_v = Reaction(
        "v_production",
        reactants=[],
        products=[StoichiometryEntry("v")],
        kinetic_law=hill_v,
        modifiers=["u"],
    )

    deg_u_law = LinearDegradation(rate=1.0)
    deg_u_law.species_name = "u"
    rxn_deg_u = Reaction(
        "u_degradation",
        reactants=[StoichiometryEntry("u")],
        products=[],
        kinetic_law=deg_u_law,
    )

    deg_v_law = LinearDegradation(rate=1.0)
    deg_v_law.species_name = "v"
    rxn_deg_v = Reaction(
        "v_degradation",
        reactants=[StoichiometryEntry("v")],
        products=[],
        kinetic_law=deg_v_law,
    )

    model.add_reaction(rxn_u)
    model.add_reaction(rxn_v)
    model.add_reaction(rxn_deg_u)
    model.add_reaction(rxn_deg_v)
    return model


def _build_repressilator():
    """Build a three-gene repressilator model."""
    from bioprover.models.bio_model import BioModel
    from bioprover.models.species import Species, SpeciesType
    from bioprover.models.reactions import (
        Reaction, StoichiometryEntry, HillRepression, LinearDegradation,
    )

    model = BioModel("repressilator")
    names = ["lacI", "tetR", "cI"]
    for name in names:
        model.add_species(
            Species(name, initial_concentration=1.0, species_type=SpeciesType.PROTEIN)
        )

    repressor_of = {"lacI": "cI", "tetR": "lacI", "cI": "tetR"}

    for name in names:
        rep = repressor_of[name]
        hill = HillRepression(Vmax=5.0, K=1.0, n=2.0)
        hill.repressor_name = rep
        rxn_prod = Reaction(
            f"{name}_production",
            reactants=[],
            products=[StoichiometryEntry(name)],
            kinetic_law=hill,
            modifiers=[rep],
        )
        deg_law = LinearDegradation(rate=1.0)
        deg_law.species_name = name
        rxn_deg = Reaction(
            f"{name}_degradation",
            reactants=[StoichiometryEntry(name)],
            products=[],
            kinetic_law=deg_law,
        )
        model.add_reaction(rxn_prod)
        model.add_reaction(rxn_deg)

    return model


def _build_cascade():
    """Build a simple two-step enzymatic cascade A -> B -> C."""
    from bioprover.models.bio_model import BioModel
    from bioprover.models.species import Species, SpeciesType
    from bioprover.models.reactions import (
        Reaction, StoichiometryEntry, MichaelisMenten, LinearDegradation,
    )

    model = BioModel("cascade")
    for name, ic in [("A", 5.0), ("B", 0.0), ("C", 0.0)]:
        model.add_species(
            Species(name, initial_concentration=ic, species_type=SpeciesType.PROTEIN)
        )

    mm1 = MichaelisMenten(Vmax=2.0, Km=1.0)
    mm1.substrate_name = "A"
    rxn1 = Reaction(
        "A_to_B",
        reactants=[StoichiometryEntry("A")],
        products=[StoichiometryEntry("B")],
        kinetic_law=mm1,
    )

    mm2 = MichaelisMenten(Vmax=1.5, Km=0.5)
    mm2.substrate_name = "B"
    rxn2 = Reaction(
        "B_to_C",
        reactants=[StoichiometryEntry("B")],
        products=[StoichiometryEntry("C")],
        kinetic_law=mm2,
    )

    deg_law = LinearDegradation(rate=0.1)
    deg_law.species_name = "C"
    rxn_deg = Reaction(
        "C_degradation",
        reactants=[StoichiometryEntry("C")],
        products=[],
        kinetic_law=deg_law,
    )

    model.add_reaction(rxn1)
    model.add_reaction(rxn2)
    model.add_reaction(rxn_deg)
    return model


@pytest.fixture
def toggle_switch_model():
    return _build_toggle_switch()


@pytest.fixture
def repressilator_model():
    return _build_repressilator()


@pytest.fixture
def cascade_model():
    return _build_cascade()


# ---------------------------------------------------------------------------
# STL formula fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stl_always_positive():
    """G[0,10](x > 0)."""
    from bioprover.temporal.stl_ast import (
        Predicate, Expression, ComparisonOp, Always, Interval,
    )
    expr = Expression(variable="x")
    pred = Predicate(expr=expr, op=ComparisonOp.GT, threshold=0.0)
    return Always(child=pred, interval=Interval(lo=0.0, hi=10.0))


@pytest.fixture
def stl_eventually_high():
    """F[0,5](x > 2)."""
    from bioprover.temporal.stl_ast import (
        Predicate, Expression, ComparisonOp, Eventually, Interval,
    )
    expr = Expression(variable="x")
    pred = Predicate(expr=expr, op=ComparisonOp.GT, threshold=2.0)
    return Eventually(child=pred, interval=Interval(lo=0.0, hi=5.0))


@pytest.fixture
def stl_bistability():
    """(G[5,10](x > 2)) | (G[5,10](x < 0.5)) — bistability proxy."""
    from bioprover.temporal.stl_ast import (
        Predicate, Expression, ComparisonOp, Always, STLOr, Interval,
    )
    expr = Expression(variable="x")
    high = Predicate(expr=expr, op=ComparisonOp.GT, threshold=2.0)
    low = Predicate(expr=expr, op=ComparisonOp.LT, threshold=0.5)
    return STLOr(
        left=Always(child=high, interval=Interval(lo=5.0, hi=10.0)),
        right=Always(child=low, interval=Interval(lo=5.0, hi=10.0)),
    )


# ---------------------------------------------------------------------------
# Signal fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def constant_signal():
    """A constant signal x(t)=3 over [0, 10]."""
    from bioprover.temporal.robustness import Signal
    times = np.linspace(0, 10, 101)
    values = np.full_like(times, 3.0)
    return Signal(times=times, values=values, name="x")


@pytest.fixture
def sine_signal():
    """A sine-wave signal x(t)=sin(t) over [0, 10]."""
    from bioprover.temporal.robustness import Signal
    times = np.linspace(0, 10, 201)
    values = np.sin(times)
    return Signal(times=times, values=values, name="x")
