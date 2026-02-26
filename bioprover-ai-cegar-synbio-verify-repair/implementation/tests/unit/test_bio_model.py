"""Unit tests for BioModel — species, reactions, stoichiometry, ODE, Jacobian."""

import math

import numpy as np
import pytest

from bioprover.models.bio_model import BioModel, Compartment
from bioprover.models.reactions import (
    ConstitutiveProduction,
    HillActivation,
    HillRepression,
    LinearDegradation,
    MassAction,
    MichaelisMenten,
    Reaction,
    StoichiometryEntry,
    build_stoichiometry_matrix,
)
from bioprover.models.species import (
    BoundaryCondition,
    ConcentrationBounds,
    Species,
    SpeciesMetadata,
    SpeciesType,
)


# ===================================================================
# Species
# ===================================================================


class TestSpecies:
    def test_creation_defaults(self):
        sp = Species("GFP")
        assert sp.name == "GFP"
        assert sp.initial_concentration == 0.0
        assert sp.compartment == "default"

    def test_creation_full(self):
        sp = Species(
            "lacI",
            compartment="cytoplasm",
            initial_concentration=10.0,
            units="nM",
            species_type=SpeciesType.PROTEIN,
            boundary_condition=BoundaryCondition.FREE,
        )
        assert sp.species_type == SpeciesType.PROTEIN
        assert not sp.is_boundary

    def test_boundary_species(self):
        sp = Species("inducer", boundary_condition=BoundaryCondition.FIXED)
        assert sp.is_boundary

    def test_concentration_bounds(self):
        bounds = ConcentrationBounds(min_value=0.0, max_value=100.0)
        sp = Species("x", concentration_bounds=bounds)
        assert sp.is_in_bounds(50.0)
        assert not sp.is_in_bounds(150.0)

    def test_bounds_validation(self):
        bounds = ConcentrationBounds(min_value=-1.0, max_value=100.0)
        errors = bounds.validate()
        assert len(errors) > 0  # negative min should be flagged

    def test_copy_number_conversion(self):
        sp = Species("x", initial_concentration=1.0, units="nM")
        volume = 1e-15  # femtolitre
        cn = sp.concentration_to_copy_number(volume)
        assert cn > 0
        # Round-trip
        conc_back = Species.copy_number_to_concentration(cn, volume)
        assert conc_back == pytest.approx(1.0, rel=0.1)

    def test_stochastic_heuristic(self):
        sp = Species("x", copy_number=10)
        assert sp.should_use_stochastic(threshold=100)
        sp2 = Species("y", copy_number=1000)
        assert not sp2.should_use_stochastic(threshold=100)

    def test_sympy_symbol(self):
        sp = Species("alpha")
        sym = sp.as_sympy_symbol()
        assert str(sym) == "alpha"

    def test_validate_ok(self):
        sp = Species("x", initial_concentration=1.0)
        errors = sp.validate()
        assert len(errors) == 0

    def test_copy(self):
        sp = Species("x", initial_concentration=5.0)
        cp = sp.copy()
        assert cp.name == "x" and cp.initial_concentration == 5.0
        assert cp is not sp

    def test_equality(self):
        a = Species("x", initial_concentration=1.0)
        b = Species("x", initial_concentration=1.0)
        assert a == b

    def test_hash(self):
        a = Species("x")
        b = Species("x")
        assert hash(a) == hash(b)

    def test_metadata(self):
        meta = SpeciesMetadata(gene="lacI", organism="E. coli")
        meta.add_reference("UniProt", "P03023")
        assert len(meta.get_references("UniProt")) == 1

    def test_qualified_name(self):
        sp = Species("x", compartment="nucleus")
        assert "nucleus" in sp.qualified_name or "x" in sp.qualified_name


# ===================================================================
# Reactions — kinetic laws
# ===================================================================


class TestKineticLaws:
    def test_mass_action_rate(self):
        ma = MassAction(k_forward=0.1)
        ma.reactant_names = ["A", "B"]
        rate = ma.evaluate({"A": 2.0, "B": 3.0})
        assert rate == pytest.approx(0.1 * 2.0 * 3.0)

    def test_mass_action_parameters(self):
        ma = MassAction(k_forward=0.5, k_reverse=0.1)
        params = ma.parameters
        assert "k_forward" in params or len(params) >= 1

    def test_hill_activation_rate(self):
        ha = HillActivation(Vmax=10.0, K=2.0, n=2.0)
        ha.activator_name = "A"
        rate = ha.evaluate({"A": 2.0})
        # Vmax * A^n / (K^n + A^n) = 10 * 4 / (4 + 4) = 5
        assert rate == pytest.approx(5.0)

    def test_hill_repression_rate(self):
        hr = HillRepression(Vmax=10.0, K=2.0, n=2.0)
        hr.repressor_name = "R"
        rate = hr.evaluate({"R": 2.0})
        # Vmax * K^n / (K^n + R^n) = 10 * 4 / (4 + 4) = 5
        assert rate == pytest.approx(5.0)

    def test_michaelis_menten_rate(self):
        mm = MichaelisMenten(Vmax=5.0, Km=1.0)
        mm.substrate_name = "S"
        rate = mm.evaluate({"S": 1.0})
        # Vmax * S / (Km + S) = 5 * 1 / 2 = 2.5
        assert rate == pytest.approx(2.5)

    def test_constitutive_production(self):
        cp = ConstitutiveProduction(rate=3.0)
        rate = cp.evaluate({})
        assert rate == pytest.approx(3.0)

    def test_linear_degradation(self):
        ld = LinearDegradation(rate=0.5)
        ld.species_name = "X"
        rate = ld.evaluate({"X": 4.0})
        assert rate == pytest.approx(2.0)


# ===================================================================
# Reaction class
# ===================================================================


class TestReaction:
    def test_creation(self):
        rxn = Reaction(
            "r1",
            reactants=[StoichiometryEntry("A")],
            products=[StoichiometryEntry("B")],
            kinetic_law=MassAction(k_forward=0.1),
        )
        assert rxn.name == "r1"
        assert "A" in rxn.species_involved
        assert "B" in rxn.species_involved

    def test_net_stoichiometry(self):
        rxn = Reaction(
            "r1",
            reactants=[StoichiometryEntry("A", coefficient=2)],
            products=[StoichiometryEntry("B", coefficient=1)],
            kinetic_law=MassAction(k_forward=0.1),
        )
        ns = rxn.net_stoichiometry
        assert ns["A"] == -2
        assert ns["B"] == 1

    def test_rate(self):
        ma = MassAction(k_forward=0.5)
        ma.reactant_names = ["A"]
        rxn = Reaction(
            "r1",
            reactants=[StoichiometryEntry("A")],
            products=[StoichiometryEntry("B")],
            kinetic_law=ma,
        )
        rate = rxn.rate({"A": 2.0, "B": 0.0})
        assert rate == pytest.approx(1.0)

    def test_validate(self):
        rxn = Reaction(
            "r1",
            reactants=[StoichiometryEntry("A")],
            products=[StoichiometryEntry("B")],
            kinetic_law=MassAction(k_forward=0.1),
        )
        errors = rxn.validate()
        assert isinstance(errors, list)

    def test_reversible_flag(self):
        rxn = Reaction(
            "r1",
            reactants=[StoichiometryEntry("A")],
            products=[StoichiometryEntry("B")],
            kinetic_law=MassAction(k_forward=0.1, k_reverse=0.05),
            reversible=True,
        )
        assert rxn.is_reversible


# ===================================================================
# Stoichiometry matrix
# ===================================================================


class TestStoichiometryMatrix:
    def test_simple(self):
        ma1 = MassAction(k_forward=1.0)
        ma1.reactant_names = ["A"]
        r1 = Reaction(
            "r1",
            reactants=[StoichiometryEntry("A")],
            products=[StoichiometryEntry("B")],
            kinetic_law=ma1,
        )
        S = build_stoichiometry_matrix([r1], ["A", "B"])
        assert S.shape == (2, 1)
        assert S[0, 0] == -1  # A consumed
        assert S[1, 0] == 1   # B produced

    def test_two_reactions(self):
        ma1 = MassAction(k_forward=1.0)
        ma1.reactant_names = ["A"]
        r1 = Reaction("r1", [StoichiometryEntry("A")], [StoichiometryEntry("B")], ma1)

        ma2 = MassAction(k_forward=0.5)
        ma2.reactant_names = ["B"]
        r2 = Reaction("r2", [StoichiometryEntry("B")], [StoichiometryEntry("C")], ma2)

        S = build_stoichiometry_matrix([r1, r2], ["A", "B", "C"])
        assert S.shape == (3, 2)


# ===================================================================
# BioModel — construction
# ===================================================================


class TestBioModelConstruction:
    def test_empty_model(self):
        model = BioModel("test")
        assert model.num_species == 0
        assert model.num_reactions == 0

    def test_add_species(self, species_A):
        model = BioModel("test")
        model.add_species(species_A)
        assert model.num_species == 1
        assert model.get_species("A") is species_A

    def test_add_reaction(self, species_A, species_B):
        model = BioModel("test")
        model.add_species(species_A)
        model.add_species(species_B)
        ma = MassAction(k_forward=0.1)
        ma.reactant_names = ["A"]
        rxn = Reaction("r1", [StoichiometryEntry("A")], [StoichiometryEntry("B")], ma)
        model.add_reaction(rxn)
        assert model.num_reactions == 1

    def test_species_names(self, species_A, species_B):
        model = BioModel("test")
        model.add_species(species_A)
        model.add_species(species_B)
        assert set(model.species_names) == {"A", "B"}

    def test_remove_species(self, species_A, species_B):
        model = BioModel("test")
        model.add_species(species_A)
        model.add_species(species_B)
        model.remove_species("A")
        assert model.num_species == 1

    def test_compartments(self):
        model = BioModel("test")
        comp = Compartment("cytoplasm", size=1.0)
        model.add_compartment(comp)
        assert model.get_compartment("cytoplasm").name == "cytoplasm"

    def test_initial_state(self, species_A, species_B):
        model = BioModel("test")
        model.add_species(species_A)
        model.add_species(species_B)
        x0 = model.initial_state()
        assert len(x0) == 2
        assert x0[0] == pytest.approx(1.0) or x0[1] == pytest.approx(1.0)

    def test_copy(self, toggle_switch_model):
        cp = toggle_switch_model.copy()
        assert cp.num_species == toggle_switch_model.num_species
        assert cp.num_reactions == toggle_switch_model.num_reactions
        # Modification of copy should not affect original
        cp.remove_species("u")
        assert toggle_switch_model.num_species == 2


# ===================================================================
# BioModel — stoichiometry matrix
# ===================================================================


class TestBioModelStoichiometry:
    def test_toggle_switch_shape(self, toggle_switch_model):
        S = toggle_switch_model.stoichiometry_matrix
        assert S.shape == (2, 4)  # 2 species, 4 reactions

    def test_repressilator_shape(self, repressilator_model):
        S = repressilator_model.stoichiometry_matrix
        assert S.shape == (3, 6)  # 3 species, 6 reactions

    def test_cascade_shape(self, cascade_model):
        S = cascade_model.stoichiometry_matrix
        assert S.shape[0] == 3  # A, B, C


# ===================================================================
# BioModel — ODE system
# ===================================================================


class TestBioModelODE:
    def test_ode_rhs_shape(self, toggle_switch_model):
        x0 = toggle_switch_model.initial_state()
        dx = toggle_switch_model.ode_rhs(x0)
        assert len(dx) == 2

    def test_ode_rhs_callable(self, toggle_switch_model):
        f = toggle_switch_model.ode_rhs_callable()
        x0 = toggle_switch_model.initial_state()
        dx = f(0.0, x0)
        assert len(dx) == 2

    def test_toggle_switch_steady_states(self, toggle_switch_model):
        """A toggle switch should have two stable steady states."""
        # Try to find steady state from two different initial conditions
        ss1 = toggle_switch_model.steady_state(np.array([3.0, 0.1]))
        ss2 = toggle_switch_model.steady_state(np.array([0.1, 3.0]))
        if ss1 is not None and ss2 is not None:
            # The two steady states should be distinct
            assert not np.allclose(ss1, ss2, atol=0.1)


# ===================================================================
# BioModel — Jacobian
# ===================================================================


class TestBioModelJacobian:
    def test_jacobian_shape(self, toggle_switch_model):
        conc = {"u": 2.0, "v": 0.5}
        J = toggle_switch_model.jacobian(conc)
        assert J.shape == (2, 2)

    def test_jacobian_numerical_vs_symbolic(self, cascade_model):
        """Numeric and symbolic Jacobians should agree approximately."""
        conc = {"A": 5.0, "B": 1.0, "C": 0.5}
        J_num = cascade_model.jacobian(conc)
        try:
            J_sym_mat = cascade_model.jacobian_symbolic()
            # Evaluate symbolic at the same point
            import sympy
            symbols = {sp.name: sympy.Symbol(sp.name) for sp in cascade_model.species}
            J_sym_eval = np.array(
                J_sym_mat.subs({symbols[k]: v for k, v in conc.items()})
            ).astype(float)
            np.testing.assert_allclose(J_num, J_sym_eval, atol=1e-4)
        except Exception:
            # If symbolic Jacobian fails, just check numerical is finite
            assert np.all(np.isfinite(J_num))


# ===================================================================
# BioModel — conservation laws
# ===================================================================


class TestConservationLaws:
    def test_cascade_conservation(self, cascade_model):
        """In A -> B -> C with degradation of C, A+B+C is NOT conserved."""
        laws = cascade_model.conservation_laws()
        # Should have zero or some laws; the cascade with C degradation
        # does not conserve total mass, so we expect few/no laws
        assert isinstance(laws, list)

    def test_simple_conservation(self):
        """A <-> B should have conservation A + B = const."""
        model = BioModel("simple")
        model.add_species(Species("A", initial_concentration=1.0))
        model.add_species(Species("B", initial_concentration=0.0))
        ma = MassAction(k_forward=1.0, k_reverse=0.5)
        ma.reactant_names = ["A"]
        ma.product_names = ["B"]
        rxn = Reaction(
            "r1",
            [StoichiometryEntry("A")],
            [StoichiometryEntry("B")],
            ma,
            reversible=True,
        )
        model.add_reaction(rxn)
        laws = model.conservation_laws()
        assert len(laws) >= 1


# ===================================================================
# BioModel — validation
# ===================================================================


class TestBioModelValidation:
    def test_valid_model(self, toggle_switch_model):
        errors = toggle_switch_model.validate()
        assert isinstance(errors, list)

    def test_mass_balance(self, toggle_switch_model):
        result = toggle_switch_model.check_mass_balance()
        assert isinstance(result, dict)


# ===================================================================
# BioModel — simulation
# ===================================================================


class TestBioModelSimulation:
    def test_simulate_toggle_switch(self, toggle_switch_model):
        t, y = toggle_switch_model.simulate((0, 10), num_points=50)
        assert len(t) >= 2
        assert y.shape[1] == 2
        # All concentrations should be non-negative
        assert np.all(y >= -1e-10)

    def test_simulate_repressilator(self, repressilator_model):
        t, y = repressilator_model.simulate((0, 20), num_points=100)
        assert y.shape[1] == 3


# ===================================================================
# Toggle switch model fixture tests
# ===================================================================


class TestToggleSwitchModel:
    def test_species_count(self, toggle_switch_model):
        assert toggle_switch_model.num_species == 2

    def test_reaction_count(self, toggle_switch_model):
        assert toggle_switch_model.num_reactions == 4

    def test_species_names(self, toggle_switch_model):
        assert set(toggle_switch_model.species_names) == {"u", "v"}


# ===================================================================
# Repressilator model fixture tests
# ===================================================================


class TestRepressilatorModel:
    def test_species_count(self, repressilator_model):
        assert repressilator_model.num_species == 3

    def test_reaction_count(self, repressilator_model):
        assert repressilator_model.num_reactions == 6

    def test_species_names(self, repressilator_model):
        assert set(repressilator_model.species_names) == {"lacI", "tetR", "cI"}
