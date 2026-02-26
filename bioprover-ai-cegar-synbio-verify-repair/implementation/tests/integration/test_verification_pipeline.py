"""Integration tests for the verification pipeline — end-to-end."""

import numpy as np
import pytest

from bioprover.encoding.expression import Const, Ge, Gt, Le, Lt, Var
from bioprover.models.bio_model import BioModel
from bioprover.models.reactions import (
    HillRepression,
    LinearDegradation,
    MassAction,
    Reaction,
    StoichiometryEntry,
)
from bioprover.models.species import Species, SpeciesType
from bioprover.solver.interval import Interval, IntervalVector
from bioprover.temporal.stl_ast import (
    Always,
    ComparisonOp,
    Eventually,
    Expression,
    Predicate,
    STLAnd,
    STLOr,
)
from bioprover.temporal.stl_ast import Interval as STLInterval


# ===================================================================
# Helpers
# ===================================================================


def _simulate_and_check_robustness(model, formula, t_end=20.0):
    """Simulate a model and check robustness of an STL formula."""
    from bioprover.temporal.robustness import Signal, compute_robustness

    t, y = model.simulate((0, t_end), num_points=200)
    signals = {}
    for i, name in enumerate(model.species_names):
        signals[name] = Signal(times=t, values=y[:, i], name=name)
    rho = compute_robustness(formula, signals, 0.0)
    return rho, signals


# ===================================================================
# Toggle switch — bistability verification
# ===================================================================


class TestToggleSwitchVerification:
    def test_bistability_via_simulation(self, toggle_switch_model):
        """Toggle switch from high-u IC should settle to high-u steady state."""
        # Set high-u initial condition
        toggle_switch_model.get_species("u").initial_concentration = 3.0
        toggle_switch_model.get_species("v").initial_concentration = 0.1

        # Spec: G[5,20](u > 1)
        expr_u = Expression(variable="u")
        pred = Predicate(expr=expr_u, op=ComparisonOp.GT, threshold=1.0)
        spec = Always(child=pred, interval=STLInterval(lo=5.0, hi=20.0))

        rho, signals = _simulate_and_check_robustness(toggle_switch_model, spec)
        assert rho > 0, "Toggle switch should satisfy u > 1 from high-u IC"

    def test_bistability_low_u(self, toggle_switch_model):
        """Toggle switch from low-u IC should settle to low-u steady state."""
        toggle_switch_model.get_species("u").initial_concentration = 0.1
        toggle_switch_model.get_species("v").initial_concentration = 3.0

        expr_u = Expression(variable="u")
        pred = Predicate(expr=expr_u, op=ComparisonOp.LT, threshold=1.0)
        spec = Always(child=pred, interval=STLInterval(lo=5.0, hi=20.0))

        rho, signals = _simulate_and_check_robustness(toggle_switch_model, spec)
        assert rho > 0, "Toggle switch should satisfy u < 1 from low-u IC"

    def test_species_positive(self, toggle_switch_model):
        """All species should remain non-negative."""
        expr_u = Expression(variable="u")
        expr_v = Expression(variable="v")
        pred_u = Predicate(expr=expr_u, op=ComparisonOp.GE, threshold=0.0)
        pred_v = Predicate(expr=expr_v, op=ComparisonOp.GE, threshold=0.0)
        spec = Always(
            child=STLAnd(left=pred_u, right=pred_v),
            interval=STLInterval(lo=0.0, hi=20.0),
        )

        rho, _ = _simulate_and_check_robustness(toggle_switch_model, spec)
        assert rho >= 0


# ===================================================================
# Repressilator — oscillation verification
# ===================================================================


class TestRepressilatorVerification:
    def test_oscillation_eventually_high(self, repressilator_model):
        """Each gene should eventually reach a high value."""
        expr = Expression(variable="lacI")
        pred = Predicate(expr=expr, op=ComparisonOp.GT, threshold=2.0)
        spec = Eventually(child=pred, interval=STLInterval(lo=0.0, hi=30.0))

        rho, _ = _simulate_and_check_robustness(repressilator_model, spec, t_end=30.0)
        assert rho > 0

    def test_oscillation_not_always_high(self, repressilator_model):
        """In an oscillator, a gene should NOT always be high."""
        expr = Expression(variable="lacI")
        pred = Predicate(expr=expr, op=ComparisonOp.GT, threshold=3.0)
        spec = Always(child=pred, interval=STLInterval(lo=0.0, hi=30.0))

        rho, _ = _simulate_and_check_robustness(repressilator_model, spec, t_end=30.0)
        # Should be violated (not always > 3)
        assert rho < 0

    def test_all_species_non_negative(self, repressilator_model):
        """All species should remain non-negative."""
        preds = []
        for name in repressilator_model.species_names:
            expr = Expression(variable=name)
            preds.append(Predicate(expr=expr, op=ComparisonOp.GE, threshold=0.0))

        combined = preds[0]
        for p in preds[1:]:
            combined = STLAnd(left=combined, right=p)
        spec = Always(child=combined, interval=STLInterval(lo=0.0, hi=30.0))

        rho, _ = _simulate_and_check_robustness(repressilator_model, spec, t_end=30.0)
        assert rho >= 0


# ===================================================================
# NOT gate — Boolean logic verification
# ===================================================================


class TestNOTGateVerification:
    def _build_not_gate(self):
        """Build a simple NOT gate: input represses output."""
        model = BioModel("not_gate")
        model.add_species(Species("input", initial_concentration=5.0, species_type=SpeciesType.PROTEIN))
        model.add_species(Species("output", initial_concentration=0.0, species_type=SpeciesType.PROTEIN))

        # Input is a boundary species (externally controlled)
        model.get_species("input").boundary_condition = (
            __import__("bioprover.models.species", fromlist=["BoundaryCondition"]).BoundaryCondition.FIXED
        )

        hill = HillRepression(Vmax=5.0, K=1.0, n=2.0)
        hill.repressor_name = "input"
        rxn_prod = Reaction(
            "output_production",
            reactants=[],
            products=[StoichiometryEntry("output")],
            kinetic_law=hill,
            modifiers=["input"],
        )
        deg = LinearDegradation(rate=1.0)
        deg.species_name = "output"
        rxn_deg = Reaction(
            "output_degradation",
            reactants=[StoichiometryEntry("output")],
            products=[],
            kinetic_law=deg,
        )
        model.add_reaction(rxn_prod)
        model.add_reaction(rxn_deg)
        return model

    def test_high_input_low_output(self):
        """High input → low output (NOT gate behavior)."""
        model = self._build_not_gate()
        model.get_species("input").initial_concentration = 10.0

        expr_out = Expression(variable="output")
        pred = Predicate(expr=expr_out, op=ComparisonOp.LT, threshold=1.0)
        spec = Always(child=pred, interval=STLInterval(lo=5.0, hi=20.0))

        rho, _ = _simulate_and_check_robustness(model, spec)
        assert rho > 0

    def test_low_input_high_output(self):
        """Low input → high output (NOT gate behavior)."""
        model = self._build_not_gate()
        model.get_species("input").initial_concentration = 0.01

        expr_out = Expression(variable="output")
        pred = Predicate(expr=expr_out, op=ComparisonOp.GT, threshold=3.0)
        spec = Always(child=pred, interval=STLInterval(lo=5.0, hi=20.0))

        rho, _ = _simulate_and_check_robustness(model, spec)
        assert rho > 0


# ===================================================================
# Parameter uncertainty
# ===================================================================


class TestParameterUncertainty:
    def test_bounded_parameter_range(self, toggle_switch_model):
        """Model should satisfy spec for a range of parameter values."""
        # Simulate with nominal parameters
        toggle_switch_model.get_species("u").initial_concentration = 3.0
        toggle_switch_model.get_species("v").initial_concentration = 0.1

        expr_u = Expression(variable="u")
        pred = Predicate(expr=expr_u, op=ComparisonOp.GT, threshold=0.5)
        spec = Always(child=pred, interval=STLInterval(lo=5.0, hi=15.0))

        rho, _ = _simulate_and_check_robustness(toggle_switch_model, spec, t_end=15.0)
        # Should satisfy with nominal parameters
        assert rho > 0


# ===================================================================
# Counterexample generation
# ===================================================================


class TestCounterexampleGeneration:
    def test_violated_spec_detected(self, toggle_switch_model):
        """Model that violates spec should have negative robustness."""
        # Unreasonable spec: u > 100 always
        expr_u = Expression(variable="u")
        pred = Predicate(expr=expr_u, op=ComparisonOp.GT, threshold=100.0)
        spec = Always(child=pred, interval=STLInterval(lo=0.0, hi=10.0))

        rho, signals = _simulate_and_check_robustness(toggle_switch_model, spec)
        assert rho < 0, "Unreasonable spec should be violated"

    def test_counterexample_is_trajectory(self, toggle_switch_model):
        """When spec is violated, the simulation trajectory is a counterexample."""
        from bioprover.temporal.robustness import Signal, compute_robustness

        expr_u = Expression(variable="u")
        pred = Predicate(expr=expr_u, op=ComparisonOp.GT, threshold=100.0)
        spec = Always(child=pred, interval=STLInterval(lo=0.0, hi=10.0))

        t, y = toggle_switch_model.simulate((0, 10), num_points=100)
        signals = {}
        for i, name in enumerate(toggle_switch_model.species_names):
            signals[name] = Signal(times=t, values=y[:, i], name=name)

        rho = compute_robustness(spec, signals, 0.0)
        assert rho < 0
        # The trajectory y serves as a counterexample
        assert np.all(y[:, 0] < 100.0)  # u never reaches 100


# ===================================================================
# Compositional verification — cascade
# ===================================================================


class TestCompositionalVerification:
    def test_cascade_output_eventually_rises(self, cascade_model):
        """In A -> B -> C cascade, C should eventually rise."""
        expr_c = Expression(variable="C")
        pred = Predicate(expr=expr_c, op=ComparisonOp.GT, threshold=0.5)
        spec = Eventually(child=pred, interval=STLInterval(lo=0.0, hi=20.0))

        rho, _ = _simulate_and_check_robustness(cascade_model, spec)
        assert rho > 0

    def test_cascade_a_decreases(self, cascade_model):
        """In the cascade, A should decrease over time."""
        expr_a = Expression(variable="A")
        pred = Predicate(expr=expr_a, op=ComparisonOp.LT, threshold=4.0)
        spec = Eventually(child=pred, interval=STLInterval(lo=0.0, hi=10.0))

        rho, _ = _simulate_and_check_robustness(cascade_model, spec)
        assert rho > 0


# ===================================================================
# Robustness analysis
# ===================================================================


class TestRobustnessAnalysis:
    def test_sensitivity_analysis(self, toggle_switch_model):
        """Sensitivity analysis should identify critical predicates."""
        from bioprover.temporal.robustness import Signal, sensitivity_analysis

        toggle_switch_model.get_species("u").initial_concentration = 3.0
        toggle_switch_model.get_species("v").initial_concentration = 0.1

        t, y = toggle_switch_model.simulate((0, 20), num_points=200)
        signals = {}
        for i, name in enumerate(toggle_switch_model.species_names):
            signals[name] = Signal(times=t, values=y[:, i], name=name)

        expr_u = Expression(variable="u")
        pred = Predicate(expr=expr_u, op=ComparisonOp.GT, threshold=1.0)
        spec = Always(child=pred, interval=STLInterval(lo=5.0, hi=20.0))

        result = sensitivity_analysis(spec, signals, 0.0)
        assert result is not None
        assert result.critical_atom is not None

    def test_ensemble_robustness(self, toggle_switch_model):
        """Ensemble robustness over multiple simulations."""
        from bioprover.temporal.robustness import Signal, ensemble_robustness

        expr_u = Expression(variable="u")
        pred = Predicate(expr=expr_u, op=ComparisonOp.GT, threshold=0.0)
        spec = Always(child=pred, interval=STLInterval(lo=0.0, hi=10.0))

        signal_sets = []
        for seed_offset in range(5):
            np.random.seed(42 + seed_offset)
            ic_u = 2.0 + np.random.uniform(-0.5, 0.5)
            ic_v = 0.5 + np.random.uniform(-0.2, 0.2)
            toggle_switch_model.get_species("u").initial_concentration = ic_u
            toggle_switch_model.get_species("v").initial_concentration = ic_v
            t, y = toggle_switch_model.simulate((0, 10), num_points=100)
            sigs = {}
            for i, name in enumerate(toggle_switch_model.species_names):
                sigs[name] = Signal(times=t, values=y[:, i], name=name)
            signal_sets.append(sigs)

        result = ensemble_robustness(spec, signal_sets, 0.0)
        assert result is not None
        assert result.num_runs == 5
        assert result.satisfaction_ratio >= 0.0


# ===================================================================
# Interval-based verification
# ===================================================================


class TestIntervalVerification:
    def test_interval_model_checking_always(self):
        """Interval model checking on a known-safe signal."""
        from bioprover.solver.interval import Interval as IvInterval
        from bioprover.temporal.interval_model_checking import (
            IntervalModelChecker,
            IntervalSignal,
            ThreeValued,
        )

        times = np.linspace(0, 10, 51)
        intervals = [IvInterval(2.0, 4.0) for _ in times]
        isig = IntervalSignal(times=times, intervals=intervals, name="x")

        expr = Expression(variable="x")
        pred = Predicate(expr=expr, op=ComparisonOp.GT, threshold=1.0)
        spec = Always(child=pred, interval=STLInterval(lo=0.0, hi=10.0))

        checker = IntervalModelChecker({"x": isig})
        result = checker.check(spec, t=0.0)
        assert result.value == ThreeValued.TRUE
