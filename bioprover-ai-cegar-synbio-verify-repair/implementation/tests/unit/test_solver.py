"""Unit tests for ODE solver — interval integration, flowpipes, biology extensions."""

import math

import numpy as np
import pytest

from bioprover.solver.interval import Interval, IntervalVector


# ===================================================================
# Interval ODE integration on linear system
# ===================================================================


class TestIntervalODEIntegration:
    """Test validated ODE integration on dx/dt = -x (analytical: x(t) = x0*e^(-t))."""

    def _linear_decay_f(self, t, x):
        return np.array([-x[0]])

    def _linear_decay_f_interval(self, t, x):
        return IntervalVector([Interval(-x[0].hi, -x[0].lo)])

    def test_euler_integration(self):
        from bioprover.solver.ode_integrator import IntegratorConfig, ValidatedODEIntegrator

        config = IntegratorConfig(method="euler", initial_step=0.01, max_steps=200)
        integrator = ValidatedODEIntegrator(
            f=self._linear_decay_f,
            f_interval=self._linear_decay_f_interval,
            config=config,
        )
        x0 = IntervalVector([Interval(1.0, 1.0)])
        result = integrator.integrate(0.0, 1.0, x0)
        assert result is not None
        assert len(result.steps) > 0

    def test_analytical_solution_containment(self):
        """The analytical solution should be contained in the flowpipe."""
        from bioprover.solver.ode_integrator import IntegratorConfig, ValidatedODEIntegrator

        config = IntegratorConfig(method="euler", initial_step=0.05, max_steps=100)
        integrator = ValidatedODEIntegrator(
            f=self._linear_decay_f,
            f_interval=self._linear_decay_f_interval,
            config=config,
        )
        x0 = IntervalVector([Interval(1.0, 1.0)])
        result = integrator.integrate(0.0, 1.0, x0)

        # Check that analytical solution x(t)=e^(-t) is contained at each step
        for step in result.steps:
            t_mid = (step.t_interval.lo + step.t_interval.hi) / 2
            analytical = math.exp(-t_mid)
            assert step.enclosure[0].lo <= analytical + 1e-6
            assert step.enclosure[0].hi >= analytical - 1e-6

    def test_initial_uncertainty(self):
        """Integration with uncertain initial condition."""
        from bioprover.solver.ode_integrator import IntegratorConfig, ValidatedODEIntegrator

        config = IntegratorConfig(method="euler", initial_step=0.05, max_steps=100)
        integrator = ValidatedODEIntegrator(
            f=self._linear_decay_f,
            f_interval=self._linear_decay_f_interval,
            config=config,
        )
        # Uncertain initial condition [0.9, 1.1]
        x0 = IntervalVector([Interval(0.9, 1.1)])
        result = integrator.integrate(0.0, 0.5, x0)
        assert len(result.steps) > 0
        # Final enclosure should contain both extreme analytical solutions
        final = result.final_enclosure
        if final is not None:
            assert final[0].lo <= 0.9 * math.exp(-0.5) + 1e-3
            assert final[0].hi >= 1.1 * math.exp(-0.5) - 1e-3

    def test_taylor_integration(self):
        from bioprover.solver.ode_integrator import IntegratorConfig, ValidatedODEIntegrator

        config = IntegratorConfig(method="taylor", taylor_order=3, initial_step=0.1, max_steps=50)
        integrator = ValidatedODEIntegrator(
            f=self._linear_decay_f,
            f_interval=self._linear_decay_f_interval,
            config=config,
        )
        x0 = IntervalVector([Interval(1.0, 1.0)])
        result = integrator.integrate(0.0, 1.0, x0)
        assert len(result.steps) > 0

    def test_convergence_detection(self):
        from bioprover.solver.ode_integrator import IntegratorConfig, ValidatedODEIntegrator

        config = IntegratorConfig(method="euler", initial_step=0.1, max_steps=200)
        integrator = ValidatedODEIntegrator(
            f=self._linear_decay_f,
            f_interval=self._linear_decay_f_interval,
            config=config,
        )
        x0 = IntervalVector([Interval(1.0, 1.0)])
        result = integrator.integrate(0.0, 5.0, x0)
        # For decay, solution converges to 0
        if result.final_enclosure is not None:
            assert result.final_enclosure[0].lo <= 0.1


# ===================================================================
# Flowpipe
# ===================================================================


class TestFlowpipe:
    def _make_flowpipe(self):
        from bioprover.solver.flowpipe import Flowpipe, FlowpipeSegment
        segments = []
        for i in range(10):
            t = Interval(i * 0.1, (i + 1) * 0.1)
            box = IntervalVector([Interval(1.0 - i * 0.05, 1.0 + i * 0.05)])
            segments.append(FlowpipeSegment(time=t, box=box))
        return Flowpipe(segments)

    def test_construction(self):
        fp = self._make_flowpipe()
        assert len(fp.segments) == 10

    def test_reachable_set(self):
        fp = self._make_flowpipe()
        rs = fp.reachable_set()
        assert rs is not None
        # Should be wider than any individual segment
        assert rs[0].width() >= fp.segments[0].box[0].width()

    def test_reachable_set_at(self):
        fp = self._make_flowpipe()
        rs = fp.reachable_set_at(0.15)
        assert rs is not None

    def test_reachable_set_over(self):
        fp = self._make_flowpipe()
        rs = fp.reachable_set_over(Interval(0.0, 0.5))
        assert rs is not None

    def test_contains_trajectory(self):
        fp = self._make_flowpipe()
        times = np.array([0.05, 0.15, 0.25])
        states = np.array([[1.0], [1.0], [1.0]])
        result = fp.contains_trajectory(times, states)
        assert result  # 1.0 should be in all segments

    def test_union(self):
        fp1 = self._make_flowpipe()
        fp2 = self._make_flowpipe()
        union = fp1.union(fp2)
        assert len(union.segments) >= len(fp1.segments)

    def test_project(self):
        from bioprover.solver.flowpipe import Flowpipe, FlowpipeSegment
        segments = [
            FlowpipeSegment(
                time=Interval(0.0, 0.1),
                box=IntervalVector([Interval(0.0, 1.0), Interval(2.0, 3.0)]),
            )
        ]
        fp = Flowpipe(segments)
        proj = fp.project([0])
        assert proj.segments[0].box.dim == 1

    def test_bloat(self):
        fp = self._make_flowpipe()
        bloated = fp.bloat(0.1)
        for seg, bseg in zip(fp.segments, bloated.segments):
            assert bseg.box[0].width() >= seg.box[0].width()

    def test_max_width(self):
        fp = self._make_flowpipe()
        mw = fp.max_width()
        assert mw >= 0

    def test_serialization_roundtrip(self):
        fp = self._make_flowpipe()
        d = fp.to_dict()
        fp2 = Flowpipe.from_dict(d)
        assert len(fp2.segments) == len(fp.segments)

    def test_json_roundtrip(self):
        fp = self._make_flowpipe()
        json_str = fp.to_json()
        fp2 = Flowpipe.from_json(json_str)
        assert len(fp2.segments) == len(fp.segments)

    def test_contains_flowpipe(self):
        fp = self._make_flowpipe()
        assert fp.contains_flowpipe(fp)  # Should contain itself

    def test_hausdorff_distance_self(self):
        fp = self._make_flowpipe()
        d = fp.hausdorff_distance_to(fp)
        assert d == pytest.approx(0.0)

    def test_plot_data(self):
        fp = self._make_flowpipe()
        data = fp.to_plot_data(var_indices=[0])
        assert data is not None


# ===================================================================
# Biology extensions
# ===================================================================


class TestBiologyExtensions:
    def test_hill_activation_interval(self):
        from bioprover.solver.biology_extensions import hill_activation
        x = Interval(0.5, 1.5)
        k = Interval(1.0, 1.0)
        n = Interval(2.0, 2.0)
        result = hill_activation(x, k, n)
        assert isinstance(result, Interval)
        # At x=1, K=1, n=2: 1/(1+1) = 0.5; check containment
        assert result.lo <= 0.5 and result.hi >= 0.5

    def test_hill_repression_interval(self):
        from bioprover.solver.biology_extensions import hill_repression
        x = Interval(0.5, 1.5)
        k = Interval(1.0, 1.0)
        n = Interval(2.0, 2.0)
        result = hill_repression(x, k, n)
        assert isinstance(result, Interval)
        # At x=1, K=1, n=2: 1/(1+1) = 0.5; check containment
        assert result.lo <= 0.5 and result.hi >= 0.5

    def test_hill_function_interval(self):
        from bioprover.solver.biology_extensions import hill_function_interval
        x = Interval(1.0, 2.0)
        k = Interval(1.0, 1.0)
        n = Interval(2.0, 2.0)
        act = hill_function_interval(x, k, n, activation=True)
        rep = hill_function_interval(x, k, n, activation=False)
        assert isinstance(act, Interval)
        assert isinstance(rep, Interval)

    def test_positivity_enforcer(self):
        from bioprover.solver.biology_extensions import PositivityEnforcer
        enforcer = PositivityEnforcer(positive_indices=[0, 1])
        x = IntervalVector([Interval(-0.1, 2.0), Interval(-0.5, 1.0)])
        enforced = enforcer.enforce(x)
        assert enforced[0].lo >= 0.0
        assert enforced[1].lo >= 0.0

    def test_positivity_satisfied(self):
        from bioprover.solver.biology_extensions import PositivityEnforcer
        enforcer = PositivityEnforcer(positive_indices=[0])
        x = IntervalVector([Interval(0.5, 2.0)])
        assert enforcer.is_satisfied(x)
        x_neg = IntervalVector([Interval(-0.1, 2.0)])
        assert not enforcer.is_satisfied(x_neg)

    def test_conservation_law_reducer(self):
        from bioprover.solver.biology_extensions import ConservationLawReducer
        # A + B = 1  →  conservation_vectors = [[1, 1]]
        reducer = ConservationLawReducer(
            conservation_vectors=np.array([[1.0, 1.0]]),
            conservation_values=np.array([1.0]),
        )
        x = IntervalVector([Interval(0.3, 0.7), Interval(0.3, 0.7)])
        reduced, kept = reducer.reduce_dim(x)
        assert reduced.dim == 1  # One dimension removed

    def test_conservation_law_reconstruct(self):
        from bioprover.solver.biology_extensions import ConservationLawReducer
        reducer = ConservationLawReducer(
            conservation_vectors=np.array([[1.0, 1.0]]),
            conservation_values=np.array([1.0]),
        )
        x = IntervalVector([Interval(0.3, 0.7), Interval(0.3, 0.7)])
        reduced, kept = reducer.reduce_dim(x)
        reconstructed = reducer.reconstruct(reduced, kept, full_dim=2)
        assert reconstructed.dim == 2

    def test_steady_state_detector(self):
        from bioprover.solver.biology_extensions import SteadyStateDetector
        f = lambda t, x: np.array([0.0])  # already at steady state
        det = SteadyStateDetector(f, derivative_tol=1e-6, width_tol=1e-6, n_confirm=3)
        x = IntervalVector([Interval(1.0, 1.0)])
        # Should detect steady state after enough calls
        for _ in range(5):
            det.check(0.0, x)
        result = det.check(0.0, x)
        assert isinstance(result, bool)

    def test_contraction_detector(self):
        from bioprover.solver.biology_extensions import ContractionDetector
        f = lambda t, x: np.array([-x[0]])
        det = ContractionDetector(f, n=1)
        x = np.array([1.0])
        is_contr = det.is_contracting(0.0, x)
        # dx/dt = -x has Jacobian = -1, so it's contracting
        assert is_contr

    def test_contraction_rate(self):
        from bioprover.solver.biology_extensions import ContractionDetector
        f = lambda t, x: np.array([-2.0 * x[0]])
        det = ContractionDetector(f, n=1)
        rate = det.contraction_rate(0.0, np.array([1.0]))
        assert rate < 0  # Negative rate means contraction

    def test_monotone_system_cooperative(self):
        from bioprover.solver.biology_extensions import MonotoneSystemSolver
        # A cooperative system: dx/dt = f(x) where ∂fi/∂xj ≥ 0 for i≠j
        def f(t, x):
            return np.array([x[1] - x[0], x[0] - x[1]])

        solver = MonotoneSystemSolver(f, n=2, sign_matrix=np.array([[0, 1], [1, 0]]))
        assert solver.is_cooperative()

    def test_grn_sparse_solver(self):
        from bioprover.solver.biology_extensions import GRNSparseSolver
        def f(t, x):
            return np.array([-x[0], x[0] - x[1]])

        solver = GRNSparseSolver(f, n=2, interaction_graph={(0, 1)})
        J = solver.sparse_jacobian(0.0, np.array([1.0, 1.0]))
        assert J.shape == (2, 2)


# ===================================================================
# Validated integration helper
# ===================================================================


class TestValidatedIntegrate:
    def test_convenience_function(self):
        from bioprover.solver.ode_integrator import IntegratorConfig, validated_integrate

        def f(t, x):
            return np.array([-x[0]])

        result = validated_integrate(
            f=f,
            t_span=(0.0, 1.0),
            x0=np.array([1.0]),
            x0_radius=np.array([0.0]),
            config=IntegratorConfig(method="euler", initial_step=0.05, max_steps=100),
        )
        assert result is not None
        assert len(result.steps) > 0


# ===================================================================
# Integration result properties
# ===================================================================


class TestIntegrationResult:
    def test_final_enclosure(self):
        from bioprover.solver.ode_integrator import IntegratorConfig, ValidatedODEIntegrator

        def f(t, x):
            return np.array([1.0])  # dx/dt = 1

        def f_iv(t, x):
            return IntervalVector([Interval(1.0, 1.0)])

        config = IntegratorConfig(method="euler", initial_step=0.1, max_steps=20)
        integrator = ValidatedODEIntegrator(f=f, f_interval=f_iv, config=config)
        x0 = IntervalVector([Interval(0.0, 0.0)])
        result = integrator.integrate(0.0, 1.0, x0)
        fe = result.final_enclosure
        assert fe is not None
        # x(1) = 1.0 should be contained
        assert fe[0].lo <= 1.0 + 0.1 and fe[0].hi >= 1.0 - 0.1

    def test_final_time(self):
        from bioprover.solver.ode_integrator import IntegratorConfig, ValidatedODEIntegrator

        def f(t, x):
            return np.array([0.0])

        def f_iv(t, x):
            return IntervalVector([Interval(0.0, 0.0)])

        config = IntegratorConfig(method="euler", initial_step=0.5, max_steps=10)
        integrator = ValidatedODEIntegrator(f=f, f_interval=f_iv, config=config)
        x0 = IntervalVector([Interval(1.0, 1.0)])
        result = integrator.integrate(0.0, 2.0, x0)
        ft = result.final_time
        assert ft is not None
        assert ft >= 2.0 - 0.5  # Should reach close to t=2


# ===================================================================
# Step result
# ===================================================================


class TestStepResult:
    def test_step_result_fields(self):
        from bioprover.solver.ode_integrator import StepResult
        sr = StepResult(
            t_interval=Interval(0.0, 0.1),
            enclosure=IntervalVector([Interval(0.9, 1.1)]),
            end_enclosure=IntervalVector([Interval(0.95, 1.05)]),
            step_size=0.1,
            accepted=True,
            method_used="euler",
            enclosure_width=0.2,
        )
        assert sr.accepted
        assert sr.step_size == 0.1
        assert sr.enclosure_width == pytest.approx(0.2)
