"""Tests for CSL model checker, error certification, and models."""

import numpy as np
import pytest

from tn_check.tensor.mps import MPS, ones_mps, threshold_mps, characteristic_mps
from tn_check.tensor.mpo import MPO, identity_mpo
from tn_check.tensor.operations import (
    mps_inner_product, mps_to_dense, mpo_to_dense, mps_compress,
    mps_hadamard_product, mps_total_probability,
)
from tn_check.checker.csl_ast import (
    AtomicProp, TrueFormula, Negation, Conjunction,
    ProbabilityOp, BoundedUntil, UnboundedUntil,
    ComparisonOp, LinearPredicate, parse_csl,
)
from tn_check.checker.satisfaction import (
    compute_satisfaction_set, ThreeValued, SatisfactionResult,
    _compute_linear_predicate_mps,
)
from tn_check.error.certification import (
    ErrorCertificate, ErrorTracker, nonneg_preserving_round,
    clamping_error_bound,
)
from tn_check.error.propagation import (
    semigroup_error_bound, csl_error_propagation,
)


class TestCSLParsing:
    """Test CSL formula parsing."""

    def test_parse_atomic(self):
        f = parse_csl("X_2 >= 5")
        assert isinstance(f, AtomicProp)
        assert f.species_index == 2
        assert f.threshold == 5
        assert f.direction == "greater_equal"

    def test_parse_atomic_less(self):
        f = parse_csl("X_0 < 10")
        assert isinstance(f, AtomicProp)
        assert f.direction == "less"

    def test_parse_true(self):
        f = parse_csl("true")
        assert isinstance(f, TrueFormula)

    def test_parse_probability_bounded_until(self):
        f = parse_csl("P>=0.9 [true U<=100 X_1 >= 5]")
        assert isinstance(f, ProbabilityOp)
        assert f.threshold == 0.9
        assert f.comparison == ComparisonOp.GEQ
        path = f.path_formula
        assert isinstance(path, BoundedUntil)
        assert path.time_bound == 100.0

    def test_parse_eventually(self):
        f = parse_csl("P>=0.99 [F<=3600 X_0 >= 10]")
        assert isinstance(f, ProbabilityOp)
        path = f.path_formula
        assert isinstance(path, BoundedUntil)
        assert isinstance(path.phi1, TrueFormula)
        assert path.time_bound == 3600.0

    def test_parse_conjunction(self):
        f = parse_csl("X_0 >= 5 & X_1 < 10")
        assert isinstance(f, Conjunction)

    def test_parse_negation(self):
        f = parse_csl("!X_0 >= 5")
        assert isinstance(f, Negation)


class TestSatisfactionSets:
    """Test satisfaction-set computation."""

    def test_true_formula(self):
        sat = compute_satisfaction_set(TrueFormula(), 3, 5)
        v = mps_to_dense(sat)
        assert np.allclose(v, 1.0)

    def test_atomic_threshold(self):
        # X_0 >= 3 with d=5 species
        sat = compute_satisfaction_set(
            AtomicProp(0, 3, "greater_equal"), 2, 5,
        )
        v = mps_to_dense(sat)
        # States (3,*) and (4,*) should be 1, rest 0
        v2d = v.reshape(5, 5)
        assert np.allclose(v2d[:3, :], 0.0)
        assert np.allclose(v2d[3:, :], 1.0)

    def test_conjunction(self):
        # X_0 >= 3 AND X_1 < 2
        formula = Conjunction(
            AtomicProp(0, 3, "greater_equal"),
            AtomicProp(1, 2, "less"),
        )
        sat = compute_satisfaction_set(formula, 2, 5)
        v = mps_to_dense(sat).reshape(5, 5)
        # Should be 1 only where X_0 >= 3 AND X_1 < 2
        for i in range(5):
            for j in range(5):
                expected = 1.0 if (i >= 3 and j < 2) else 0.0
                assert abs(v[i, j] - expected) < 1e-10, \
                    f"Mismatch at ({i},{j}): got {v[i,j]}, expected {expected}"

    def test_negation(self):
        formula = Negation(AtomicProp(0, 3, "greater_equal"))
        sat = compute_satisfaction_set(formula, 2, 5)
        v = mps_to_dense(sat).reshape(5, 5)
        for i in range(5):
            for j in range(5):
                expected = 0.0 if i >= 3 else 1.0
                assert abs(v[i, j] - expected) < 1e-10

    def test_linear_predicate(self):
        # X_0 + X_1 >= 3 with d=4
        pred = LinearPredicate(
            coefficients=((0, 1.0), (1, 1.0)),
            threshold=3.0,
            direction="greater_equal",
        )
        sat = _compute_linear_predicate_mps(pred, 2, 4)
        v = mps_to_dense(sat).reshape(4, 4)
        for i in range(4):
            for j in range(4):
                expected = 1.0 if (i + j >= 3) else 0.0
                assert abs(v[i, j] - expected) < 1e-10, \
                    f"Linear pred mismatch at ({i},{j}): got {v[i,j]}, expected {expected}"

    def test_satisfaction_rank1_for_atomic(self):
        """Axis-aligned atomic propositions should produce rank-1 TT."""
        sat = compute_satisfaction_set(
            AtomicProp(1, 5, "greater_equal"), 4, 10,
        )
        assert sat.max_bond_dim == 1


class TestErrorCertification:
    """Test error bounds and certification."""

    def test_clamping_error_bound(self):
        """Proposition 1: clamping error ≤ 2 * truncation error."""
        from tn_check.tensor.mps import random_mps
        mps = random_mps(3, 4, 5, seed=42)
        bound = clamping_error_bound(mps, 0.01)
        assert bound == 0.02  # 2 * epsilon

    def test_error_certificate_total(self):
        cert = ErrorCertificate(
            truncation_error=0.01,
            clamping_error=0.005,
            fsp_error=0.001,
        )
        total = cert.compute_total()
        # clamping_bound = min(0.005, 2*0.01) = 0.005
        # total = 0.01 + 0.005 + 0.001 = 0.016
        assert abs(total - 0.016) < 1e-10

    def test_error_tracker(self):
        tracker = ErrorTracker()
        tracker.record_step(0, 0.001, [5, 5])
        tracker.record_step(1, 0.002, [5, 5])
        tracker.record_clamping(1, 0.0005)

        assert abs(tracker.accumulated_truncation_error() - 0.003) < 1e-10
        assert abs(tracker.accumulated_clamping_error() - 0.0005) < 1e-10

        cert = tracker.certify()
        assert cert.truncation_error == 0.003
        assert cert.clamping_error == 0.0005

    def test_semigroup_error_bound_metzler(self):
        """Theorem 1: linear error accumulation for Metzler generators."""
        errors = [0.001] * 100
        analysis = semigroup_error_bound(errors, time_horizon=10.0, is_metzler=True)
        assert analysis.contractivity_factor == 1.0
        assert abs(analysis.accumulated_error - 0.1) < 1e-10
        # With clamping: 2 * accumulated
        assert abs(analysis.amplification_bound - 0.2) < 1e-10

    def test_semigroup_error_bound_non_metzler(self):
        """Non-Metzler: exponential amplification."""
        errors = [0.001] * 10
        analysis = semigroup_error_bound(errors, time_horizon=1.0, is_metzler=False)
        assert analysis.contractivity_factor > 1.0

    def test_nonneg_preserving_round(self):
        """Non-negativity-preserving rounding should produce non-negative MPS."""
        # Create an MPS with some negative entries
        cores = [
            np.array([[[1.0], [0.5], [-0.1], [0.3]]]),  # has negative
            np.array([[[0.2], [0.4], [0.1], [0.3]]]),
        ]
        mps = MPS(cores, copy_cores=True)
        result, error, cert = nonneg_preserving_round(mps, max_bond_dim=10)
        # Check result is non-negative
        v = mps_to_dense(result)
        assert np.all(v >= -1e-14), f"Found negatives: {v[v < 0]}"

    def test_csl_error_propagation(self):
        result = csl_error_propagation(
            inner_error=0.01,
            threshold=0.9,
            comparison=">=",
            num_states_near_threshold=5,
            total_states=100,
        )
        assert result["sound"] is True
        assert result["indeterminate_width"] == 0.02


class TestThreeValuedSemantics:
    """Test three-valued satisfaction classification."""

    def test_definite_true(self):
        result = SatisfactionResult(
            satisfaction_mps=ones_mps(2, 3),
            probability_lower=0.95,
            probability_upper=0.98,
        )
        v = result.classify(0.9, ">=", epsilon=0.01)
        assert v == ThreeValued.TRUE

    def test_definite_false(self):
        result = SatisfactionResult(
            satisfaction_mps=ones_mps(2, 3),
            probability_lower=0.3,
            probability_upper=0.5,
        )
        v = result.classify(0.9, ">=", epsilon=0.01)
        assert v == ThreeValued.FALSE

    def test_indeterminate(self):
        result = SatisfactionResult(
            satisfaction_mps=ones_mps(2, 3),
            probability_lower=0.89,
            probability_upper=0.91,
        )
        v = result.classify(0.9, ">=", epsilon=0.01)
        assert v == ThreeValued.INDETERMINATE


class TestModels:
    """Test model library."""

    def test_birth_death(self):
        from tn_check.models.library import birth_death
        model = birth_death(birth_rate=1.0, death_rate=0.1, max_copy=20)
        assert len(model.species) == 1
        assert len(model.reactions) == 2
        assert model.name == "birth_death"

    def test_toggle_switch(self):
        from tn_check.models.library import toggle_switch
        model = toggle_switch(max_copy=30)
        assert len(model.species) == 2
        assert len(model.reactions) == 4

    def test_repressilator(self):
        from tn_check.models.library import repressilator
        model = repressilator(n_genes=3, max_copy=20)
        assert len(model.species) == 3
        assert len(model.reactions) == 6

    def test_cascade(self):
        from tn_check.models.library import cascade
        model = cascade(n_layers=5, max_copy=30)
        assert len(model.species) == 5

    def test_schlogl(self):
        from tn_check.models.library import schlogl
        model = schlogl(max_copy=100)
        assert len(model.species) == 1
        assert len(model.reactions) == 4


class TestOrdering:
    """Test species ordering strategies."""

    def test_identity(self):
        from tn_check.ordering import identity_ordering
        assert identity_ordering(5) == [0, 1, 2, 3, 4]

    def test_rcm(self):
        from tn_check.ordering import reverse_cuthill_mckee
        from tn_check.models.library import cascade
        model = cascade(n_layers=5)
        order = reverse_cuthill_mckee(model)
        assert sorted(order) == [0, 1, 2, 3, 4]

    def test_spectral(self):
        from tn_check.ordering import spectral_ordering
        from tn_check.models.library import cascade
        model = cascade(n_layers=5)
        order = spectral_ordering(model)
        assert sorted(order) == [0, 1, 2, 3, 4]

    def test_greedy(self):
        from tn_check.ordering import greedy_entanglement_ordering
        from tn_check.models.library import cascade
        model = cascade(n_layers=5)
        order = greedy_entanglement_ordering(model)
        assert sorted(order) == [0, 1, 2, 3, 4]


class TestAdaptive:
    """Test adaptive rank controller."""

    def test_initialize(self):
        from tn_check.adaptive import AdaptiveRankController
        ctrl = AdaptiveRankController()
        dims = ctrl.initialize(4)
        assert len(dims) == 4
        assert all(d == 10 for d in dims)

    def test_double_on_high_error(self):
        from tn_check.adaptive import AdaptiveRankController
        ctrl = AdaptiveRankController()
        ctrl.initialize(3)
        decision = ctrl.decide(truncation_error=0.1, target_error=0.01)
        assert decision.action == "double"
        assert all(d == 20 for d in decision.bond_dims)

    def test_keep_on_low_error(self):
        from tn_check.adaptive import AdaptiveRankController
        ctrl = AdaptiveRankController()
        ctrl.initialize(3)
        decision = ctrl.decide(truncation_error=0.005, target_error=0.01)
        assert decision.action == "keep"


class TestCertificateVerifier:
    """Test independent certificate verification."""

    def test_sound_trace_passes(self):
        """A correctly constructed trace should pass all checks."""
        from tn_check.verifier import VerificationTrace, CertificateVerifier

        trace = VerificationTrace(
            model_name="test_model",
            num_species=2,
            physical_dims=[5, 5],
            max_bond_dim=50,
        )
        # Record well-behaved steps
        for i in range(10):
            trace.record_step(
                step_index=i,
                time=i * 0.1,
                truncation_error=0.001,
                clamping_error=0.0005,  # < 2 * 0.001
                bond_dims=[5, 5],
                total_probability=1.0,
            )
        trace.record_fsp_bounds([5, 5], fsp_error_bound=0.001)
        trace.record_csl_check(
            formula_str="P>=0.9 [F<=10 X_0 >= 3]",
            probability_lower=0.92,
            probability_upper=0.95,
            verdict="true",
            total_certified_error=0.03,
        )
        trace.finalize()

        verifier = CertificateVerifier()
        report = verifier.verify(trace)
        assert report.overall_sound, report.summary()

    def test_clamping_violation_detected(self):
        """Clamping error exceeding 2*trunc should be flagged."""
        from tn_check.verifier import VerificationTrace, CertificateVerifier

        trace = VerificationTrace(model_name="bad_clamp")
        trace.record_step(
            step_index=0, time=0.1,
            truncation_error=0.001,
            clamping_error=0.01,  # > 2 * 0.001 = violation!
            bond_dims=[5],
            total_probability=1.0,
        )
        trace.finalize()

        verifier = CertificateVerifier()
        report = verifier.verify(trace)
        assert not report.overall_sound
        clamping_checks = [c for c in report.checks if c.check_name == "clamping_bound"]
        assert len(clamping_checks) == 1
        assert not clamping_checks[0].passed

    def test_probability_conservation_warning(self):
        """Large probability deviation should be flagged."""
        from tn_check.verifier import VerificationTrace, CertificateVerifier

        trace = VerificationTrace(model_name="bad_prob")
        trace.record_step(
            step_index=0, time=0.1,
            truncation_error=0.001,
            clamping_error=0.0,
            bond_dims=[5],
            total_probability=0.8,  # deviation from 1.0
        )
        trace.finalize()

        verifier = CertificateVerifier()
        report = verifier.verify(trace)
        prob_checks = [c for c in report.checks if c.check_name == "probability_conservation"]
        assert len(prob_checks) == 1
        assert not prob_checks[0].passed

    def test_json_roundtrip(self, tmp_path):
        """Trace should survive JSON serialization."""
        from tn_check.verifier import VerificationTrace

        trace = VerificationTrace(model_name="roundtrip_test", num_species=3)
        trace.record_step(0, 0.1, 0.001, 0.0005, [5, 5], 1.0)
        trace.record_fsp_bounds([10, 10, 10], 0.01)
        trace.finalize()

        path = str(tmp_path / "trace.json")
        trace.to_json(path)

        loaded = VerificationTrace.from_json(path)
        assert loaded.model_name == "roundtrip_test"
        assert loaded.num_species == 3
        assert len(loaded.steps) == 1
        assert loaded.fsp_bounds is not None
        assert abs(loaded.total_truncation_error - 0.001) < 1e-10

    def test_error_underestimate_detected(self):
        """Claimed error less than recomputed should fail."""
        from tn_check.verifier import VerificationTrace, CertificateVerifier

        trace = VerificationTrace(model_name="underestimate")
        trace.record_step(0, 0.1, 0.1, 0.05, [5], 1.0)
        trace.finalize()
        # Recomputed total = 0.1 + min(0.05, 0.2) + 0 = 0.15
        # Set claimed to something less than recomputed
        trace.total_certified_error = 0.001  # way too low

        verifier = CertificateVerifier()
        # Prevent re-finalize from overwriting our underestimate
        original_finalize = trace.finalize
        trace.finalize = lambda: None
        report = verifier.verify(trace)
        trace.finalize = original_finalize
        comp_checks = [c for c in report.checks if c.check_name == "error_composition"]
        assert len(comp_checks) == 1
        assert not comp_checks[0].passed


class TestSpectralGap:
    """Test spectral gap estimation."""

    def test_spectral_gap_estimate_structure(self):
        """SpectralGapEstimate should have expected fields."""
        from tn_check.checker.spectral import SpectralGapEstimate

        est = SpectralGapEstimate(
            gap_estimate=0.1,
            confidence="high",
            estimated_mixing_time=10.0,
            predicted_iterations=50,
            feasible=True,
            method="power_iteration",
        )
        assert est.predicted_iteration_count(1e-8) > 0
        assert est.feasible

    def test_adaptive_fallback_time(self):
        """Fallback time should scale inversely with gap."""
        from tn_check.checker.spectral import adaptive_fallback_time_bound

        t_fast = adaptive_fallback_time_bound(1.0)
        t_slow = adaptive_fallback_time_bound(0.001)
        assert t_slow > t_fast

    def test_zero_gap_gives_max_time(self):
        from tn_check.checker.spectral import adaptive_fallback_time_bound

        t = adaptive_fallback_time_bound(0.0)
        assert t == 100000.0

    def test_convergence_diagnostics_ratio(self):
        """ConvergenceDiagnostics should compute geometric ratio."""
        from tn_check.checker.model_checker import ConvergenceDiagnostics

        diag = ConvergenceDiagnostics()
        diag.iteration_errors = [1.0, 0.5, 0.25, 0.125, 0.0625]
        ratio = diag.geometric_convergence_ratio()
        assert ratio is not None
        assert abs(ratio - 0.5) < 0.01


class TestClampingBoundRigorous:
    """Rigorous tests for the clamping error bound (Proposition 1)."""

    def test_clamping_bound_holds_on_random_mps(self):
        """Verify Proposition 1 numerically: clamp_err ≤ trunc_err."""
        from tn_check.tensor.mps import random_mps
        from tn_check.tensor.decomposition import tensor_to_mps

        rng = np.random.default_rng(123)
        for trial in range(5):
            # Create a random probability vector
            d = 6
            n_sites = 2
            v = rng.dirichlet(np.ones(d**n_sites))
            p_exact = MPS(
                [tensor_to_mps(v, [d]*n_sites, max_bond_dim=100).cores[k]
                 for k in range(n_sites)],
                copy_cores=True,
            )

            # Truncate aggressively
            p_svd, trunc_err = mps_compress(p_exact, max_bond_dim=2, tolerance=1e-14)
            p_svd_dense = mps_to_dense(p_svd)

            # Clamp
            neg_mask = p_svd_dense < 0
            clamp_err = float(np.sum(np.abs(p_svd_dense[neg_mask])))

            # Proposition 1: clamp_err ≤ trunc_err (in L1 norm)
            # Note: trunc_err from mps_compress is Frobenius, not L1
            # For L1, we compute directly
            l1_trunc = float(np.sum(np.abs(v - p_svd_dense)))
            assert clamp_err <= l1_trunc + 1e-10, (
                f"Trial {trial}: clamp_err={clamp_err:.2e} > l1_trunc={l1_trunc:.2e}"
            )

    def test_nonneg_preserving_round_reduces_negativity(self):
        """NNTT rounding should reduce negative mass."""
        cores = [
            np.array([[[0.5], [0.3], [-0.2], [0.1]]]),
            np.array([[[0.3], [0.4], [0.2], [0.1]]]),
        ]
        mps = MPS(cores, copy_cores=True)
        v_before = mps_to_dense(mps)
        neg_before = float(np.sum(np.abs(v_before[v_before < 0])))

        result, error, cert = nonneg_preserving_round(mps, max_bond_dim=10)
        v_after = mps_to_dense(result)
        neg_after = float(np.sum(np.abs(v_after[v_after < 0])))

        assert neg_after <= neg_before + 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
