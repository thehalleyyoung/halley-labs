"""Tests for the formal composition theorem proof engine."""

import pytest
import numpy as np


class TestFormalProofEngine:
    """Test the Z3-based formal proof verification."""

    def _make_engine(self):
        from causalbound.composition.formal_proof import FormalProofEngine
        return FormalProofEngine(timeout_ms=10000)

    def test_single_instance_all_verified(self):
        """All obligations should verify for a well-formed instance."""
        engine = self._make_engine()
        result = engine.verify_composition_theorem(
            n_subgraphs=3,
            n_separators=2,
            max_separator_size=3,
            lipschitz_constant=2.0,
            discretization=0.01,
            subgraph_lower_bounds=[0.2, 0.3, 0.25],
            subgraph_upper_bounds=[0.6, 0.7, 0.65],
        )
        assert result.all_verified, (
            f"Not all verified: {[o.obligation_id + ':' + o.status for o in result.obligations]}"
        )
        assert result.validity_verified
        assert result.gap_bound_verified
        assert result.certificate_hash

    def test_gap_bound_correct(self):
        """Gap bound should equal m * L * s * eps."""
        engine = self._make_engine()
        result = engine.verify_composition_theorem(
            n_subgraphs=2,
            n_separators=1,
            max_separator_size=4,
            lipschitz_constant=5.0,
            discretization=0.05,
            subgraph_lower_bounds=[0.1, 0.2],
            subgraph_upper_bounds=[0.8, 0.9],
        )
        expected_gap = 1 * 5.0 * 4 * 0.05
        assert abs(result.details["gap_bound"] - expected_gap) < 1e-10

    def test_restriction_soundness_lemma(self):
        """Lemma 1 (restriction soundness) should verify."""
        engine = self._make_engine()
        ob = engine._verify_restriction_soundness(3, 2, 3)
        assert ob.status == "verified", f"Lemma 1 failed: {ob.z3_result}"

    def test_local_containment_lemma(self):
        """Lemma 2 should verify for well-ordered bounds."""
        engine = self._make_engine()
        ob = engine._verify_local_bound_containment(
            [0.1, 0.2], [0.8, 0.9]
        )
        assert ob.status == "verified"

    def test_lipschitz_error_lemma(self):
        """Lemma 4 (Lipschitz gap) should verify."""
        engine = self._make_engine()
        ob = engine._verify_lipschitz_error_propagation(
            K=3, m=2, s=4, L_const=3.0, eps=0.01,
            lower_bounds=[0.1, 0.2, 0.15],
            upper_bounds=[0.8, 0.9, 0.85],
        )
        assert ob.status == "verified"

    def test_monotone_fixed_point_lemma(self):
        """Lemma 5 (monotone operator) should verify."""
        engine = self._make_engine()
        ob = engine._verify_monotone_fixed_point(
            [0.1, 0.2], [0.8, 0.9]
        )
        assert ob.status == "verified"

    def test_global_validity_lemma(self):
        """Main theorem should verify."""
        engine = self._make_engine()
        ob = engine._verify_global_validity(
            K=2, m=1, s=3, L_const=2.0, eps=0.01,
            lower_bounds=[0.3, 0.4],
            upper_bounds=[0.7, 0.6],
        )
        assert ob.status == "verified"

    def test_random_instances_high_verification_rate(self):
        """Random instances should achieve high verification rate."""
        engine = self._make_engine()
        stats = engine.verify_random_instances(
            n_instances=20,
            max_subgraphs=5,
            max_separators=4,
            max_sep_size=4,
            seed=42,
        )
        assert stats["verification_rate"] >= 0.9, (
            f"Verification rate {stats['verification_rate']:.2f} < 0.9"
        )

    def test_proof_result_summary(self):
        """Summary should be well-formed."""
        engine = self._make_engine()
        result = engine.verify_composition_theorem(
            n_subgraphs=2, n_separators=1,
            max_separator_size=2,
            lipschitz_constant=1.0,
            discretization=0.01,
            subgraph_lower_bounds=[0.3],
            subgraph_upper_bounds=[0.7],
        )
        summary = result.summary()
        assert "Formal proof" in summary
        assert "obligations verified" in summary

    def test_edge_case_single_subgraph(self):
        """Should handle single subgraph (trivial decomposition)."""
        engine = self._make_engine()
        result = engine.verify_composition_theorem(
            n_subgraphs=1, n_separators=0,
            max_separator_size=0,
            lipschitz_constant=1.0,
            discretization=0.01,
            subgraph_lower_bounds=[0.4],
            subgraph_upper_bounds=[0.6],
        )
        assert result.details["gap_bound"] == 0.0

    def test_interval_arithmetic_soundness(self):
        """Interval arithmetic proof should verify."""
        engine = self._make_engine()
        ob = engine.verify_interval_arithmetic_soundness(
            float_lower=0.199999, float_upper=0.700001,
            exact_lower_num=1, exact_lower_den=5,
            exact_upper_num=7, exact_upper_den=10,
            ulp_bound=1e-5,
        )
        assert ob.status == "verified"

    def test_discretization_composition_coupling(self):
        """Discretization-composition coupling should verify."""
        engine = self._make_engine()
        ob = engine.verify_discretization_composition_coupling(
            n_variables=5,
            per_variable_tv_bounds=[0.05, 0.03, 0.04, 0.06, 0.02],
            lipschitz_constant=2.0,
            n_separators=3,
        )
        assert ob.status == "verified"

    def test_parametric_verification(self):
        """Parametric gap bound should verify for all parameter ranges."""
        engine = self._make_engine()
        ob = engine.verify_parametric()
        assert ob.status == "verified"
