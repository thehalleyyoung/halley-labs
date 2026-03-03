"""
Comprehensive tests for dp_forge.certificates — dataclasses, generator,
verifier, chain, and serialization.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from dp_forge.certificates import (
    ApproximationCertificate,
    CertificateChain,
    CertificateGenerator,
    CertificateVerifier,
    ComposedCertificate,
    LPOptimalityCertificate,
    SDPOptimalityCertificate,
    from_json,
    to_json,
    to_latex,
    verify_chain,
)
from dp_forge.types import (
    ExtractedMechanism,
    LPStruct,
    QuerySpec,
    QueryType,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_lp_cert(
    *,
    dual_ub=None,
    dual_eq=None,
    duality_gap=1e-8,
    primal_obj=1.0,
    dual_obj=1.0 - 1e-8,
    primal_slack=None,
    dual_slack=None,
    n_vars=4,
    n_ub=6,
    n_eq=2,
    solver_name="highs",
    timestamp="2025-01-01T00:00:00+00:00",
):
    if dual_ub is None:
        dual_ub = np.array([0.1, 0.2, 0.3, 0.15, 0.05, 0.2])
    return LPOptimalityCertificate(
        dual_ub=dual_ub,
        dual_eq=dual_eq,
        duality_gap=duality_gap,
        primal_obj=primal_obj,
        dual_obj=dual_obj,
        primal_slack=primal_slack,
        dual_slack=dual_slack,
        n_vars=n_vars,
        n_ub=n_ub,
        n_eq=n_eq,
        solver_name=solver_name,
        timestamp=timestamp,
    )


def _make_sdp_cert(
    *,
    dim=3,
    duality_gap=1e-7,
    primal_obj=2.5,
    dual_obj=2.5 - 1e-7,
    solver_name="scs",
    timestamp="2025-01-01T00:00:00+00:00",
):
    dual_matrix = np.eye(dim) * 0.5
    return SDPOptimalityCertificate(
        dual_matrix=dual_matrix,
        duality_gap=duality_gap,
        primal_obj=primal_obj,
        dual_obj=dual_obj,
        solver_name=solver_name,
        timestamp=timestamp,
    )


def _make_query_spec(n=2, epsilon=1.0, k=5):
    return QuerySpec(
        query_values=np.arange(n, dtype=np.float64),
        domain=list(range(n)),
        sensitivity=1.0,
        epsilon=epsilon,
        k=k,
    )


def _make_dp_mechanism(n=2, k=5, epsilon=1.0):
    """Build a valid DP mechanism satisfying row-sum=1 and DP constraints."""
    p = np.ones((n, k), dtype=np.float64) / k
    return ExtractedMechanism(p_final=p)


# ═══════════════════════════════════════════════════════════════════════════
# §1  LPOptimalityCertificate
# ═══════════════════════════════════════════════════════════════════════════


class TestLPOptimalityCertificate:
    """Tests for the LPOptimalityCertificate dataclass."""

    def test_construction_basic(self):
        cert = _make_lp_cert()
        assert cert.n_vars == 4
        assert cert.n_ub == 6
        assert cert.n_eq == 2
        assert cert.solver_name == "highs"
        assert isinstance(cert.dual_ub, np.ndarray)

    def test_dual_ub_converted_to_ndarray(self):
        cert = _make_lp_cert(dual_ub=[1.0, 2.0, 3.0])
        assert isinstance(cert.dual_ub, np.ndarray)
        assert cert.dual_ub.dtype == np.float64

    def test_dual_eq_converted_to_ndarray(self):
        cert = _make_lp_cert(dual_eq=[0.5, -0.3])
        assert isinstance(cert.dual_eq, np.ndarray)
        assert cert.dual_eq.dtype == np.float64

    def test_dual_eq_none_stays_none(self):
        cert = _make_lp_cert(dual_eq=None)
        assert cert.dual_eq is None

    def test_primal_slack_converted(self):
        cert = _make_lp_cert(primal_slack=[0.1, 0.0, 0.2, 0.0, 0.3, 0.0])
        assert isinstance(cert.primal_slack, np.ndarray)
        assert cert.primal_slack.dtype == np.float64

    def test_dual_slack_converted(self):
        cert = _make_lp_cert(dual_slack=[0.0, 0.0, 0.0, 0.0])
        assert isinstance(cert.dual_slack, np.ndarray)

    def test_timestamp_auto_generated(self):
        cert = _make_lp_cert(timestamp="")
        assert len(cert.timestamp) > 0

    def test_timestamp_preserved_when_provided(self):
        ts = "2024-06-15T12:00:00+00:00"
        cert = _make_lp_cert(timestamp=ts)
        assert cert.timestamp == ts

    def test_relative_gap_with_unit_primal(self):
        cert = _make_lp_cert(primal_obj=1.0, duality_gap=0.01)
        assert cert.relative_gap == pytest.approx(0.01)

    def test_relative_gap_denominator_at_least_one(self):
        cert = _make_lp_cert(primal_obj=0.1, duality_gap=0.01)
        # denom = max(|0.1|, 1.0) = 1.0
        assert cert.relative_gap == pytest.approx(0.01)

    def test_relative_gap_large_primal(self):
        cert = _make_lp_cert(primal_obj=100.0, duality_gap=0.01)
        assert cert.relative_gap == pytest.approx(0.01 / 100.0)

    def test_relative_gap_negative_primal(self):
        cert = _make_lp_cert(primal_obj=-50.0, duality_gap=0.5)
        assert cert.relative_gap == pytest.approx(0.5 / 50.0)

    def test_is_tight_small_gap(self):
        cert = _make_lp_cert(primal_obj=10.0, duality_gap=1e-9)
        assert cert.is_tight is True

    def test_is_tight_large_gap(self):
        cert = _make_lp_cert(primal_obj=1.0, duality_gap=0.1)
        assert cert.is_tight is False

    def test_is_tight_boundary(self):
        # relative_gap = 1e-6 / max(|1.0|, 1.0) = 1e-6 -> exactly at threshold
        cert = _make_lp_cert(primal_obj=1.0, duality_gap=1e-6)
        assert cert.is_tight is True

    def test_is_tight_just_above_boundary(self):
        cert = _make_lp_cert(primal_obj=1.0, duality_gap=1.1e-6)
        assert cert.is_tight is False

    def test_zero_gap(self):
        cert = _make_lp_cert(primal_obj=5.0, dual_obj=5.0, duality_gap=0.0)
        assert cert.relative_gap == 0.0
        assert cert.is_tight is True

    def test_repr(self):
        cert = _make_lp_cert(duality_gap=1e-8, n_vars=4, n_ub=6, n_eq=2)
        r = repr(cert)
        assert "LPOptimalityCertificate" in r
        assert "gap=" in r
        assert "vars=4" in r


# ═══════════════════════════════════════════════════════════════════════════
# §2  SDPOptimalityCertificate
# ═══════════════════════════════════════════════════════════════════════════


class TestSDPOptimalityCertificate:
    """Tests for the SDPOptimalityCertificate dataclass."""

    def test_construction_identity(self):
        cert = _make_sdp_cert(dim=4)
        assert cert.matrix_dim == 4
        assert cert.dual_matrix.shape == (4, 4)
        assert cert.solver_name == "scs"

    def test_auto_matrix_dim(self):
        mat = np.eye(5) * 0.1
        cert = SDPOptimalityCertificate(
            dual_matrix=mat, duality_gap=0.0, primal_obj=1.0, dual_obj=1.0,
        )
        assert cert.matrix_dim == 5

    def test_auto_min_eigenvalue(self):
        mat = np.diag([0.5, 0.3, 0.1])
        cert = SDPOptimalityCertificate(
            dual_matrix=mat, duality_gap=0.0, primal_obj=1.0, dual_obj=1.0,
        )
        assert cert.min_eigenvalue == pytest.approx(0.1)

    def test_min_eigenvalue_psd(self):
        cert = _make_sdp_cert(dim=3)
        assert cert.is_psd is True

    def test_min_eigenvalue_not_psd(self):
        mat = np.diag([1.0, -0.5, 0.3])
        cert = SDPOptimalityCertificate(
            dual_matrix=mat, duality_gap=0.0, primal_obj=1.0, dual_obj=1.0,
        )
        assert cert.is_psd is False
        assert cert.min_eigenvalue == pytest.approx(-0.5)

    def test_non_square_matrix_raises(self):
        with pytest.raises(ValueError, match="square"):
            SDPOptimalityCertificate(
                dual_matrix=np.ones((2, 3)),
                duality_gap=0.0,
                primal_obj=1.0,
                dual_obj=1.0,
            )

    def test_1d_matrix_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            SDPOptimalityCertificate(
                dual_matrix=np.ones(5),
                duality_gap=0.0,
                primal_obj=1.0,
                dual_obj=1.0,
            )

    def test_relative_gap(self):
        cert = _make_sdp_cert(primal_obj=10.0, duality_gap=0.05)
        assert cert.relative_gap == pytest.approx(0.05 / 10.0)

    def test_is_tight_true(self):
        cert = _make_sdp_cert(primal_obj=100.0, duality_gap=1e-8)
        assert cert.is_tight is True

    def test_is_tight_false(self):
        cert = _make_sdp_cert(primal_obj=1.0, duality_gap=1.0)
        assert cert.is_tight is False

    def test_timestamp_auto(self):
        cert = SDPOptimalityCertificate(
            dual_matrix=np.eye(2), duality_gap=0.0,
            primal_obj=0.0, dual_obj=0.0,
        )
        assert len(cert.timestamp) > 0

    def test_repr(self):
        cert = _make_sdp_cert(dim=3)
        r = repr(cert)
        assert "SDPOptimalityCertificate" in r
        assert "dim=3" in r


# ═══════════════════════════════════════════════════════════════════════════
# §3  ComposedCertificate
# ═══════════════════════════════════════════════════════════════════════════


class TestComposedCertificate:
    """Tests for the ComposedCertificate dataclass."""

    def test_basic_composition(self):
        c1 = _make_lp_cert()
        c2 = _make_lp_cert()
        composed = ComposedCertificate(
            components=[(c1, 0.5, 0.0), (c2, 0.3, 0.0)],
            composition_type="basic",
        )
        assert composed.total_epsilon == pytest.approx(0.8)
        assert composed.total_delta == pytest.approx(0.0)
        assert composed.n_components == 2

    def test_basic_composition_with_delta(self):
        c1 = _make_lp_cert()
        composed = ComposedCertificate(
            components=[(c1, 1.0, 1e-5), (c1, 1.0, 1e-5)],
            composition_type="basic",
        )
        assert composed.total_epsilon == pytest.approx(2.0)
        assert composed.total_delta == pytest.approx(2e-5)

    def test_advanced_composition(self):
        c1 = _make_lp_cert()
        composed = ComposedCertificate(
            components=[(c1, 1.0, 1e-5), (c1, 1.0, 1e-5), (c1, 1.0, 1e-5)],
            composition_type="advanced",
        )
        assert composed.total_epsilon > 0
        assert composed.total_delta == pytest.approx(3e-5)
        # Verify the formula: eps_sq/(2*eps_sum) + sqrt(2*eps_sq*log(1/delta_sum))
        eps_sq = 3.0
        eps_sum = 3.0
        delta_sum = 3e-5
        expected = eps_sq / (2 * eps_sum) + math.sqrt(2 * eps_sq * math.log(1 / delta_sum))
        assert composed.total_epsilon == pytest.approx(expected)

    def test_empty_components(self):
        composed = ComposedCertificate(
            components=[], composition_type="advanced",
        )
        assert composed.n_components == 0
        assert composed.total_epsilon == 0.0
        assert composed.total_delta == 0.0

    def test_pre_set_total_epsilon(self):
        c1 = _make_lp_cert()
        composed = ComposedCertificate(
            components=[(c1, 1.0, 0.0)],
            composition_type="basic",
            total_epsilon=42.0,
            total_delta=0.01,
        )
        # When total_epsilon is pre-set and nonzero, it should be preserved
        assert composed.total_epsilon == pytest.approx(42.0)

    def test_timestamp_auto(self):
        composed = ComposedCertificate(components=[])
        assert len(composed.timestamp) > 0

    def test_repr(self):
        c1 = _make_lp_cert()
        composed = ComposedCertificate(
            components=[(c1, 0.5, 1e-5)], composition_type="basic",
        )
        r = repr(composed)
        assert "ComposedCertificate" in r
        assert "n=1" in r


# ═══════════════════════════════════════════════════════════════════════════
# §4  ApproximationCertificate
# ═══════════════════════════════════════════════════════════════════════════


class TestApproximationCertificate:
    """Tests for the ApproximationCertificate dataclass."""

    def test_construction(self):
        cert = ApproximationCertificate(
            k=100, grid_spacing=0.1,
            discretization_error_bound=0.01,
        )
        assert cert.k == 100
        assert cert.grid_spacing == pytest.approx(0.1)
        assert cert.sensitivity == 1.0
        assert cert.epsilon == 1.0

    def test_total_error_bound(self):
        cert = ApproximationCertificate(
            k=50, grid_spacing=0.2,
            discretization_error_bound=0.01,
            interpolation_error=0.005,
        )
        assert cert.total_error_bound == pytest.approx(0.015)

    def test_total_error_bound_no_interpolation(self):
        cert = ApproximationCertificate(
            k=50, grid_spacing=0.2,
            discretization_error_bound=0.01,
        )
        assert cert.total_error_bound == pytest.approx(0.01)

    def test_negative_disc_error_raises(self):
        with pytest.raises(ValueError, match="discretization_error_bound"):
            ApproximationCertificate(
                k=10, grid_spacing=0.1, discretization_error_bound=-0.1,
            )

    def test_negative_interp_error_raises(self):
        with pytest.raises(ValueError, match="interpolation_error"):
            ApproximationCertificate(
                k=10, grid_spacing=0.1,
                discretization_error_bound=0.01,
                interpolation_error=-0.01,
            )

    def test_from_spec(self):
        spec = _make_query_spec(n=3, epsilon=1.0, k=50)
        cert = ApproximationCertificate.from_spec(spec, k=50)
        assert cert.k == 50
        assert cert.sensitivity == pytest.approx(1.0)
        assert cert.epsilon == pytest.approx(1.0)
        assert cert.grid_spacing > 0
        assert cert.discretization_error_bound > 0
        # disc_error = (grid_spacing)^2 / 12
        assert cert.discretization_error_bound == pytest.approx(
            cert.grid_spacing ** 2 / 12.0
        )

    def test_timestamp_auto(self):
        cert = ApproximationCertificate(
            k=10, grid_spacing=0.5, discretization_error_bound=0.01,
        )
        assert len(cert.timestamp) > 0

    def test_repr(self):
        cert = ApproximationCertificate(
            k=100, grid_spacing=0.05, discretization_error_bound=0.001,
        )
        r = repr(cert)
        assert "ApproximationCertificate" in r
        assert "k=100" in r


# ═══════════════════════════════════════════════════════════════════════════
# §5  CertificateGenerator
# ═══════════════════════════════════════════════════════════════════════════


class TestCertificateGenerator:
    """Tests for the CertificateGenerator class."""

    def test_generate_from_dict(self):
        gen = CertificateGenerator(tol=1e-6)
        spec = _make_query_spec(n=2, epsilon=1.0, k=5)
        lp_result = {
            "fun": 1.5,
            "dual_ub": [0.1, 0.2, 0.3],
            "x": [0.2, 0.3, 0.25, 0.25],
        }
        cert = gen.generate(lp_result, spec)
        assert isinstance(cert, LPOptimalityCertificate)
        assert cert.primal_obj == pytest.approx(1.5)

    def test_generate_from_namespace(self):
        gen = CertificateGenerator()
        spec = _make_query_spec()
        result = SimpleNamespace(fun=2.0, dual_ub=np.array([0.1, 0.2]), x=np.array([0.5, 0.5]))
        cert = gen.generate(result, spec)
        assert cert.primal_obj == pytest.approx(2.0)
        assert isinstance(cert.dual_ub, np.ndarray)

    def test_generate_with_lp_struct(self):
        gen = CertificateGenerator()
        spec = _make_query_spec(n=2, k=3)
        n_vars = 6
        n_ub = 4
        n_eq = 2
        c = np.ones(n_vars)
        A_ub = sparse.csr_matrix(np.random.rand(n_ub, n_vars))
        b_ub = np.ones(n_ub) * 2.0
        A_eq = sparse.csr_matrix(np.ones((n_eq, n_vars)) / n_vars)
        b_eq = np.ones(n_eq)
        bounds = [(0.0, 1.0)] * n_vars
        var_map = {(i, j): i * 3 + j for i in range(2) for j in range(3)}
        y_grid = np.linspace(0, 1, 3)

        lp_struct = LPStruct(
            c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, var_map=var_map, y_grid=y_grid,
        )

        x = np.ones(n_vars) / n_vars
        lp_result = SimpleNamespace(
            fun=float(c @ x),
            dual_ub=np.abs(np.random.rand(n_ub)),
            dual_eq=np.random.rand(n_eq),
            x=x,
        )

        cert = gen.generate(lp_result, spec, lp_struct=lp_struct)
        assert cert.n_vars == n_vars
        assert cert.n_ub == n_ub
        assert cert.n_eq == n_eq
        assert cert.primal_slack is not None
        assert cert.dual_slack is not None

    def test_generate_no_primal_obj_raises(self):
        gen = CertificateGenerator()
        spec = _make_query_spec()
        with pytest.raises(Exception):
            gen.generate(object(), spec)

    def test_generate_extracts_solver_name_from_method(self):
        gen = CertificateGenerator()
        spec = _make_query_spec()
        result = SimpleNamespace(fun=1.0, dual_ub=np.array([0.1]), method="revised simplex")
        cert = gen.generate(result, spec)
        assert cert.solver_name == "revised simplex"

    def test_generate_extracts_solver_name_from_dict(self):
        gen = CertificateGenerator()
        spec = _make_query_spec()
        result = {"fun": 1.0, "dual_ub": [0.1], "solver": "glpk"}
        cert = gen.generate(result, spec)
        assert cert.solver_name == "glpk"

    def test_generate_fallback_solver_name(self):
        gen = CertificateGenerator()
        spec = _make_query_spec()
        result = SimpleNamespace(fun=1.0, dual_ub=np.array([]))
        cert = gen.generate(result, spec)
        assert cert.solver_name == "unknown"

    def test_generate_objective_value_key(self):
        gen = CertificateGenerator()
        spec = _make_query_spec()
        result = SimpleNamespace(objective_value=3.14, dual_ub=np.array([0.1]))
        cert = gen.generate(result, spec)
        assert cert.primal_obj == pytest.approx(3.14)

    def test_generate_dict_objective_value(self):
        gen = CertificateGenerator()
        spec = _make_query_spec()
        result = {"objective_value": 2.71}
        cert = gen.generate(result, spec)
        assert cert.primal_obj == pytest.approx(2.71)


# ═══════════════════════════════════════════════════════════════════════════
# §6  CertificateVerifier
# ═══════════════════════════════════════════════════════════════════════════


class TestCertificateVerifier:
    """Tests for the CertificateVerifier class."""

    def test_check_strong_duality_passes(self):
        verifier = CertificateVerifier(tol=1e-4)
        cert = _make_lp_cert(primal_obj=1.0, dual_obj=1.0, duality_gap=0.0)
        assert verifier.check_strong_duality(cert) is True

    def test_check_strong_duality_fails(self):
        verifier = CertificateVerifier(tol=1e-6)
        cert = _make_lp_cert(primal_obj=1.0, dual_obj=0.5, duality_gap=0.5)
        assert verifier.check_strong_duality(cert) is False

    def test_check_strong_duality_with_tolerance_override(self):
        verifier = CertificateVerifier(tol=1e-10)
        cert = _make_lp_cert(primal_obj=1.0, dual_obj=0.9999, duality_gap=1e-4)
        # Should fail with default tol=1e-10
        assert verifier.check_strong_duality(cert) is False
        # Should pass with relaxed tolerance
        assert verifier.check_strong_duality(cert, tol=1e-3) is True

    def test_check_dual_feasibility_lp_passes(self):
        verifier = CertificateVerifier()
        cert = _make_lp_cert(dual_ub=np.array([0.1, 0.2, 0.3]))
        assert verifier.check_dual_feasibility(cert) is True

    def test_check_dual_feasibility_lp_fails(self):
        verifier = CertificateVerifier(tol=1e-6)
        cert = _make_lp_cert(dual_ub=np.array([0.1, -1.0, 0.3]))
        assert verifier.check_dual_feasibility(cert) is False

    def test_check_dual_feasibility_lp_small_negative_ok(self):
        verifier = CertificateVerifier(tol=1e-4)
        cert = _make_lp_cert(dual_ub=np.array([0.1, -1e-5, 0.3]))
        assert verifier.check_dual_feasibility(cert) is True

    def test_check_dual_feasibility_empty_dual(self):
        verifier = CertificateVerifier()
        cert = _make_lp_cert(dual_ub=np.array([]))
        assert verifier.check_dual_feasibility(cert) is True

    def test_check_dual_feasibility_sdp_passes(self):
        verifier = CertificateVerifier()
        cert = _make_sdp_cert(dim=3)  # identity * 0.5 -> PSD
        assert verifier.check_dual_feasibility(cert) is True

    def test_check_dual_feasibility_sdp_fails(self):
        verifier = CertificateVerifier(tol=1e-6)
        mat = np.diag([1.0, -1.0, 0.5])
        cert = SDPOptimalityCertificate(
            dual_matrix=mat, duality_gap=0.0, primal_obj=1.0, dual_obj=1.0,
        )
        assert verifier.check_dual_feasibility(cert) is False

    def test_check_complementary_slackness_no_slack(self):
        verifier = CertificateVerifier()
        cert = _make_lp_cert(primal_slack=None)
        # When no slack is available, CS check is skipped -> True
        assert verifier.check_complementary_slackness(cert) is True

    def test_check_complementary_slackness_passes(self):
        verifier = CertificateVerifier(tol=1e-4)
        dual_ub = np.array([0.0, 0.5, 0.0, 0.3])
        primal_slack = np.array([1.0, 0.0, 0.5, 0.0])
        cert = _make_lp_cert(dual_ub=dual_ub, primal_slack=primal_slack)
        # Products: [0, 0, 0, 0] -> all zero -> passes
        assert verifier.check_complementary_slackness(cert) is True

    def test_check_complementary_slackness_fails(self):
        verifier = CertificateVerifier(tol=1e-6)
        dual_ub = np.array([0.5, 0.5])
        primal_slack = np.array([0.5, 0.5])
        cert = _make_lp_cert(dual_ub=dual_ub, primal_slack=primal_slack)
        # Products: [0.25, 0.25] -> max violation 0.25 > 1e-6
        assert verifier.check_complementary_slackness(cert) is False

    def test_check_complementary_slackness_sdp(self):
        verifier = CertificateVerifier(tol=1e-4)
        cert = _make_sdp_cert(primal_obj=1.0, duality_gap=1e-8)
        assert verifier.check_complementary_slackness(cert) is True

    def test_check_feasibility_valid_mechanism(self):
        verifier = CertificateVerifier(tol=1e-4)
        spec = _make_query_spec(n=2, epsilon=1.0, k=5)
        mech = _make_dp_mechanism(n=2, k=5, epsilon=1.0)
        cert = _make_lp_cert()
        assert verifier.check_feasibility(cert, mech, spec) is True

    def test_check_feasibility_negative_probs(self):
        verifier = CertificateVerifier(tol=1e-8)
        spec = _make_query_spec(n=2, epsilon=1.0, k=3)
        p = np.array([[0.5, 0.5, 0.0], [-0.1, 0.6, 0.5]])
        # ExtractedMechanism will reject negative probs, so test via verifier
        # directly with a mock
        mech = SimpleNamespace(p_final=p)
        cert = _make_lp_cert()
        assert verifier.check_feasibility(cert, mech, spec) is False

    def test_check_feasibility_row_sum_violation(self):
        verifier = CertificateVerifier(tol=1e-8)
        spec = _make_query_spec(n=2, epsilon=1.0, k=3)
        p = np.array([[0.5, 0.5, 0.1], [0.3, 0.3, 0.4]])
        mech = SimpleNamespace(p_final=p)
        cert = _make_lp_cert()
        assert verifier.check_feasibility(cert, mech, spec) is False

    def test_verify_all_checks_pass(self):
        verifier = CertificateVerifier(tol=1e-2)
        spec = _make_query_spec(n=2, epsilon=1.0, k=5)
        mech = _make_dp_mechanism(n=2, k=5, epsilon=1.0)
        cert = _make_lp_cert(
            primal_obj=1.0, dual_obj=1.0, duality_gap=0.0,
            dual_ub=np.array([0.1, 0.2]),
            primal_slack=np.array([0.0, 0.0]),
        )
        assert verifier.verify(cert, mech, spec) is True

    def test_verify_fails_strong_duality(self):
        verifier = CertificateVerifier(tol=1e-8)
        spec = _make_query_spec(n=2, epsilon=1.0, k=5)
        mech = _make_dp_mechanism(n=2, k=5)
        cert = _make_lp_cert(primal_obj=1.0, dual_obj=0.0, duality_gap=1.0)
        assert verifier.verify(cert, mech, spec) is False

    def test_verify_fails_dual_feasibility(self):
        verifier = CertificateVerifier(tol=1e-8)
        spec = _make_query_spec(n=2, epsilon=1.0, k=5)
        mech = _make_dp_mechanism(n=2, k=5)
        cert = _make_lp_cert(
            primal_obj=1.0, dual_obj=1.0, duality_gap=0.0,
            dual_ub=np.array([-10.0, 0.2]),
        )
        assert verifier.verify(cert, mech, spec) is False


# ═══════════════════════════════════════════════════════════════════════════
# §7  CertificateChain
# ═══════════════════════════════════════════════════════════════════════════


class TestCertificateChain:
    """Tests for the CertificateChain dataclass and verify_chain."""

    def test_construction(self):
        c1 = _make_lp_cert()
        c2 = _make_lp_cert()
        chain = CertificateChain(
            certificates=[c1, c2],
            epsilons=[1.0, 0.5],
            deltas=[0.0, 0.0],
        )
        assert chain.n_components == 2

    def test_total_epsilon_basic(self):
        c1 = _make_lp_cert()
        chain = CertificateChain(
            certificates=[c1, c1],
            epsilons=[1.0, 0.5],
            deltas=[0.0, 0.0],
            composition_type="basic",
        )
        assert chain.total_epsilon == pytest.approx(1.5)

    def test_total_delta(self):
        c1 = _make_lp_cert()
        chain = CertificateChain(
            certificates=[c1, c1],
            epsilons=[1.0, 1.0],
            deltas=[1e-5, 2e-5],
        )
        assert chain.total_delta == pytest.approx(3e-5)

    def test_max_duality_gap(self):
        c1 = _make_lp_cert(duality_gap=1e-8)
        c2 = _make_lp_cert(duality_gap=1e-5)
        chain = CertificateChain(
            certificates=[c1, c2],
            epsilons=[1.0, 1.0],
            deltas=[0.0, 0.0],
        )
        assert chain.max_duality_gap == pytest.approx(1e-5)

    def test_max_duality_gap_empty(self):
        chain = CertificateChain(
            certificates=[], epsilons=[], deltas=[],
        )
        assert chain.max_duality_gap == 0.0

    def test_mismatched_lengths_raises(self):
        c1 = _make_lp_cert()
        with pytest.raises(ValueError, match="same length"):
            CertificateChain(
                certificates=[c1, c1],
                epsilons=[1.0],
                deltas=[0.0, 0.0],
            )

    def test_mismatched_deltas_raises(self):
        c1 = _make_lp_cert()
        with pytest.raises(ValueError, match="same length"):
            CertificateChain(
                certificates=[c1],
                epsilons=[1.0],
                deltas=[0.0, 0.0],
            )

    def test_append(self):
        c1 = _make_lp_cert()
        chain = CertificateChain(
            certificates=[c1], epsilons=[1.0], deltas=[0.0],
        )
        c2 = _make_lp_cert(duality_gap=1e-3)
        chain.append(c2, epsilon=0.5, delta=1e-6)
        assert chain.n_components == 2
        assert chain.epsilons[-1] == 0.5
        assert chain.deltas[-1] == pytest.approx(1e-6)

    def test_repr(self):
        c1 = _make_lp_cert()
        chain = CertificateChain(
            certificates=[c1], epsilons=[1.0], deltas=[0.0],
        )
        r = repr(chain)
        assert "CertificateChain" in r
        assert "n=1" in r

    def test_advanced_composition_epsilon(self):
        c1 = _make_lp_cert()
        chain = CertificateChain(
            certificates=[c1, c1, c1],
            epsilons=[1.0, 1.0, 1.0],
            deltas=[1e-5, 1e-5, 1e-5],
            composition_type="advanced",
        )
        assert chain.total_epsilon > 0
        # Verify formula: sqrt(2*eps_sq*log(1/delta_sum)) + eps_sq/(2*eps_sum)
        eps_sq = 3.0
        eps_sum = 3.0
        delta_sum = 3e-5
        expected = math.sqrt(2 * eps_sq * math.log(1 / delta_sum)) + eps_sq / (2 * eps_sum)
        assert chain.total_epsilon == pytest.approx(expected)


class TestVerifyChain:
    """Tests for the verify_chain function."""

    def test_verify_chain_valid(self):
        spec = _make_query_spec(n=2, epsilon=1.0, k=5)
        mech = _make_dp_mechanism(n=2, k=5)
        cert = _make_lp_cert(
            primal_obj=1.0, dual_obj=1.0, duality_gap=0.0,
            dual_ub=np.array([0.1]),
            primal_slack=np.array([0.0]),
        )
        chain = CertificateChain(
            certificates=[cert], epsilons=[1.0], deltas=[0.0],
        )
        result = verify_chain(chain, [mech], [spec], tol=1e-2)
        assert result is True

    def test_verify_chain_mismatched_mechanisms(self):
        cert = _make_lp_cert()
        chain = CertificateChain(
            certificates=[cert, cert],
            epsilons=[1.0, 1.0],
            deltas=[0.0, 0.0],
        )
        mech = _make_dp_mechanism()
        spec = _make_query_spec()
        # 2 certs but 1 mechanism
        assert verify_chain(chain, [mech], [spec, spec]) is False

    def test_verify_chain_mismatched_specs(self):
        cert = _make_lp_cert()
        chain = CertificateChain(
            certificates=[cert, cert],
            epsilons=[1.0, 1.0],
            deltas=[0.0, 0.0],
        )
        mech = _make_dp_mechanism()
        spec = _make_query_spec()
        # 2 certs but 1 spec
        assert verify_chain(chain, [mech, mech], [spec]) is False


# ═══════════════════════════════════════════════════════════════════════════
# §8  Serialization: to_json / from_json
# ═══════════════════════════════════════════════════════════════════════════


class TestSerialization:
    """Tests for to_json and from_json round-trips."""

    def test_lp_cert_roundtrip(self):
        cert = _make_lp_cert(
            dual_ub=np.array([0.1, 0.2, 0.3]),
            dual_eq=np.array([0.5, -0.3]),
            primal_slack=np.array([0.0, 0.1, 0.0]),
            dual_slack=np.array([0.0, 0.0, 0.01, 0.0]),
        )
        js = to_json(cert)
        parsed = json.loads(js)
        assert parsed["type"] == "LPOptimalityCertificate"
        assert parsed["primal_obj"] == pytest.approx(cert.primal_obj)
        assert parsed["dual_obj"] == pytest.approx(cert.dual_obj)

        restored = from_json(js)
        assert isinstance(restored, LPOptimalityCertificate)
        np.testing.assert_allclose(restored.dual_ub, cert.dual_ub)
        assert restored.primal_obj == pytest.approx(cert.primal_obj)
        assert restored.duality_gap == pytest.approx(cert.duality_gap)

    def test_lp_cert_roundtrip_no_eq(self):
        cert = _make_lp_cert(dual_eq=None, primal_slack=None, dual_slack=None)
        js = to_json(cert)
        restored = from_json(js)
        assert isinstance(restored, LPOptimalityCertificate)
        assert restored.dual_eq is None
        assert restored.primal_slack is None

    def test_sdp_cert_roundtrip(self):
        cert = _make_sdp_cert(dim=3)
        js = to_json(cert)
        parsed = json.loads(js)
        assert parsed["type"] == "SDPOptimalityCertificate"

        restored = from_json(js)
        assert isinstance(restored, SDPOptimalityCertificate)
        np.testing.assert_allclose(restored.dual_matrix, cert.dual_matrix)
        assert restored.primal_obj == pytest.approx(cert.primal_obj)

    def test_approximation_cert_roundtrip(self):
        cert = ApproximationCertificate(
            k=100, grid_spacing=0.05,
            discretization_error_bound=0.001,
            sensitivity=1.0, epsilon=0.5,
            interpolation_error=0.0005,
        )
        js = to_json(cert)
        restored = from_json(js)
        assert isinstance(restored, ApproximationCertificate)
        assert restored.k == 100
        assert restored.grid_spacing == pytest.approx(0.05)
        assert restored.discretization_error_bound == pytest.approx(0.001)
        assert restored.interpolation_error == pytest.approx(0.0005)

    def test_composed_cert_roundtrip(self):
        c1 = _make_lp_cert(dual_ub=np.array([0.1]))
        c2 = _make_lp_cert(dual_ub=np.array([0.2]))
        composed = ComposedCertificate(
            components=[(c1, 1.0, 0.0), (c2, 0.5, 1e-5)],
            composition_type="basic",
        )
        js = to_json(composed)
        restored = from_json(js)
        assert isinstance(restored, ComposedCertificate)
        assert restored.n_components == 2
        assert restored.composition_type == "basic"

    def test_from_json_dict_input(self):
        cert = _make_lp_cert()
        js = to_json(cert)
        d = json.loads(js)
        restored = from_json(d)
        assert isinstance(restored, LPOptimalityCertificate)

    def test_from_json_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown certificate type"):
            from_json({"type": "FooBarCert"})

    def test_json_is_valid_json(self):
        cert = _make_lp_cert()
        js = to_json(cert)
        parsed = json.loads(js)
        assert isinstance(parsed, dict)

    def test_certificate_chain_serialization(self):
        c1 = _make_lp_cert(dual_ub=np.array([0.1]))
        chain = CertificateChain(
            certificates=[c1], epsilons=[1.0], deltas=[0.0],
        )
        js = to_json(chain)
        parsed = json.loads(js)
        assert parsed["type"] == "CertificateChain"
        assert parsed["n_components"] == 1
        assert "certificates" in parsed


# ═══════════════════════════════════════════════════════════════════════════
# §9  to_latex
# ═══════════════════════════════════════════════════════════════════════════


class TestToLatex:
    """Tests for the to_latex function."""

    def test_lp_cert_latex(self):
        cert = _make_lp_cert(
            primal_obj=1.5, dual_obj=1.5, duality_gap=1e-10,
            primal_slack=np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
        )
        latex = to_latex(cert)
        assert r"\begin{theorem}" in latex
        assert r"\end{proof}" in latex
        assert "LP" in latex or "primal" in latex.lower()

    def test_sdp_cert_latex(self):
        cert = _make_sdp_cert(dim=3)
        latex = to_latex(cert)
        assert r"\begin{theorem}" in latex
        assert "SDP" in latex or "dual" in latex.lower()

    def test_approximation_cert_latex(self):
        cert = ApproximationCertificate(
            k=100, grid_spacing=0.05,
            discretization_error_bound=0.001,
        )
        latex = to_latex(cert)
        assert r"\begin{theorem}" in latex
        assert "k = 100" in latex or "k=100" in latex

    def test_lp_tight_cert_mentions_strong_duality(self):
        cert = _make_lp_cert(primal_obj=1.0, dual_obj=1.0, duality_gap=1e-10)
        latex = to_latex(cert)
        assert "strong duality" in latex.lower() or "optimal" in latex.lower()

    def test_lp_not_tight_cert(self):
        cert = _make_lp_cert(primal_obj=1.0, dual_obj=0.5, duality_gap=0.5)
        latex = to_latex(cert)
        assert "near-optimal" in latex.lower() or "not tight" in latex.lower()

    def test_sdp_not_psd_latex(self):
        mat = np.diag([1.0, -1.0])
        cert = SDPOptimalityCertificate(
            dual_matrix=mat, duality_gap=0.0, primal_obj=1.0, dual_obj=1.0,
        )
        latex = to_latex(cert)
        assert "Warning" in latex or "not PSD" in latex or "invalid" in latex.lower()

    def test_approximation_with_interpolation_latex(self):
        cert = ApproximationCertificate(
            k=50, grid_spacing=0.1,
            discretization_error_bound=0.01,
            interpolation_error=0.005,
        )
        latex = to_latex(cert)
        assert "interpolation" in latex.lower()


# ═══════════════════════════════════════════════════════════════════════════
# §10  Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case tests for certificate module."""

    def test_lp_cert_with_zero_length_duals(self):
        cert = LPOptimalityCertificate(
            dual_ub=np.array([]),
            dual_eq=None,
            duality_gap=0.0,
            primal_obj=0.0,
            dual_obj=0.0,
        )
        assert len(cert.dual_ub) == 0
        assert cert.is_tight is True

    def test_sdp_cert_1x1_matrix(self):
        cert = SDPOptimalityCertificate(
            dual_matrix=np.array([[0.5]]),
            duality_gap=0.0,
            primal_obj=1.0,
            dual_obj=1.0,
        )
        assert cert.matrix_dim == 1
        assert cert.min_eigenvalue == pytest.approx(0.5)
        assert cert.is_psd is True

    def test_verifier_custom_tolerance(self):
        verifier = CertificateVerifier(tol=0.1)
        cert = _make_lp_cert(primal_obj=1.0, dual_obj=0.95, duality_gap=0.05)
        assert verifier.check_strong_duality(cert) is True

    def test_verifier_very_strict_tolerance(self):
        verifier = CertificateVerifier(tol=1e-15)
        cert = _make_lp_cert(primal_obj=1.0, dual_obj=1.0 - 1e-10, duality_gap=1e-10)
        assert verifier.check_strong_duality(cert) is False

    def test_large_duality_gap(self):
        cert = _make_lp_cert(primal_obj=100.0, dual_obj=0.0, duality_gap=100.0)
        assert cert.relative_gap == pytest.approx(1.0)
        assert cert.is_tight is False

    def test_generator_with_eqlin_marginals(self):
        gen = CertificateGenerator()
        spec = _make_query_spec()
        ineqlin = SimpleNamespace(marginals=np.array([0.1, 0.2]))
        eqlin = SimpleNamespace(marginals=np.array([0.5]))
        result = SimpleNamespace(
            fun=1.0, ineqlin=ineqlin, eqlin=eqlin, x=np.array([0.5, 0.5]),
        )
        cert = gen.generate(result, spec)
        np.testing.assert_allclose(cert.dual_ub, [0.1, 0.2])
        np.testing.assert_allclose(cert.dual_eq, [0.5])

    def test_composed_cert_single_component(self):
        c1 = _make_lp_cert()
        composed = ComposedCertificate(
            components=[(c1, 0.5, 1e-5)], composition_type="basic",
        )
        assert composed.total_epsilon == pytest.approx(0.5)
        assert composed.total_delta == pytest.approx(1e-5)
        assert composed.n_components == 1

    def test_chain_total_epsilon_zero_delta_advanced(self):
        c1 = _make_lp_cert()
        chain = CertificateChain(
            certificates=[c1, c1],
            epsilons=[1.0, 1.0],
            deltas=[0.0, 0.0],
            composition_type="advanced",
        )
        # With zero total delta, advanced falls back to basic
        assert chain.total_epsilon == pytest.approx(2.0)

    def test_from_spec_approximation_small_k(self):
        spec = _make_query_spec(n=2, epsilon=1.0, k=5)
        cert = ApproximationCertificate.from_spec(spec, k=2)
        assert cert.k == 2
        assert cert.grid_spacing > 0
        assert cert.discretization_error_bound > 0
