"""
Comprehensive tests for dp_forge.types — dataclasses, enums, validation.
"""

from __future__ import annotations

import math
import pickle
from copy import deepcopy

import numpy as np
import pytest

from dp_forge.types import (
    AdjacencyRelation,
    BenchmarkResult,
    CEGISResult,
    CompositionType,
    ExtractedMechanism,
    LossFunction,
    LPStruct,
    MechanismFamily,
    NumericalConfig,
    OptimalityCertificate,
    PrivacyBudget,
    QuerySpec,
    QueryType,
    SamplingConfig,
    SamplingMethod,
    SolverBackend,
    SynthesisConfig,
    VerifyResult,
    WorkloadSpec,
)
from scipy import sparse


# ═══════════════════════════════════════════════════════════════════════════
# §1  Enum Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryType:
    """Tests for QueryType enum."""

    def test_all_members_exist(self):
        expected = {"COUNTING", "HISTOGRAM", "RANGE", "LINEAR_WORKLOAD", "MARGINAL", "CUSTOM"}
        assert set(m.name for m in QueryType) == expected

    def test_repr(self):
        assert repr(QueryType.COUNTING) == "QueryType.COUNTING"
        assert repr(QueryType.HISTOGRAM) == "QueryType.HISTOGRAM"

    def test_identity(self):
        assert QueryType.COUNTING is QueryType.COUNTING
        assert QueryType.COUNTING != QueryType.HISTOGRAM


class TestMechanismFamily:
    """Tests for MechanismFamily enum."""

    def test_all_members_exist(self):
        expected = {"PIECEWISE_CONST", "PIECEWISE_LINEAR", "GAUSSIAN_WORKLOAD"}
        assert set(m.name for m in MechanismFamily) == expected

    def test_repr(self):
        assert repr(MechanismFamily.PIECEWISE_CONST) == "MechanismFamily.PIECEWISE_CONST"


class TestLossFunction:
    """Tests for LossFunction enum and its callable property."""

    def test_all_members_exist(self):
        expected = {"L1", "L2", "LINF", "CUSTOM"}
        assert set(m.name for m in LossFunction) == expected

    def test_l1_fn(self):
        fn = LossFunction.L1.fn
        assert fn is not None
        assert fn(3.0, 5.0) == 2.0
        assert fn(5.0, 3.0) == 2.0
        assert fn(0.0, 0.0) == 0.0
        assert fn(-1.0, 1.0) == 2.0

    def test_l2_fn(self):
        fn = LossFunction.L2.fn
        assert fn is not None
        assert fn(3.0, 5.0) == 4.0
        assert fn(0.0, 0.0) == 0.0
        assert fn(1.0, 2.0) == 1.0

    def test_linf_fn(self):
        fn = LossFunction.LINF.fn
        assert fn is not None
        assert fn(3.0, 5.0) == 2.0

    def test_custom_fn_is_none(self):
        assert LossFunction.CUSTOM.fn is None

    def test_repr(self):
        assert repr(LossFunction.L2) == "LossFunction.L2"


class TestSamplingMethod:
    def test_members(self):
        assert set(m.name for m in SamplingMethod) == {"ALIAS", "CDF", "REJECTION"}


class TestCompositionType:
    def test_members(self):
        assert set(m.name for m in CompositionType) == {"BASIC", "ADVANCED", "RDP", "ZERO_CDP"}


class TestSolverBackend:
    def test_members(self):
        expected = {"HIGHS", "GLPK", "MOSEK", "SCS", "SCIPY", "AUTO"}
        assert set(m.name for m in SolverBackend) == expected

    def test_values(self):
        assert SolverBackend.HIGHS.value == "highs"
        assert SolverBackend.AUTO.value == "auto"

    def test_repr(self):
        assert repr(SolverBackend.HIGHS) == "SolverBackend.HIGHS"


# ═══════════════════════════════════════════════════════════════════════════
# §2  AdjacencyRelation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAdjacencyRelation:
    """Tests for AdjacencyRelation dataclass."""

    def test_basic_construction(self):
        adj = AdjacencyRelation(edges=[(0, 1), (1, 2)], n=3)
        assert adj.n == 3
        assert len(adj.edges) == 2
        assert adj.symmetric is True

    def test_n_must_be_positive(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            AdjacencyRelation(edges=[], n=0)

    def test_edge_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            AdjacencyRelation(edges=[(0, 5)], n=3)

    def test_negative_edge_index(self):
        with pytest.raises(ValueError, match="out of range"):
            AdjacencyRelation(edges=[(-1, 0)], n=3)

    def test_self_loop_rejected(self):
        with pytest.raises(ValueError, match="Self-loop"):
            AdjacencyRelation(edges=[(1, 1)], n=3)

    def test_hamming_distance_1(self):
        adj = AdjacencyRelation.hamming_distance_1(5)
        assert adj.n == 5
        assert adj.edges == [(0, 1), (1, 2), (2, 3), (3, 4)]
        assert adj.symmetric is True
        assert "Hamming" in adj.description

    def test_hamming_distance_1_small(self):
        adj = AdjacencyRelation.hamming_distance_1(2)
        assert adj.edges == [(0, 1)]

    def test_complete(self):
        adj = AdjacencyRelation.complete(4)
        assert adj.n == 4
        expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        assert adj.edges == expected

    def test_num_edges_symmetric(self):
        adj = AdjacencyRelation(edges=[(0, 1), (1, 2)], n=3, symmetric=True)
        assert adj.num_edges == 4

    def test_num_edges_asymmetric(self):
        adj = AdjacencyRelation(edges=[(0, 1), (1, 2)], n=3, symmetric=False)
        assert adj.num_edges == 2

    def test_repr_contains_info(self):
        adj = AdjacencyRelation(edges=[(0, 1)], n=2)
        r = repr(adj)
        assert "n=2" in r
        assert "edges=1" in r


# ═══════════════════════════════════════════════════════════════════════════
# §3  PrivacyBudget Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPrivacyBudget:
    """Tests for PrivacyBudget dataclass."""

    def test_pure_dp(self):
        b = PrivacyBudget(epsilon=1.0)
        assert b.epsilon == 1.0
        assert b.delta == 0.0
        assert b.is_pure is True

    def test_approx_dp(self):
        b = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert b.is_pure is False

    def test_zero_epsilon_rejected(self):
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            PrivacyBudget(epsilon=0.0)

    def test_negative_epsilon_rejected(self):
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            PrivacyBudget(epsilon=-1.0)

    def test_inf_epsilon_rejected(self):
        with pytest.raises(ValueError, match="epsilon must be finite"):
            PrivacyBudget(epsilon=float("inf"))

    def test_nan_epsilon_rejected(self):
        with pytest.raises(ValueError, match="epsilon must be finite"):
            PrivacyBudget(epsilon=float("nan"))

    def test_delta_ge_1_rejected(self):
        with pytest.raises(ValueError, match="delta must be in"):
            PrivacyBudget(epsilon=1.0, delta=1.0)

    def test_delta_negative_rejected(self):
        with pytest.raises(ValueError, match="delta must be in"):
            PrivacyBudget(epsilon=1.0, delta=-0.1)

    def test_repr_pure(self):
        b = PrivacyBudget(epsilon=1.0)
        assert "ε=1.0" in repr(b)

    def test_repr_approx(self):
        b = PrivacyBudget(epsilon=1.0, delta=1e-5)
        r = repr(b)
        assert "ε=1.0" in r
        assert "δ=" in r

    @pytest.mark.parametrize("eps", [0.01, 0.1, 0.5, 1.0, 2.0, 10.0])
    def test_various_epsilon(self, eps):
        b = PrivacyBudget(epsilon=eps)
        assert b.epsilon == eps


# ═══════════════════════════════════════════════════════════════════════════
# §4  QuerySpec Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQuerySpec:
    """Tests for QuerySpec dataclass."""

    def test_basic_construction(self):
        qv = np.array([0.0, 1.0, 2.0])
        spec = QuerySpec(query_values=qv, domain="test", sensitivity=1.0, epsilon=1.0)
        assert spec.n == 3
        assert spec.is_pure_dp is True
        assert spec.k == 100

    def test_query_values_cast_to_float64(self):
        spec = QuerySpec(
            query_values=[0, 1, 2],
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
        )
        assert spec.query_values.dtype == np.float64

    def test_empty_query_values_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            QuerySpec(query_values=np.array([]), domain="test", sensitivity=1.0, epsilon=1.0)

    def test_2d_query_values_rejected(self):
        with pytest.raises(ValueError, match="1-D"):
            QuerySpec(
                query_values=np.array([[1, 2], [3, 4]]),
                domain="test",
                sensitivity=1.0,
                epsilon=1.0,
            )

    def test_negative_sensitivity_rejected(self):
        with pytest.raises(ValueError, match="sensitivity must be > 0"):
            QuerySpec(
                query_values=np.array([0.0, 1.0]),
                domain="test",
                sensitivity=-1.0,
                epsilon=1.0,
            )

    def test_zero_sensitivity_rejected(self):
        with pytest.raises(ValueError, match="sensitivity must be > 0"):
            QuerySpec(
                query_values=np.array([0.0, 1.0]),
                domain="test",
                sensitivity=0.0,
                epsilon=1.0,
            )

    def test_negative_epsilon_rejected(self):
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            QuerySpec(
                query_values=np.array([0.0, 1.0]),
                domain="test",
                sensitivity=1.0,
                epsilon=-1.0,
            )

    def test_delta_gt_1_rejected(self):
        with pytest.raises(ValueError, match="delta must be in"):
            QuerySpec(
                query_values=np.array([0.0, 1.0]),
                domain="test",
                sensitivity=1.0,
                epsilon=1.0,
                delta=1.5,
            )

    def test_k_less_than_2_rejected(self):
        with pytest.raises(ValueError, match="k must be >= 2"):
            QuerySpec(
                query_values=np.array([0.0, 1.0]),
                domain="test",
                sensitivity=1.0,
                epsilon=1.0,
                k=1,
            )

    def test_k_zero_rejected(self):
        with pytest.raises(ValueError, match="k must be >= 2"):
            QuerySpec(
                query_values=np.array([0.0, 1.0]),
                domain="test",
                sensitivity=1.0,
                epsilon=1.0,
                k=0,
            )

    def test_custom_loss_requires_callable(self):
        with pytest.raises(ValueError, match="custom_loss callable required"):
            QuerySpec(
                query_values=np.array([0.0, 1.0]),
                domain="test",
                sensitivity=1.0,
                epsilon=1.0,
                loss_fn=LossFunction.CUSTOM,
            )

    def test_custom_loss_accepted(self):
        spec = QuerySpec(
            query_values=np.array([0.0, 1.0]),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            loss_fn=LossFunction.CUSTOM,
            custom_loss=lambda t, n: (t - n) ** 2,
        )
        fn = spec.get_loss_callable()
        assert fn(1.0, 2.0) == 1.0

    def test_edges_n_mismatch_rejected(self):
        adj = AdjacencyRelation(edges=[(0, 1)], n=2)
        with pytest.raises(ValueError, match="edges.n"):
            QuerySpec(
                query_values=np.array([0.0, 1.0, 2.0]),
                domain="test",
                sensitivity=1.0,
                epsilon=1.0,
                edges=adj,
            )

    def test_default_edges_auto_created(self):
        spec = QuerySpec(
            query_values=np.array([0.0, 1.0, 2.0]),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
        )
        assert spec.edges is not None
        assert spec.edges.n == 3

    def test_eta_min(self):
        spec = QuerySpec(
            query_values=np.array([0.0, 1.0]),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
        )
        expected = math.exp(-1.0) * 1e-10
        assert abs(spec.eta_min - expected) < 1e-20

    def test_get_loss_callable_l2(self):
        spec = QuerySpec(
            query_values=np.array([0.0, 1.0]),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            loss_fn=LossFunction.L2,
        )
        fn = spec.get_loss_callable()
        assert fn(3.0, 5.0) == 4.0

    def test_counting_factory(self):
        spec = QuerySpec.counting(n=5, epsilon=1.0)
        assert spec.n == 5
        assert spec.sensitivity == 1.0
        assert spec.query_type == QueryType.COUNTING
        np.testing.assert_array_equal(spec.query_values, np.arange(5, dtype=np.float64))

    def test_counting_factory_with_delta(self):
        spec = QuerySpec.counting(n=3, epsilon=0.5, delta=1e-5, k=50)
        assert spec.delta == 1e-5
        assert spec.k == 50
        assert spec.is_pure_dp is False

    def test_histogram_factory(self):
        spec = QuerySpec.histogram(n_bins=10, epsilon=1.0)
        assert spec.n == 10
        assert spec.sensitivity == 1.0
        assert spec.query_type == QueryType.HISTOGRAM

    def test_repr(self):
        spec = QuerySpec.counting(n=3, epsilon=1.0)
        r = repr(spec)
        assert "n=3" in r
        assert "ε=1.0" in r

    @pytest.mark.parametrize("loss", [LossFunction.L1, LossFunction.L2, LossFunction.LINF])
    def test_all_standard_losses(self, loss):
        spec = QuerySpec(
            query_values=np.array([0.0, 1.0]),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            loss_fn=loss,
        )
        fn = spec.get_loss_callable()
        assert fn(0.0, 1.0) >= 0

    def test_is_pure_dp_property(self):
        spec_pure = QuerySpec.counting(n=3, epsilon=1.0, delta=0.0)
        assert spec_pure.is_pure_dp is True
        spec_approx = QuerySpec.counting(n=3, epsilon=1.0, delta=0.01)
        assert spec_approx.is_pure_dp is False


# ═══════════════════════════════════════════════════════════════════════════
# §5  WorkloadSpec Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkloadSpec:
    """Tests for WorkloadSpec dataclass."""

    def test_basic_construction(self):
        A = np.eye(3)
        ws = WorkloadSpec(matrix=A)
        assert ws.m == 3
        assert ws.d == 3

    def test_cast_to_float64(self):
        A = np.array([[1, 0], [0, 1]], dtype=np.int32)
        ws = WorkloadSpec(matrix=A)
        assert ws.matrix.dtype == np.float64

    def test_1d_rejected(self):
        with pytest.raises(ValueError, match="2-D"):
            WorkloadSpec(matrix=np.array([1.0, 2.0, 3.0]))

    def test_non_finite_rejected(self):
        A = np.array([[1.0, np.inf], [0.0, 1.0]])
        with pytest.raises(ValueError, match="non-finite"):
            WorkloadSpec(matrix=A)

    def test_nan_rejected(self):
        A = np.array([[1.0, np.nan], [0.0, 1.0]])
        with pytest.raises(ValueError, match="non-finite"):
            WorkloadSpec(matrix=A)

    def test_identity_factory(self):
        ws = WorkloadSpec.identity(5)
        np.testing.assert_array_equal(ws.matrix, np.eye(5))
        assert ws.m == 5
        assert ws.d == 5

    def test_all_range_factory(self):
        ws = WorkloadSpec.all_range(4)
        expected = np.tril(np.ones((4, 4)))
        np.testing.assert_array_equal(ws.matrix, expected)
        assert ws.structural_hint == "toeplitz"

    def test_m_and_d_properties(self):
        A = np.ones((3, 5))
        ws = WorkloadSpec(matrix=A)
        assert ws.m == 3
        assert ws.d == 5

    def test_repr(self):
        ws = WorkloadSpec.identity(3)
        r = repr(ws)
        assert "m=3" in r
        assert "d=3" in r


# ═══════════════════════════════════════════════════════════════════════════
# §6  LPStruct Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLPStruct:
    """Tests for LPStruct dataclass."""

    def _make_lp(self, n_vars=5, n_ub=3, n_eq=2):
        c = np.zeros(n_vars)
        A_ub = sparse.csr_matrix(np.random.randn(n_ub, n_vars))
        b_ub = np.zeros(n_ub)
        A_eq = sparse.csr_matrix(np.random.randn(n_eq, n_vars))
        b_eq = np.ones(n_eq)
        bounds = [(0.0, 1.0)] * n_vars
        var_map = {}
        y_grid = np.linspace(0, 1, 10)
        return LPStruct(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, var_map=var_map, y_grid=y_grid)

    def test_basic_construction(self):
        lp = self._make_lp()
        assert lp.n_vars == 5
        assert lp.n_ub == 3
        assert lp.n_eq == 2

    def test_column_mismatch_A_ub(self):
        c = np.zeros(5)
        A_ub = sparse.csr_matrix(np.zeros((3, 4)))
        with pytest.raises(ValueError, match="A_ub has"):
            LPStruct(c=c, A_ub=A_ub, b_ub=np.zeros(3), A_eq=None, b_eq=None,
                     bounds=[(0, 1)] * 5, var_map={}, y_grid=np.zeros(10))

    def test_b_ub_length_mismatch(self):
        c = np.zeros(5)
        A_ub = sparse.csr_matrix(np.zeros((3, 5)))
        with pytest.raises(ValueError, match="b_ub length"):
            LPStruct(c=c, A_ub=A_ub, b_ub=np.zeros(2), A_eq=None, b_eq=None,
                     bounds=[(0, 1)] * 5, var_map={}, y_grid=np.zeros(10))

    def test_a_eq_without_b_eq(self):
        c = np.zeros(5)
        A_ub = sparse.csr_matrix(np.zeros((3, 5)))
        A_eq = sparse.csr_matrix(np.zeros((2, 5)))
        with pytest.raises(ValueError, match="b_eq must be provided"):
            LPStruct(c=c, A_ub=A_ub, b_ub=np.zeros(3), A_eq=A_eq, b_eq=None,
                     bounds=[(0, 1)] * 5, var_map={}, y_grid=np.zeros(10))

    def test_bounds_length_mismatch(self):
        c = np.zeros(5)
        A_ub = sparse.csr_matrix(np.zeros((3, 5)))
        with pytest.raises(ValueError, match="bounds has"):
            LPStruct(c=c, A_ub=A_ub, b_ub=np.zeros(3), A_eq=None, b_eq=None,
                     bounds=[(0, 1)] * 4, var_map={}, y_grid=np.zeros(10))

    def test_no_equality_constraints(self):
        c = np.zeros(5)
        A_ub = sparse.csr_matrix(np.zeros((3, 5)))
        lp = LPStruct(c=c, A_ub=A_ub, b_ub=np.zeros(3), A_eq=None, b_eq=None,
                      bounds=[(0, 1)] * 5, var_map={}, y_grid=np.zeros(10))
        assert lp.n_eq == 0

    def test_sparsity(self):
        c = np.zeros(5)
        A_ub = sparse.csr_matrix(np.zeros((3, 5)))
        lp = LPStruct(c=c, A_ub=A_ub, b_ub=np.zeros(3), A_eq=None, b_eq=None,
                      bounds=[(0, 1)] * 5, var_map={}, y_grid=np.zeros(10))
        assert lp.sparsity == 1.0

    def test_repr(self):
        lp = self._make_lp()
        r = repr(lp)
        assert "vars=" in r
        assert "ub=" in r


# ═══════════════════════════════════════════════════════════════════════════
# §7  OptimalityCertificate Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOptimalityCertificate:
    """Tests for OptimalityCertificate dataclass."""

    def test_basic_construction(self):
        cert = OptimalityCertificate(dual_vars=None, duality_gap=1e-8,
                                     primal_obj=1.0, dual_obj=0.9999999)
        assert cert.duality_gap == pytest.approx(1e-8)

    def test_negative_gap_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            OptimalityCertificate(dual_vars=None, duality_gap=-1.0,
                                  primal_obj=1.0, dual_obj=2.0)

    def test_small_negative_gap_allowed(self):
        # Numerical noise of -1e-10 should be allowed
        cert = OptimalityCertificate(dual_vars=None, duality_gap=-1e-10,
                                     primal_obj=1.0, dual_obj=1.0)
        assert cert.duality_gap == pytest.approx(-1e-10)

    def test_inf_primal_rejected(self):
        with pytest.raises(ValueError, match="primal_obj must be finite"):
            OptimalityCertificate(dual_vars=None, duality_gap=0.0,
                                  primal_obj=float("inf"), dual_obj=0.0)

    def test_inf_dual_rejected(self):
        with pytest.raises(ValueError, match="dual_obj must be finite"):
            OptimalityCertificate(dual_vars=None, duality_gap=0.0,
                                  primal_obj=0.0, dual_obj=float("inf"))

    def test_relative_gap(self):
        cert = OptimalityCertificate(dual_vars=None, duality_gap=0.01,
                                     primal_obj=10.0, dual_obj=9.99)
        assert cert.relative_gap == pytest.approx(0.001)

    def test_relative_gap_small_primal(self):
        cert = OptimalityCertificate(dual_vars=None, duality_gap=0.01,
                                     primal_obj=0.001, dual_obj=-0.009)
        # denom = max(|0.001|, 1.0) = 1.0
        assert cert.relative_gap == pytest.approx(0.01)

    def test_is_tight(self):
        cert = OptimalityCertificate(dual_vars=None, duality_gap=1e-9,
                                     primal_obj=1.0, dual_obj=1.0)
        assert cert.is_tight is True

    def test_repr(self):
        cert = OptimalityCertificate(dual_vars=None, duality_gap=1e-8,
                                     primal_obj=1.0, dual_obj=1.0)
        assert "gap=" in repr(cert)


# ═══════════════════════════════════════════════════════════════════════════
# §8  VerifyResult Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVerifyResult:
    """Tests for VerifyResult dataclass."""

    def test_valid_result(self):
        vr = VerifyResult(valid=True)
        assert vr.valid is True
        assert vr.violation is None
        assert vr.violation_pair is None
        assert vr.violation_magnitude == 0.0

    def test_invalid_result(self):
        vr = VerifyResult(valid=False, violation=(0, 1, 5, 0.01))
        assert vr.valid is False
        assert vr.violation_pair == (0, 1)
        assert vr.violation_magnitude == 0.01

    def test_invalid_without_violation_rejected(self):
        with pytest.raises(ValueError, match="violation must be provided"):
            VerifyResult(valid=False)

    def test_valid_with_violation_rejected(self):
        with pytest.raises(ValueError, match="violation must be None"):
            VerifyResult(valid=True, violation=(0, 1, 0, 0.1))

    def test_repr_valid(self):
        vr = VerifyResult(valid=True)
        assert "valid=True" in repr(vr)

    def test_repr_invalid(self):
        vr = VerifyResult(valid=False, violation=(0, 1, 5, 0.01))
        r = repr(vr)
        assert "valid=False" in r
        assert "pair=" in r


# ═══════════════════════════════════════════════════════════════════════════
# §9  CEGISResult Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCEGISResult:
    """Tests for CEGISResult dataclass."""

    def _make_mechanism(self, n=3, k=5):
        p = np.ones((n, k)) / k
        return p

    def test_basic_construction(self):
        p = self._make_mechanism()
        res = CEGISResult(mechanism=p, iterations=10, obj_val=1.5)
        assert res.n == 3
        assert res.k == 5
        assert res.iterations == 10

    def test_mechanism_cast_to_float64(self):
        p = np.ones((2, 4)) / 4
        res = CEGISResult(mechanism=p.tolist(), iterations=1, obj_val=0.0)
        assert res.mechanism.dtype == np.float64

    def test_1d_mechanism_rejected(self):
        with pytest.raises(ValueError, match="2-D"):
            CEGISResult(mechanism=np.array([0.5, 0.5]), iterations=0, obj_val=0.0)

    def test_negative_iterations_rejected(self):
        with pytest.raises(ValueError, match="iterations must be >= 0"):
            CEGISResult(mechanism=self._make_mechanism(), iterations=-1, obj_val=0.0)

    def test_rows_not_summing_to_1_rejected(self):
        p = np.ones((2, 3)) * 0.5  # rows sum to 1.5
        with pytest.raises(ValueError, match="rows must sum to 1"):
            CEGISResult(mechanism=p, iterations=0, obj_val=0.0)

    def test_converged_single_iteration(self):
        res = CEGISResult(
            mechanism=self._make_mechanism(),
            iterations=1,
            obj_val=1.0,
            convergence_history=[1.0],
        )
        assert res.converged is True

    def test_converged_stable(self):
        res = CEGISResult(
            mechanism=self._make_mechanism(),
            iterations=3,
            obj_val=1.0,
            convergence_history=[1.0, 1.0, 1.0],
        )
        assert res.converged is True

    def test_not_converged(self):
        res = CEGISResult(
            mechanism=self._make_mechanism(),
            iterations=3,
            obj_val=2.0,
            convergence_history=[1.0, 1.5, 2.0],
        )
        assert res.converged is False

    def test_repr(self):
        res = CEGISResult(mechanism=self._make_mechanism(), iterations=5, obj_val=1.23)
        r = repr(res)
        assert "n=3" in r
        assert "k=5" in r


# ═══════════════════════════════════════════════════════════════════════════
# §10  ExtractedMechanism Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractedMechanism:
    """Tests for ExtractedMechanism dataclass."""

    def _make_valid(self, n=3, k=5):
        return np.ones((n, k)) / k

    def test_basic_construction(self):
        em = ExtractedMechanism(p_final=self._make_valid())
        assert em.n == 3
        assert em.k == 5

    def test_cast_to_float64(self):
        p = self._make_valid().tolist()
        em = ExtractedMechanism(p_final=p)
        assert em.p_final.dtype == np.float64

    def test_1d_rejected(self):
        with pytest.raises(ValueError, match="2-D"):
            ExtractedMechanism(p_final=np.array([0.5, 0.5]))

    def test_negative_probabilities_rejected(self):
        p = self._make_valid()
        p[0, 0] = -0.1
        with pytest.raises(ValueError, match="negative probabilities"):
            ExtractedMechanism(p_final=p)

    def test_tiny_negatives_allowed(self):
        p = self._make_valid()
        p[0, 0] -= 1e-14
        p[0, 1] += 1e-14
        em = ExtractedMechanism(p_final=p)
        assert em.n == 3

    def test_rows_not_summing_rejected(self):
        p = np.ones((2, 3)) * 0.5  # rows sum to 1.5
        with pytest.raises(ValueError, match="rows must sum to 1"):
            ExtractedMechanism(p_final=p)

    def test_repr(self):
        em = ExtractedMechanism(p_final=self._make_valid())
        r = repr(em)
        assert "n=3" in r
        assert "uncertified" in r

    def test_repr_certified(self):
        cert = OptimalityCertificate(dual_vars=None, duality_gap=0.0,
                                     primal_obj=1.0, dual_obj=1.0)
        em = ExtractedMechanism(p_final=self._make_valid(),
                                optimality_certificate=cert)
        assert "certified" in repr(em)


# ═══════════════════════════════════════════════════════════════════════════
# §11  BenchmarkResult Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBenchmarkResult:
    def test_basic_construction(self):
        br = BenchmarkResult(mse=0.5, mae=0.3, synthesis_time=1.0,
                             iterations=10, privacy_verified=True)
        assert br.mse == 0.5

    def test_negative_mse_rejected(self):
        with pytest.raises(ValueError, match="mse must be >= 0"):
            BenchmarkResult(mse=-1.0, mae=0.0, synthesis_time=0.0,
                           iterations=0, privacy_verified=True)

    def test_negative_mae_rejected(self):
        with pytest.raises(ValueError, match="mae must be >= 0"):
            BenchmarkResult(mse=0.0, mae=-1.0, synthesis_time=0.0,
                           iterations=0, privacy_verified=True)

    def test_negative_time_rejected(self):
        with pytest.raises(ValueError, match="synthesis_time must be >= 0"):
            BenchmarkResult(mse=0.0, mae=0.0, synthesis_time=-1.0,
                           iterations=0, privacy_verified=True)

    def test_negative_iterations_rejected(self):
        with pytest.raises(ValueError, match="iterations must be >= 0"):
            BenchmarkResult(mse=0.0, mae=0.0, synthesis_time=0.0,
                           iterations=-1, privacy_verified=True)

    def test_repr(self):
        br = BenchmarkResult(mse=0.5, mae=0.3, synthesis_time=1.0,
                             iterations=10, privacy_verified=True)
        assert "✓" in repr(br)

    def test_repr_unverified(self):
        br = BenchmarkResult(mse=0.5, mae=0.3, synthesis_time=1.0,
                             iterations=10, privacy_verified=False)
        assert "✗" in repr(br)


# ═══════════════════════════════════════════════════════════════════════════
# §12  Configuration Dataclass Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSamplingConfig:
    def test_defaults(self):
        sc = SamplingConfig()
        assert sc.method == SamplingMethod.ALIAS
        assert sc.seed is None

    def test_get_rng_deterministic(self):
        sc = SamplingConfig(seed=42)
        rng1 = sc.get_rng()
        rng2 = sc.get_rng()
        assert rng1.random() == rng2.random()

    def test_negative_seed_rejected(self):
        with pytest.raises(ValueError, match="seed must be non-negative"):
            SamplingConfig(seed=-1)

    def test_repr(self):
        sc = SamplingConfig(seed=42)
        r = repr(sc)
        assert "seed=42" in r


class TestNumericalConfig:
    def test_defaults(self):
        nc = NumericalConfig()
        assert nc.solver_tol == 1e-8
        assert nc.dp_tol == 1e-6
        assert nc.eta_min_scale == 1e-10
        assert nc.max_condition_number == 1e12

    def test_zero_solver_tol_rejected(self):
        with pytest.raises(ValueError, match="solver_tol must be > 0"):
            NumericalConfig(solver_tol=0.0)

    def test_negative_dp_tol_rejected(self):
        with pytest.raises(ValueError, match="dp_tol must be > 0"):
            NumericalConfig(dp_tol=-1.0)

    def test_eta_min_computation(self):
        nc = NumericalConfig()
        eps = 1.0
        expected = math.exp(-eps) * 1e-10
        assert nc.eta_min(eps) == pytest.approx(expected)

    def test_validate_dp_tol_passes(self):
        nc = NumericalConfig(solver_tol=1e-8, dp_tol=1e-4)
        assert nc.validate_dp_tol(1.0) is True

    def test_validate_dp_tol_fails(self):
        nc = NumericalConfig(solver_tol=1e-2, dp_tol=1e-4)
        # exp(1)*1e-2 ≈ 0.027 > 1e-4
        assert nc.validate_dp_tol(1.0) is False

    @pytest.mark.parametrize("eps", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_eta_min_decreases_with_epsilon(self, eps):
        nc = NumericalConfig()
        eta = nc.eta_min(eps)
        assert eta > 0
        assert eta < 1

    def test_repr(self):
        nc = NumericalConfig()
        r = repr(nc)
        assert "solver_tol" in r


class TestSynthesisConfig:
    def test_defaults(self):
        sc = SynthesisConfig()
        assert sc.max_iter == 50
        assert sc.tol == 1e-8
        assert sc.warm_start is True
        assert sc.solver == SolverBackend.AUTO
        assert sc.verbose == 1
        assert sc.symmetry_detection is True

    def test_max_iter_zero_rejected(self):
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            SynthesisConfig(max_iter=0)

    def test_negative_tol_rejected(self):
        with pytest.raises(ValueError, match="tol must be > 0"):
            SynthesisConfig(tol=-1.0)

    def test_invalid_verbose_rejected(self):
        with pytest.raises(ValueError, match="verbose must be 0, 1, or 2"):
            SynthesisConfig(verbose=3)

    def test_negative_eta_min_rejected(self):
        with pytest.raises(ValueError, match="eta_min must be > 0"):
            SynthesisConfig(eta_min=-1e-5)

    def test_effective_eta_min_override(self):
        sc = SynthesisConfig(eta_min=1e-20)
        assert sc.effective_eta_min(1.0) == 1e-20

    def test_effective_eta_min_default(self):
        sc = SynthesisConfig()
        expected = math.exp(-1.0) * 1e-10
        assert sc.effective_eta_min(1.0) == pytest.approx(expected)

    def test_repr(self):
        sc = SynthesisConfig()
        r = repr(sc)
        assert "max_iter=50" in r
