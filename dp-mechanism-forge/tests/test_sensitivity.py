"""
Comprehensive tests for dp_forge.query_sensitivity — global sensitivity
computation, adjacency graph builders, query-specific sensitivity classes,
and the unified QuerySensitivityAnalyzer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dp_forge.query_sensitivity import (
    SensitivityResult,
    QuerySensitivityAnalyzer,
    CountingQuerySensitivity,
    HistogramQuerySensitivity,
    RangeQuerySensitivity,
    LinearWorkloadSensitivity,
    MarginalQuerySensitivity,
    CustomFunctionSensitivity,
    sensitivity_l1,
    sensitivity_l2,
    sensitivity_linf,
    adjacency_graph,
    bounded_adjacency,
    hamming_adjacency,
    substitution_adjacency,
    add_remove_adjacency,
    generic_adjacency,
    workload_sensitivity,
    query_spec_sensitivity,
    validate_sensitivity,
    sensitivity_from_function,
)
from dp_forge.types import (
    AdjacencyRelation,
    QuerySpec,
    QueryType,
    WorkloadSpec,
)
from dp_forge.exceptions import ConfigurationError, SensitivityError


# ═══════════════════════════════════════════════════════════════════════════
# §1  Top-level sensitivity functions: sensitivity_l1, _l2, _linf
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivityL1:
    """Tests for the sensitivity_l1 function."""

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 20])
    def test_identity_various_sizes(self, n: int):
        assert sensitivity_l1(np.eye(n)) == 1.0

    def test_upper_triangular(self):
        assert sensitivity_l1(np.array([[1, 1], [0, 1]])) == 2.0

    def test_all_ones_matrix(self):
        assert sensitivity_l1(np.ones((3, 3))) == 3.0

    def test_single_column(self):
        assert sensitivity_l1(np.array([[1], [2], [3]])) == 6.0

    def test_negative_entries(self):
        # col 0: |1|+|-3|=4, col 1: |-2|+|4|=6
        assert sensitivity_l1(np.array([[1, -2], [-3, 4]])) == 6.0

    def test_zeros_matrix(self):
        assert sensitivity_l1(np.zeros((3, 3))) == 0.0

    def test_prefix_sum_matrix(self):
        d = 4
        assert sensitivity_l1(np.tril(np.ones((d, d)))) == float(d)

    def test_non_finite_raises(self):
        with pytest.raises(SensitivityError):
            sensitivity_l1(np.array([[1.0, np.inf], [0.0, 1.0]]))

    def test_nan_raises(self):
        with pytest.raises(SensitivityError):
            sensitivity_l1(np.array([[1.0, np.nan], [0.0, 1.0]]))

    def test_with_hamming_adjacency(self):
        adj = AdjacencyRelation.hamming_distance_1(3)
        # ||e_i - e_{i+1}||_1 = 2
        assert sensitivity_l1(np.eye(3), adjacency=adj) == pytest.approx(2.0)

    def test_with_complete_adjacency(self):
        adj = AdjacencyRelation.complete(3)
        assert sensitivity_l1(np.eye(3), adjacency=adj) == pytest.approx(2.0)

    @pytest.mark.parametrize(
        "matrix, expected",
        [
            (np.eye(2), 1.0),
            (np.array([[1, 0], [0, 1], [1, 1]]), 2.0),
            (np.array([[2, 0], [0, 3]]), 3.0),
        ],
    )
    def test_parametrized_matrices(self, matrix, expected):
        assert sensitivity_l1(matrix) == pytest.approx(expected)


class TestSensitivityL2:
    """Tests for the sensitivity_l2 function."""

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_identity_various_sizes(self, n: int):
        assert sensitivity_l2(np.eye(n)) == 1.0

    def test_column_norms(self):
        # col 0: sqrt(9+16)=5
        assert sensitivity_l2(np.array([[3, 0], [4, 0]])) == pytest.approx(5.0)

    def test_all_ones(self):
        assert sensitivity_l2(np.ones((4, 4))) == pytest.approx(2.0)

    def test_prefix_sum(self):
        d = 4
        assert sensitivity_l2(np.tril(np.ones((d, d)))) == pytest.approx(math.sqrt(d))

    def test_non_finite_raises(self):
        with pytest.raises(SensitivityError):
            sensitivity_l2(np.array([[np.inf]]))

    def test_with_adjacency(self):
        adj = AdjacencyRelation.hamming_distance_1(3)
        assert sensitivity_l2(np.eye(3), adjacency=adj) == pytest.approx(math.sqrt(2.0))

    def test_zeros(self):
        assert sensitivity_l2(np.zeros((3, 3))) == 0.0


class TestSensitivityLinf:
    """Tests for the sensitivity_linf function."""

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_identity_various_sizes(self, n: int):
        assert sensitivity_linf(np.eye(n)) == 1.0

    def test_docstring_example(self):
        assert sensitivity_linf(np.array([[1, 2], [3, 4]])) == 4.0

    def test_negative_entries(self):
        assert sensitivity_linf(np.array([[-5, 2], [3, -4]])) == 5.0

    def test_prefix_sum(self):
        assert sensitivity_linf(np.tril(np.ones((4, 4)))) == 1.0

    def test_with_adjacency(self):
        A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        adj = AdjacencyRelation.hamming_distance_1(3)
        assert sensitivity_linf(A, adjacency=adj) == pytest.approx(3.0)

    def test_non_finite_raises(self):
        with pytest.raises(SensitivityError):
            sensitivity_linf(np.array([[np.nan, 1], [2, 3]]))


# ═══════════════════════════════════════════════════════════════════════════
# §2  Adjacency builders
# ═══════════════════════════════════════════════════════════════════════════


class TestBoundedAdjacency:

    def test_basic(self):
        adj = bounded_adjacency(5)
        assert adj.n == 5
        assert len(adj.edges) == 4
        assert adj.symmetric is True

    def test_single_element(self):
        adj = bounded_adjacency(1)
        assert len(adj.edges) == 0

    def test_edges_consecutive(self):
        for i, (a, b) in enumerate(bounded_adjacency(5).edges):
            assert (a, b) == (i, i + 1)

    def test_invalid_n(self):
        with pytest.raises(ConfigurationError):
            bounded_adjacency(0)

    def test_num_edges_property(self):
        assert bounded_adjacency(5).num_edges == 8  # 4 undirected = 8 directed


class TestHammingAdjacency:

    def test_hamming_1_dim_2(self):
        adj = hamming_adjacency(2, k=1)
        assert adj.n == 4
        assert len(adj.edges) == 4

    def test_hamming_1_dim_3(self):
        adj = hamming_adjacency(3, k=1)
        assert adj.n == 8
        assert len(adj.edges) == 12  # 8*3/2

    def test_hamming_2_dim_3(self):
        assert len(hamming_adjacency(3, k=2).edges) == 12

    def test_hamming_d_equals_k(self):
        assert len(hamming_adjacency(3, k=3).edges) == 4

    def test_symmetric(self):
        assert hamming_adjacency(3, k=1).symmetric is True

    def test_invalid_k_zero(self):
        with pytest.raises(ConfigurationError):
            hamming_adjacency(3, k=0)

    def test_invalid_k_exceeds_d(self):
        with pytest.raises(ConfigurationError):
            hamming_adjacency(3, k=4)

    def test_d_too_large(self):
        with pytest.raises(ConfigurationError):
            hamming_adjacency(17)

    def test_dim_1(self):
        adj = hamming_adjacency(1, k=1)
        assert adj.n == 2 and adj.edges == [(0, 1)]


class TestSubstitutionAdjacency:

    @pytest.mark.parametrize("n, expected", [(1, 0), (2, 1), (3, 3), (5, 10)])
    def test_edge_count(self, n: int, expected: int):
        assert len(substitution_adjacency(n).edges) == expected

    def test_three_element_edges(self):
        edge_set = set(substitution_adjacency(3).edges)
        assert edge_set == {(0, 1), (0, 2), (1, 2)}

    def test_invalid_n(self):
        with pytest.raises(ConfigurationError):
            substitution_adjacency(0)


class TestAddRemoveAdjacency:

    def test_matches_bounded(self):
        for n in [2, 5, 10]:
            assert add_remove_adjacency(n).edges == bounded_adjacency(n).edges

    def test_invalid_n(self):
        with pytest.raises(ConfigurationError):
            add_remove_adjacency(0)


class TestGenericAdjacency:

    def test_distance_one(self):
        adj = generic_adjacency([0, 1, 2, 3], lambda x, y: abs(x - y) == 1)
        assert adj.n == 4 and len(adj.edges) == 3 and adj.symmetric is True

    def test_asymmetric_relation(self):
        adj = generic_adjacency([0, 1, 2], lambda x, y: x < y)
        assert adj.symmetric is False

    def test_complete_via_lambda(self):
        assert len(generic_adjacency([0, 1, 2], lambda x, y: x != y).edges) == 3

    def test_empty_relation(self):
        assert len(generic_adjacency([0, 1, 2], lambda x, y: False).edges) == 0

    def test_single_element_domain(self):
        assert generic_adjacency([42], lambda x, y: True).n == 1

    def test_string_domain(self):
        adj = generic_adjacency(["a", "b", "c"], lambda x, y: abs(ord(x) - ord(y)) == 1)
        assert adj.n == 3 and adj.symmetric is True


# ═══════════════════════════════════════════════════════════════════════════
# §3  AdjacencyRelation dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestAdjacencyRelation:

    def test_hamming_distance_1(self):
        adj = AdjacencyRelation.hamming_distance_1(5)
        assert adj.n == 5 and len(adj.edges) == 4

    def test_complete(self):
        assert len(AdjacencyRelation.complete(4).edges) == 6

    def test_invalid_n_zero(self):
        with pytest.raises(ValueError):
            AdjacencyRelation(edges=[], n=0)

    def test_invalid_edge_out_of_range(self):
        with pytest.raises(ValueError):
            AdjacencyRelation(edges=[(0, 5)], n=3)

    def test_self_loop_rejected(self):
        with pytest.raises(ValueError):
            AdjacencyRelation(edges=[(1, 1)], n=3)

    def test_num_edges_symmetric_vs_asymmetric(self):
        edges = [(0, 1), (1, 2)]
        assert AdjacencyRelation(edges=edges, n=3, symmetric=True).num_edges == 4
        assert AdjacencyRelation(edges=edges, n=3, symmetric=False).num_edges == 2


# ═══════════════════════════════════════════════════════════════════════════
# §4  SensitivityResult dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivityResult:

    def test_basic_construction(self):
        sr = SensitivityResult(l1=1.0, l2=1.0, linf=1.0)
        assert sr.l1 == 1.0 and sr.l2 == 1.0 and sr.linf == 1.0

    def test_defaults(self):
        sr = SensitivityResult(l1=1.0, l2=1.0, linf=1.0)
        assert sr.query_type == "unknown"
        assert sr.is_tight is True
        assert sr.details == {}

    def test_max_sensitivity(self):
        assert SensitivityResult(l1=3.0, l2=2.0, linf=1.0).max_sensitivity() == 3.0

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            SensitivityResult(l1=-1.0, l2=1.0, linf=1.0)

    def test_infinite_rejected(self):
        with pytest.raises(ValueError):
            SensitivityResult(l1=float("inf"), l2=1.0, linf=1.0)

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            SensitivityResult(l1=float("nan"), l2=1.0, linf=1.0)

    def test_repr(self):
        r = repr(SensitivityResult(l1=1.0, l2=1.0, linf=1.0, query_type="counting"))
        assert "SensitivityResult" in r and "counting" in r


# ═══════════════════════════════════════════════════════════════════════════
# §5  adjacency_graph function
# ═══════════════════════════════════════════════════════════════════════════


class TestAdjacencyGraph:

    def test_identity_function_complete(self):
        result = adjacency_graph(lambda x: np.array([x]), [0, 1, 2])
        # Complete adjacency: max |i-j| over all pairs = 2
        assert result["sensitivity_l1"] == pytest.approx(2.0)

    def test_with_hamming_adjacency(self):
        adj = AdjacencyRelation.hamming_distance_1(4)
        result = adjacency_graph(lambda x: np.array([x]), [0, 1, 2, 3], adjacency=adj)
        assert result["sensitivity_l1"] == pytest.approx(1.0)

    def test_quadratic_function(self):
        result = adjacency_graph(lambda x: np.array([x, x ** 2]), [0, 1, 2])
        # diff(0,2)=[2,4], L1=6
        assert result["sensitivity_l1"] == pytest.approx(6.0)

    def test_empty_domain_raises(self):
        with pytest.raises(SensitivityError):
            adjacency_graph(lambda x: np.array([x]), [])

    def test_non_finite_raises(self):
        with pytest.raises(SensitivityError):
            adjacency_graph(lambda x: np.array([float("inf")]), [0, 1])

    def test_result_keys(self):
        result = adjacency_graph(lambda x: np.array([x]), [0, 1])
        for key in ("nodes", "values", "edges", "diffs",
                     "sensitivity_l1", "sensitivity_l2", "sensitivity_linf"):
            assert key in result

    def test_single_element_domain(self):
        result = adjacency_graph(lambda x: np.array([x]), [42])
        assert result["sensitivity_l1"] == 0.0 and len(result["edges"]) == 0

    def test_constant_function(self):
        result = adjacency_graph(lambda x: np.array([5.0, 5.0]), [0, 1, 2])
        assert result["sensitivity_l1"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# §6  CountingQuerySensitivity
# ═══════════════════════════════════════════════════════════════════════════


class TestCountingQuerySensitivity:

    def test_all_norms_equal_one(self):
        cqs = CountingQuerySensitivity(n=100)
        assert cqs.global_sensitivity_l1() == 1.0
        assert cqs.global_sensitivity_l2() == 1.0
        assert cqs.global_sensitivity_linf() == 1.0

    def test_local_sensitivity(self):
        assert CountingQuerySensitivity(n=10).local_sensitivity(5) == 1.0

    def test_analyze(self):
        result = CountingQuerySensitivity(n=50).analyze()
        assert isinstance(result, SensitivityResult)
        assert result.l1 == 1.0 and result.is_tight is True
        assert result.details["closed_form"] is True

    @pytest.mark.parametrize("n", [1, 10, 100, 1000])
    def test_invariant_of_n(self, n: int):
        assert CountingQuerySensitivity(n=n).global_sensitivity_l1() == 1.0

    def test_invalid_n(self):
        with pytest.raises(ConfigurationError):
            CountingQuerySensitivity(n=0)


# ═══════════════════════════════════════════════════════════════════════════
# §7  HistogramQuerySensitivity
# ═══════════════════════════════════════════════════════════════════════════


class TestHistogramQuerySensitivity:

    def test_add_remove_all_norms(self):
        hqs = HistogramQuerySensitivity(n_bins=10)
        assert hqs.global_sensitivity_l1() == 1.0
        assert hqs.global_sensitivity_l2() == 1.0
        assert hqs.global_sensitivity_linf() == 1.0

    def test_substitution_all_norms(self):
        hqs = HistogramQuerySensitivity(n_bins=10, adjacency_type="substitution")
        assert hqs.global_sensitivity_l1() == 2.0
        assert hqs.global_sensitivity_l2() == pytest.approx(math.sqrt(2.0))
        assert hqs.global_sensitivity_linf() == 1.0

    def test_analyze_add_remove(self):
        result = HistogramQuerySensitivity(n_bins=5).analyze()
        assert result.l1 == 1.0 and result.adjacency_type == "add_remove"

    def test_analyze_substitution(self):
        result = HistogramQuerySensitivity(n_bins=5, adjacency_type="substitution").analyze()
        assert result.l1 == 2.0 and result.adjacency_type == "substitution"

    def test_local_sensitivity_add_remove(self):
        assert HistogramQuerySensitivity(n_bins=5).local_sensitivity(np.array([3, 0, 1, 0, 2])) == 1.0

    def test_local_sensitivity_substitution_nonempty(self):
        hqs = HistogramQuerySensitivity(n_bins=5, adjacency_type="substitution")
        assert hqs.local_sensitivity(np.array([3, 0, 1, 0, 2])) == 2.0

    def test_local_sensitivity_substitution_empty(self):
        hqs = HistogramQuerySensitivity(n_bins=5, adjacency_type="substitution")
        assert hqs.local_sensitivity(np.zeros(5, dtype=np.int64)) == 0.0

    def test_local_sensitivity_substitution_single_bin(self):
        hqs = HistogramQuerySensitivity(n_bins=1, adjacency_type="substitution")
        assert hqs.local_sensitivity(np.array([5])) == 0.0

    def test_invalid_n_bins(self):
        with pytest.raises(ConfigurationError):
            HistogramQuerySensitivity(n_bins=0)

    def test_invalid_adjacency_type(self):
        with pytest.raises(ConfigurationError):
            HistogramQuerySensitivity(n_bins=5, adjacency_type="invalid")

    def test_build_adjacency(self):
        adj = HistogramQuerySensitivity(n_bins=3).build_adjacency(n_records=2)
        assert isinstance(adj, AdjacencyRelation) and adj.symmetric is True


# ═══════════════════════════════════════════════════════════════════════════
# §8  RangeQuerySensitivity
# ═══════════════════════════════════════════════════════════════════════════


class TestRangeQuerySensitivity:

    def test_single_range(self):
        rqs = RangeQuerySensitivity(d=10, range_type="single")
        assert rqs.global_sensitivity_l1() == 1.0
        assert rqs.global_sensitivity_l2() == 1.0
        assert rqs.global_sensitivity_linf() == 1.0

    @pytest.mark.parametrize("d", [1, 2, 5, 10, 100])
    def test_prefix_l1_equals_d(self, d: int):
        assert RangeQuerySensitivity(d=d, range_type="prefix").global_sensitivity_l1() == float(d)

    @pytest.mark.parametrize("d", [1, 4, 9, 16])
    def test_prefix_l2_equals_sqrt_d(self, d: int):
        rqs = RangeQuerySensitivity(d=d, range_type="prefix")
        assert rqs.global_sensitivity_l2() == pytest.approx(math.sqrt(d))

    def test_prefix_linf(self):
        assert RangeQuerySensitivity(d=10, range_type="prefix").global_sensitivity_linf() == 1.0

    def test_all_range_l1(self):
        # d=5: max over i of (i+1)*(5-i) -> i=2: 3*3=9
        assert RangeQuerySensitivity(d=5, range_type="all_range").global_sensitivity_l1() == 9.0

    def test_analyze(self):
        result = RangeQuerySensitivity(d=5, range_type="prefix").analyze()
        assert result.l1 == 5.0 and result.query_type == "range"
        assert result.details["range_type"] == "prefix"

    def test_invalid_d(self):
        with pytest.raises(ConfigurationError):
            RangeQuerySensitivity(d=0)

    def test_invalid_range_type(self):
        with pytest.raises(ConfigurationError):
            RangeQuerySensitivity(d=5, range_type="invalid")


# ═══════════════════════════════════════════════════════════════════════════
# §9  LinearWorkloadSensitivity
# ═══════════════════════════════════════════════════════════════════════════


class TestLinearWorkloadSensitivity:

    @pytest.mark.parametrize("d", [1, 2, 5, 10])
    def test_identity_all_norms(self, d: int):
        lws = LinearWorkloadSensitivity(np.eye(d))
        assert lws.global_sensitivity_l1() == 1.0
        assert lws.global_sensitivity_l2() == 1.0
        assert lws.global_sensitivity_linf() == 1.0

    def test_prefix_sum_matrix(self):
        d = 4
        lws = LinearWorkloadSensitivity(np.tril(np.ones((d, d))))
        assert lws.global_sensitivity_l1() == float(d)
        assert lws.global_sensitivity_l2() == pytest.approx(math.sqrt(d))
        assert lws.global_sensitivity_linf() == 1.0

    def test_dimensions(self):
        lws = LinearWorkloadSensitivity(np.ones((3, 5)))
        assert lws.m == 3 and lws.d == 5

    def test_column_sensitivities(self):
        lws = LinearWorkloadSensitivity(np.array([[1, 2], [3, 4]]))
        col_l1 = lws.column_sensitivities(norm_ord=1)
        assert col_l1[0] == pytest.approx(4.0)
        assert col_l1[1] == pytest.approx(6.0)

    def test_analyze(self):
        result = LinearWorkloadSensitivity(np.eye(3)).analyze()
        assert result.l1 == 1.0 and result.is_tight is True
        assert result.details["m"] == 3

    def test_sensitivity_from_workload_spec(self):
        lws = LinearWorkloadSensitivity(np.eye(1))
        result = lws.sensitivity_from_workload_spec(WorkloadSpec.identity(5))
        assert result.l1 == 1.0

    def test_non_square_matrix(self):
        lws = LinearWorkloadSensitivity(np.array([[1, 0, 1], [0, 1, 1]]))
        assert lws.global_sensitivity_l1() == 2.0

    def test_non_finite_raises(self):
        with pytest.raises(SensitivityError):
            LinearWorkloadSensitivity(np.array([[np.inf, 1], [0, 1]]))


# ═══════════════════════════════════════════════════════════════════════════
# §10  MarginalQuerySensitivity
# ═══════════════════════════════════════════════════════════════════════════


class TestMarginalQuerySensitivity:

    def test_single_marginal_add_remove(self):
        mqs = MarginalQuerySensitivity(d=5, k=2)
        assert mqs.global_sensitivity_l1() == 1.0
        assert mqs.global_sensitivity_l2() == 1.0
        assert mqs.global_sensitivity_linf() == 1.0

    def test_single_marginal_substitution(self):
        mqs = MarginalQuerySensitivity(d=5, k=2, adjacency_type="substitution")
        assert mqs.global_sensitivity_l1() == 2.0
        assert mqs.global_sensitivity_l2() == pytest.approx(math.sqrt(2.0))

    def test_all_marginals_add_remove(self):
        mqs = MarginalQuerySensitivity(d=5, k=2, all_marginals=True)
        assert mqs.global_sensitivity_l1() == 4.0  # C(4,1)

    def test_all_marginals_substitution(self):
        mqs = MarginalQuerySensitivity(d=5, k=2, adjacency_type="substitution", all_marginals=True)
        assert mqs.global_sensitivity_l1() == 8.0  # 2*C(4,1)

    def test_n_marginals_and_cells(self):
        mqs = MarginalQuerySensitivity(d=5, k=2)
        assert mqs.n_marginals == 10 and mqs.cells_per_marginal == 4

    def test_invalid_d(self):
        with pytest.raises(ConfigurationError):
            MarginalQuerySensitivity(d=0, k=1)

    def test_invalid_k(self):
        with pytest.raises(ConfigurationError):
            MarginalQuerySensitivity(d=3, k=4)


# ═══════════════════════════════════════════════════════════════════════════
# §11  CustomFunctionSensitivity
# ═══════════════════════════════════════════════════════════════════════════


class TestCustomFunctionSensitivity:

    def test_identity_exact(self):
        cfs = CustomFunctionSensitivity(lambda x: np.array([x]), domain=list(range(5)))
        result = cfs.analyze_exact()
        assert result.l1 == pytest.approx(1.0) and result.is_tight is True

    def test_quadratic_exact(self):
        cfs = CustomFunctionSensitivity(lambda x: np.array([x ** 2]), domain=list(range(5)))
        # Hamming-1: max |x^2 - (x+1)^2| -> x=3: |16-9|=7
        assert cfs.analyze_exact().l1 == pytest.approx(7.0)

    def test_exact_with_complete_adjacency(self):
        adj = AdjacencyRelation.complete(4)
        cfs = CustomFunctionSensitivity(lambda x: np.array([x]), domain=[0, 1, 2, 3], adjacency=adj)
        assert cfs.analyze_exact().l1 == pytest.approx(3.0)

    def test_exact_no_domain_raises(self):
        with pytest.raises(SensitivityError):
            CustomFunctionSensitivity(lambda x: np.array([x])).analyze_exact()

    def test_sampling_exhaustive(self):
        cfs = CustomFunctionSensitivity(lambda x: np.array([x]), domain=list(range(5)))
        result = cfs.analyze_sampling(n_samples=10000, seed=42)
        assert result.l1 == pytest.approx(1.0) and result.is_tight is True

    def test_sampling_no_domain_raises(self):
        with pytest.raises(SensitivityError):
            CustomFunctionSensitivity(lambda x: np.array([x])).analyze_sampling()

    def test_multi_dim_output(self):
        cfs = CustomFunctionSensitivity(lambda x: np.array([x, x * 2]), domain=list(range(5)))
        assert cfs.analyze_exact().l1 == pytest.approx(3.0)  # 1+2 = 3

    def test_constant_function(self):
        cfs = CustomFunctionSensitivity(lambda x: np.array([7.0]), domain=list(range(5)))
        assert cfs.analyze_exact().l1 == 0.0

    def test_symbolic_with_gradient(self):
        def f(x):
            return np.array([x[0] + x[1], 2 * x[0]])
        def grad(x):
            return np.array([[1.0, 1.0], [2.0, 0.0]])
        domain = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        result = CustomFunctionSensitivity(f, domain=domain).analyze_symbolic(gradient=grad)
        assert result.l1 == pytest.approx(3.0) and result.is_tight is False

    def test_numerical_gradient(self):
        cfs = CustomFunctionSensitivity(
            lambda x: np.array([x[0] ** 2]),
            domain=[np.array([1.0]), np.array([2.0]), np.array([3.0])],
        )
        result = cfs.analyze_symbolic()
        assert result.l1 == pytest.approx(6.0, rel=1e-4)

    def test_symbolic_no_domain_raises(self):
        with pytest.raises(SensitivityError):
            CustomFunctionSensitivity(lambda x: np.array([x])).analyze_symbolic()


# ═══════════════════════════════════════════════════════════════════════════
# §12  QuerySensitivityAnalyzer
# ═══════════════════════════════════════════════════════════════════════════


class TestQuerySensitivityAnalyzer:

    def test_counting_query(self):
        result = QuerySensitivityAnalyzer().analyze(QuerySpec.counting(n=5, epsilon=1.0))
        assert result.l1 == 1.0 and result.query_type == "counting"

    def test_histogram_add_remove(self):
        result = QuerySensitivityAnalyzer().analyze(QuerySpec.histogram(n_bins=10, epsilon=1.0))
        assert result.l1 == 1.0

    def test_histogram_substitution(self):
        spec = QuerySpec.histogram(n_bins=10, epsilon=1.0)
        spec.metadata["adjacency_type"] = "substitution"
        result = QuerySensitivityAnalyzer().analyze(spec)
        assert result.l1 == 2.0

    def test_range_query(self):
        spec = QuerySpec(
            query_values=np.arange(5, dtype=np.float64), domain="range",
            sensitivity=5.0, epsilon=1.0, query_type=QueryType.RANGE,
        )
        assert QuerySensitivityAnalyzer().analyze(spec).query_type == "range"

    def test_linear_workload_with_matrix(self):
        spec = QuerySpec(
            query_values=np.arange(5, dtype=np.float64), domain="wl",
            sensitivity=1.0, epsilon=1.0, query_type=QueryType.LINEAR_WORKLOAD,
            metadata={"workload_matrix": np.eye(5)},
        )
        result = QuerySensitivityAnalyzer().analyze(spec)
        assert result.l1 == 1.0 and result.is_tight is True

    def test_linear_workload_without_matrix(self):
        spec = QuerySpec(
            query_values=np.arange(5, dtype=np.float64), domain="wl",
            sensitivity=3.0, epsilon=1.0, query_type=QueryType.LINEAR_WORKLOAD,
        )
        result = QuerySensitivityAnalyzer().analyze(spec)
        assert result.l1 == 3.0 and result.is_tight is False

    def test_custom_query(self):
        spec = QuerySpec(
            query_values=np.array([0.0, 1.0, 4.0, 9.0]), domain="ints",
            sensitivity=5.0, epsilon=1.0, query_type=QueryType.CUSTOM,
        )
        assert QuerySensitivityAnalyzer().analyze(spec).query_type == "custom"

    def test_marginal_add_remove(self):
        spec = QuerySpec(
            query_values=np.arange(4, dtype=np.float64), domain="m",
            sensitivity=1.0, epsilon=1.0, query_type=QueryType.MARGINAL,
        )
        assert QuerySensitivityAnalyzer().analyze(spec).l1 == 1.0

    def test_marginal_substitution(self):
        spec = QuerySpec(
            query_values=np.arange(4, dtype=np.float64), domain="m",
            sensitivity=2.0, epsilon=1.0, query_type=QueryType.MARGINAL,
            metadata={"adjacency_type": "substitution", "k": 2},
        )
        assert QuerySensitivityAnalyzer().analyze(spec).l1 == 2.0


class TestAnalyzerSmoothSensitivity:

    def test_identity_positive(self):
        result = QuerySensitivityAnalyzer().compute_smooth_sensitivity(
            f=lambda x: float(x), beta=0.5, domain=list(range(5)),
        )
        assert result > 0.0 and math.isfinite(result)

    def test_constant_zero(self):
        assert QuerySensitivityAnalyzer().compute_smooth_sensitivity(
            f=lambda x: 5.0, beta=0.5, domain=list(range(5)),
        ) == 0.0

    def test_invalid_beta(self):
        with pytest.raises(ConfigurationError):
            QuerySensitivityAnalyzer().compute_smooth_sensitivity(
                f=lambda x: float(x), beta=0.0, domain=[0, 1],
            )

    def test_empty_domain(self):
        with pytest.raises(SensitivityError):
            QuerySensitivityAnalyzer().compute_smooth_sensitivity(
                f=lambda x: float(x), beta=0.5, domain=[],
            )

    def test_smooth_leq_global(self):
        domain = list(range(10))
        smooth = QuerySensitivityAnalyzer().compute_smooth_sensitivity(
            lambda x: float(x), beta=0.1, domain=domain,
        )
        adj = AdjacencyRelation.hamming_distance_1(len(domain))
        global_sens = max(
            abs(domain[i] - domain[j]) for i, j in adj.edges
        )
        assert smooth <= global_sens + 1e-10


class TestAnalyzerLocalSensitivity:

    def test_identity(self):
        result = QuerySensitivityAnalyzer().compute_local_sensitivity(
            f=lambda x: float(x), x=2, domain=list(range(5)), x_index=2,
        )
        assert result == pytest.approx(1.0)

    def test_quadratic(self):
        result = QuerySensitivityAnalyzer().compute_local_sensitivity(
            f=lambda x: float(x ** 2), x=3, domain=list(range(5)), x_index=3,
        )
        assert result == pytest.approx(7.0)  # max(|9-4|, |9-16|)

    def test_non_finite_raises(self):
        with pytest.raises(SensitivityError):
            QuerySensitivityAnalyzer().compute_local_sensitivity(
                f=lambda x: float("inf"), x=0, domain=[0, 1, 2], x_index=0,
            )


class TestAnalyzerLipschitz:

    def test_identity_lipschitz(self):
        result = QuerySensitivityAnalyzer().lipschitz_sensitivity(
            lambda x: np.array([x[0]]),
            np.linspace(0, 1, 20).reshape(-1, 1), seed=42,
        )
        assert result.l1 > 0.0 and result.is_tight is False

    def test_too_few_points(self):
        with pytest.raises(SensitivityError):
            QuerySensitivityAnalyzer().lipschitz_sensitivity(
                lambda x: np.array([x[0]]), np.array([[1.0]]),
            )


# ═══════════════════════════════════════════════════════════════════════════
# §13  workload_sensitivity convenience function
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkloadSensitivity:

    def test_identity(self):
        result = workload_sensitivity(WorkloadSpec.identity(5))
        assert result.l1 == 1.0 and result.l2 == 1.0 and result.linf == 1.0

    @pytest.mark.parametrize("d", [1, 2, 5, 10])
    def test_prefix_sum(self, d: int):
        result = workload_sensitivity(WorkloadSpec.all_range(d))
        assert result.l1 == pytest.approx(float(d))
        assert result.l2 == pytest.approx(math.sqrt(d))
        assert result.linf == pytest.approx(1.0)

    def test_custom_workload(self):
        A = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        assert workload_sensitivity(WorkloadSpec(matrix=A)).l1 == 2.0


# ═══════════════════════════════════════════════════════════════════════════
# §14  query_spec_sensitivity & validate_sensitivity
# ═══════════════════════════════════════════════════════════════════════════


class TestQuerySpecSensitivity:

    def test_counting(self):
        result = query_spec_sensitivity(QuerySpec.counting(n=5, epsilon=1.0))
        assert result.l1 == 1.0 and result.query_type == "counting"

    def test_histogram(self):
        result = query_spec_sensitivity(QuerySpec.histogram(n_bins=10, epsilon=1.0))
        assert result.l1 == 1.0


class TestValidateSensitivity:

    def test_valid_exact(self):
        computed = SensitivityResult(l1=1.0, l2=1.0, linf=1.0)
        assert validate_sensitivity(1.0, computed, norm="l1") is True

    def test_valid_overestimate(self):
        computed = SensitivityResult(l1=1.0, l2=1.0, linf=1.0)
        assert validate_sensitivity(2.0, computed, norm="l1") is True

    def test_invalid_underestimate(self):
        with pytest.raises(SensitivityError):
            validate_sensitivity(0.5, SensitivityResult(l1=1.0, l2=1.0, linf=1.0), norm="l1")

    def test_l2_norm(self):
        assert validate_sensitivity(1.5, SensitivityResult(l1=2.0, l2=1.414, linf=1.0), norm="l2")

    def test_tolerance(self):
        computed = SensitivityResult(l1=1.0, l2=1.0, linf=1.0)
        assert validate_sensitivity(1.0 - 1e-12, computed, norm="l1", tolerance=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# §15  sensitivity_from_function
# ═══════════════════════════════════════════════════════════════════════════


class TestSensitivityFromFunction:

    def test_complete_adjacency(self):
        result = sensitivity_from_function(lambda x: np.array([x]), [0, 1, 2, 3])
        assert result.l1 == pytest.approx(3.0)  # max |i-j| = 3

    def test_with_hamming_adjacency(self):
        adj = AdjacencyRelation.hamming_distance_1(4)
        result = sensitivity_from_function(lambda x: np.array([x]), [0, 1, 2, 3], adjacency=adj)
        assert result.l1 == pytest.approx(1.0)

    def test_quadratic(self):
        adj = AdjacencyRelation.hamming_distance_1(5)
        result = sensitivity_from_function(
            lambda x: np.array([x ** 2]), list(range(5)), adjacency=adj,
        )
        assert result.l1 == pytest.approx(7.0)

    def test_multi_dim(self):
        adj = AdjacencyRelation.hamming_distance_1(4)
        result = sensitivity_from_function(
            lambda x: np.array([x, x ** 2]), [0, 1, 2, 3], adjacency=adj,
        )
        # diff at (2,3): [1, 5], L1=6
        assert result.l1 == pytest.approx(6.0)

    def test_result_type(self):
        result = sensitivity_from_function(lambda x: np.array([x]), [0, 1])
        assert isinstance(result, SensitivityResult) and result.is_tight is True
        assert result.details["method"] == "exhaustive_enumeration"

    def test_single_element_domain(self):
        result = sensitivity_from_function(lambda x: np.array([x]), [42])
        assert result.l1 == 0.0

    def test_constant_function(self):
        result = sensitivity_from_function(lambda x: np.array([42.0]), list(range(5)))
        assert result.l1 == 0.0

    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_linear_various_sizes(self, n: int):
        adj = AdjacencyRelation.hamming_distance_1(n)
        result = sensitivity_from_function(lambda x: np.array([x]), list(range(n)), adjacency=adj)
        assert result.l1 == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# §16  Edge cases and integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_1x1_matrix(self):
        A = np.array([[5.0]])
        assert sensitivity_l1(A) == 5.0
        assert sensitivity_l2(A) == 5.0
        assert sensitivity_linf(A) == 5.0

    def test_norm_ordering(self):
        A = np.array([[1, 1], [1, 0], [0, 1]])
        linf, l2, l1 = sensitivity_linf(A), sensitivity_l2(A), sensitivity_l1(A)
        assert linf <= l2 + 1e-10
        assert l2 <= l1 + 1e-10

    @pytest.mark.parametrize("n", [2, 5, 10, 20])
    def test_norm_ordering_random(self, n: int):
        A = np.random.default_rng(42 + n).standard_normal((n, n))
        linf, l2, l1 = sensitivity_linf(A), sensitivity_l2(A), sensitivity_l1(A)
        assert linf <= l2 + 1e-10 and l2 <= l1 + 1e-10

    def test_adjacency_edge_beyond_columns(self):
        adj = AdjacencyRelation(edges=[(0, 1), (3, 4)], n=5, symmetric=True)
        assert sensitivity_l1(np.eye(3), adjacency=adj) == pytest.approx(2.0)

    def test_consistency_counting(self):
        cqs_result = CountingQuerySensitivity(n=10).analyze()
        analyzer_result = QuerySensitivityAnalyzer().analyze(QuerySpec.counting(n=10, epsilon=1.0))
        assert cqs_result.l1 == analyzer_result.l1

    def test_consistency_histogram(self):
        hqs_result = HistogramQuerySensitivity(n_bins=10).analyze()
        analyzer_result = QuerySensitivityAnalyzer().analyze(QuerySpec.histogram(n_bins=10, epsilon=1.0))
        assert hqs_result.l1 == analyzer_result.l1

    def test_consistency_workload_function(self):
        A = np.array([[1, 0, 1], [0, 1, 1]])
        result = workload_sensitivity(WorkloadSpec(matrix=A))
        assert result.l1 == sensitivity_l1(A)
        assert result.l2 == sensitivity_l2(A)

    def test_sparse_like_matrix(self):
        A = np.zeros((10, 10))
        A[5, 3] = 2.0
        assert sensitivity_l1(A) == 2.0


# ═══════════════════════════════════════════════════════════════════════════
# §17  Parametrized cross-cutting tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossCuttingParametrized:

    @pytest.mark.parametrize(
        "matrix_fn, l1, l2, linf",
        [
            (lambda: np.eye(3), 1.0, 1.0, 1.0),
            (lambda: np.ones((3, 3)), 3.0, math.sqrt(3.0), 1.0),
            (lambda: np.zeros((3, 3)), 0.0, 0.0, 0.0),
            (lambda: np.diag([1, 2, 3]), 3.0, 3.0, 3.0),
        ],
        ids=["identity", "ones", "zeros", "diagonal"],
    )
    def test_all_norms(self, matrix_fn, l1, l2, linf):
        A = matrix_fn()
        assert sensitivity_l1(A) == pytest.approx(l1)
        assert sensitivity_l2(A) == pytest.approx(l2)
        assert sensitivity_linf(A) == pytest.approx(linf)

    @pytest.mark.parametrize("d", [1, 2, 3, 4, 5, 8, 10])
    def test_prefix_sum_all_norms(self, d: int):
        A = np.tril(np.ones((d, d)))
        assert sensitivity_l1(A) == pytest.approx(float(d))
        assert sensitivity_l2(A) == pytest.approx(math.sqrt(d))
        assert sensitivity_linf(A) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "adj_builder, n, expected",
        [
            (bounded_adjacency, 5, 4),
            (add_remove_adjacency, 5, 4),
            (substitution_adjacency, 5, 10),
            (bounded_adjacency, 1, 0),
        ],
    )
    def test_adjacency_edge_counts(self, adj_builder, n, expected):
        assert len(adj_builder(n).edges) == expected

    @pytest.mark.parametrize(
        "query_class, kwargs, expected_l1",
        [
            (CountingQuerySensitivity, {"n": 10}, 1.0),
            (HistogramQuerySensitivity, {"n_bins": 10}, 1.0),
            (HistogramQuerySensitivity, {"n_bins": 10, "adjacency_type": "substitution"}, 2.0),
            (RangeQuerySensitivity, {"d": 10, "range_type": "single"}, 1.0),
            (RangeQuerySensitivity, {"d": 10, "range_type": "prefix"}, 10.0),
        ],
    )
    def test_query_class_l1(self, query_class, kwargs, expected_l1):
        assert query_class(**kwargs).global_sensitivity_l1() == pytest.approx(expected_l1)
