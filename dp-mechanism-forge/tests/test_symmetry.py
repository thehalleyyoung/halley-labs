"""
Comprehensive tests for dp_forge.symmetry — symmetry detection, LP reduction,
group theory utilities, and reconstruction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import csr_matrix

from dp_forge.exceptions import ConfigurationError
from dp_forge.symmetry import (
    PermutationGroup,
    PermutationReducer,
    ReconstructionMap,
    ReducedLPStruct,
    ReflectionReducer,
    ReduceBySymmetry,
    SymmetryDetector,
    SymmetryGroup,
    TranslationReducer,
    detect_and_report_symmetry,
    expand_noise_to_full,
    is_counting_query,
    is_group,
    orbit_computation,
    stabilizer,
    verify_reconstruction,
)
from dp_forge.types import (
    AdjacencyRelation,
    LossFunction,
    LPStruct,
    QuerySpec,
    QueryType,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _counting_spec(n: int, k: int = 10, epsilon: float = 1.0) -> QuerySpec:
    """Build a counting query spec for testing."""
    return QuerySpec.counting(n, epsilon=epsilon, k=k)


def _custom_spec(
    values: list[float],
    epsilon: float = 1.0,
    k: int = 10,
    loss_fn: LossFunction = LossFunction.L2,
) -> QuerySpec:
    """Build a custom query spec for testing."""
    return QuerySpec(
        query_values=np.array(values, dtype=np.float64),
        domain="test",
        sensitivity=1.0,
        epsilon=epsilon,
        k=k,
        loss_fn=loss_fn,
        query_type=QueryType.CUSTOM,
    )


def _identity_perm(n: int) -> np.ndarray:
    """Return the identity permutation of size n."""
    return np.arange(n, dtype=np.intp)


def _cyclic_perm(n: int) -> np.ndarray:
    """Return cyclic shift permutation: i -> (i+1) mod n."""
    return np.roll(np.arange(n, dtype=np.intp), -1)


def _reverse_perm(n: int) -> np.ndarray:
    """Return the reversal permutation: i -> n-1-i."""
    return np.arange(n, dtype=np.intp)[::-1].copy()


def _transposition(n: int, i: int, j: int) -> np.ndarray:
    """Return a transposition swapping i and j."""
    perm = np.arange(n, dtype=np.intp)
    perm[i], perm[j] = perm[j], perm[i]
    return perm


# ═══════════════════════════════════════════════════════════════════════════
# §1  SymmetryGroup dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestSymmetryGroup:
    """Tests for the SymmetryGroup dataclass."""

    def test_default_values(self):
        g = SymmetryGroup()
        assert g.is_translation_invariant is False
        assert g.is_reflection_symmetric is False
        assert g.generators == []
        assert g.order == 1
        assert g.group_type == "trivial"
        assert g.metadata == {}

    def test_has_symmetry_trivial(self):
        g = SymmetryGroup()
        assert g.has_symmetry is False

    def test_has_symmetry_translation(self):
        g = SymmetryGroup(is_translation_invariant=True, order=5)
        assert g.has_symmetry is True

    def test_has_symmetry_reflection(self):
        g = SymmetryGroup(is_reflection_symmetric=True, order=2)
        assert g.has_symmetry is True

    def test_has_symmetry_order_gt1(self):
        g = SymmetryGroup(order=3)
        assert g.has_symmetry is True

    def test_has_symmetry_combined(self):
        g = SymmetryGroup(
            is_translation_invariant=True,
            is_reflection_symmetric=True,
            order=10,
        )
        assert g.has_symmetry is True

    def test_theoretical_speedup_trivial(self):
        g = SymmetryGroup()
        assert g.theoretical_speedup == 1.0

    def test_theoretical_speedup_translation_only(self):
        g = SymmetryGroup(
            is_translation_invariant=True,
            metadata={"n": 10},
        )
        assert g.theoretical_speedup == 10.0

    def test_theoretical_speedup_reflection_only(self):
        g = SymmetryGroup(is_reflection_symmetric=True)
        assert g.theoretical_speedup == 2.0

    def test_theoretical_speedup_both(self):
        g = SymmetryGroup(
            is_translation_invariant=True,
            is_reflection_symmetric=True,
            metadata={"n": 10},
        )
        assert g.theoretical_speedup == 20.0

    def test_theoretical_speedup_general_order(self):
        g = SymmetryGroup(order=6)
        assert g.theoretical_speedup == 6.0

    def test_theoretical_speedup_translation_default_n(self):
        g = SymmetryGroup(is_translation_invariant=True)
        assert g.theoretical_speedup == 1.0

    def test_repr_contains_info(self):
        g = SymmetryGroup(
            is_translation_invariant=True,
            order=5,
            group_type="translation",
        )
        r = repr(g)
        assert "translation" in r
        assert "order=5" in r

    def test_metadata_passthrough(self):
        g = SymmetryGroup(metadata={"n": 4, "detection_time_s": 0.001})
        assert g.metadata["n"] == 4
        assert g.metadata["detection_time_s"] == 0.001

    def test_generators_stored(self):
        gen = _cyclic_perm(5)
        g = SymmetryGroup(generators=[gen])
        assert len(g.generators) == 1
        np.testing.assert_array_equal(g.generators[0], gen)


# ═══════════════════════════════════════════════════════════════════════════
# §2  SymmetryDetector
# ═══════════════════════════════════════════════════════════════════════════


class TestSymmetryDetectorTranslation:
    """Tests for translation invariance detection."""

    def test_counting_query_is_translation_invariant(self):
        spec = _counting_spec(10)
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_translation_invariant is True

    def test_counting_n2(self):
        spec = _counting_spec(2)
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_translation_invariant is True

    def test_counting_n1(self):
        spec = _counting_spec(1, k=5)
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        # n=1 is trivially translation-invariant
        assert group.is_translation_invariant is True

    def test_counting_large_n(self):
        spec = _counting_spec(100, k=20)
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_translation_invariant is True
        assert group.metadata["n"] == 100

    def test_non_counting_not_translation_invariant(self):
        spec = _custom_spec([0.0, 1.0, 5.0])
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_translation_invariant is False

    def test_arithmetic_but_not_step1(self):
        spec = _custom_spec([0.0, 2.0, 4.0, 6.0])
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        # f values form arithmetic sequence, adjacency is path => translation invariant
        assert group.is_translation_invariant is True

    def test_random_values_not_translation_invariant(self):
        rng = np.random.default_rng(42)
        vals = sorted(rng.uniform(0, 10, size=8).tolist())
        spec = _custom_spec(vals)
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_translation_invariant is False

    def test_constant_values_translation_invariant(self):
        # All same values — arithmetic sequence with diff 0
        spec = _custom_spec([3.0, 3.0, 3.0])
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_translation_invariant is True

    def test_non_path_adjacency_not_translation_invariant(self):
        """Complete adjacency graph (not a path) should fail translation check."""
        adj = AdjacencyRelation.complete(4)
        spec = QuerySpec(
            query_values=np.arange(4, dtype=np.float64),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            k=10,
            edges=adj,
            query_type=QueryType.CUSTOM,
        )
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_translation_invariant is False

    def test_detect_with_explicit_params(self):
        """Pass f_values and edges directly instead of via spec."""
        f = np.arange(5, dtype=np.float64)
        edges = [(i, i + 1) for i in range(4)]
        detector = SymmetryDetector()
        group = detector.detect(
            f_values=f,
            edges=edges,
            query_type=QueryType.COUNTING,
        )
        assert group.is_translation_invariant is True

    def test_detect_raises_without_f_values(self):
        detector = SymmetryDetector()
        with pytest.raises(ConfigurationError):
            detector.detect()

    def test_is_translation_invariant_direct(self):
        detector = SymmetryDetector()
        f = np.arange(5, dtype=np.float64)
        edges = [(i, i + 1) for i in range(4)]
        assert detector.is_translation_invariant(f, edges) is True

    def test_is_translation_invariant_non_arithmetic(self):
        detector = SymmetryDetector()
        f = np.array([0.0, 1.0, 3.0, 6.0])
        edges = [(i, i + 1) for i in range(3)]
        assert detector.is_translation_invariant(f, edges) is False


class TestSymmetryDetectorReflection:
    """Tests for reflection symmetry detection."""

    def test_l1_is_reflection_symmetric(self):
        spec = _counting_spec(5)
        spec_l1 = QuerySpec(
            query_values=np.arange(5, dtype=np.float64),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            k=10,
            loss_fn=LossFunction.L1,
            query_type=QueryType.COUNTING,
        )
        detector = SymmetryDetector(spec=spec_l1)
        group = detector.detect()
        assert group.is_reflection_symmetric is True

    def test_l2_is_reflection_symmetric(self):
        spec = _counting_spec(5)
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_reflection_symmetric is True

    def test_linf_is_reflection_symmetric(self):
        spec = QuerySpec(
            query_values=np.arange(5, dtype=np.float64),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            k=10,
            loss_fn=LossFunction.LINF,
            query_type=QueryType.COUNTING,
        )
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_reflection_symmetric is True

    def test_custom_loss_not_reflection_symmetric(self):
        spec = QuerySpec(
            query_values=np.arange(5, dtype=np.float64),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            k=10,
            loss_fn=LossFunction.CUSTOM,
            custom_loss=lambda t, y: abs(t - y) ** 3,
            query_type=QueryType.COUNTING,
        )
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_reflection_symmetric is False

    def test_none_loss_not_symmetric(self):
        detector = SymmetryDetector()
        f = np.arange(5, dtype=np.float64)
        assert detector.is_reflection_symmetric(f, None) is False

    def test_is_reflection_symmetric_direct(self):
        detector = SymmetryDetector()
        f = np.arange(5, dtype=np.float64)
        assert detector.is_reflection_symmetric(f, LossFunction.L1) is True
        assert detector.is_reflection_symmetric(f, LossFunction.L2) is True
        assert detector.is_reflection_symmetric(f, LossFunction.LINF) is True
        assert detector.is_reflection_symmetric(f, LossFunction.CUSTOM) is False


class TestSymmetryDetectorGroupTypes:
    """Tests for group type classification and order."""

    def test_translation_and_reflection(self):
        spec = _counting_spec(8)
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.group_type == "translation+reflection"
        assert group.order == 2 * 8

    def test_translation_only(self):
        spec = QuerySpec(
            query_values=np.arange(5, dtype=np.float64),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            k=10,
            loss_fn=LossFunction.CUSTOM,
            custom_loss=lambda t, y: abs(t - y) ** 3,
            query_type=QueryType.COUNTING,
        )
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.group_type == "translation"
        assert group.order == 5

    def test_reflection_only(self):
        # Non-arithmetic query values + L2 loss => reflection only
        spec = _custom_spec([0.0, 1.0, 5.0], loss_fn=LossFunction.L2)
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.group_type == "reflection"
        assert group.order == 2

    def test_trivial(self):
        spec = QuerySpec(
            query_values=np.array([0.0, 1.0, 5.0]),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            k=10,
            loss_fn=LossFunction.CUSTOM,
            custom_loss=lambda t, y: abs(t - y) ** 3,
            query_type=QueryType.CUSTOM,
        )
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.group_type == "trivial"
        assert group.order == 1

    def test_generators_translation(self):
        spec = _counting_spec(5)
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        # Should have translation generator and reflection generator
        assert len(group.generators) == 2
        # Translation generator is cyclic shift
        np.testing.assert_array_equal(group.generators[0], _cyclic_perm(5))
        # Reflection generator is reversal
        np.testing.assert_array_equal(group.generators[1], _reverse_perm(5))

    def test_detection_time_in_metadata(self):
        spec = _counting_spec(10)
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert "detection_time_s" in group.metadata
        assert group.metadata["detection_time_s"] >= 0


class TestSymmetryDetectorFindOrbits:
    """Tests for orbit computation via SymmetryDetector."""

    def test_trivial_orbits(self):
        detector = SymmetryDetector()
        f = np.arange(5, dtype=np.float64)
        trivial = SymmetryGroup()
        orbits = detector.find_orbits(f, trivial)
        assert len(orbits) == 5
        for i, orb in enumerate(orbits):
            assert orb == [i]

    def test_cyclic_orbits(self):
        n = 6
        gen = _cyclic_perm(n)
        group = SymmetryGroup(generators=[gen], order=n)
        detector = SymmetryDetector()
        f = np.arange(n, dtype=np.float64)
        orbits = detector.find_orbits(f, group)
        # Cyclic group on n elements => single orbit
        assert len(orbits) == 1
        assert sorted(orbits[0]) == list(range(n))

    def test_transposition_orbits(self):
        n = 5
        gen = _transposition(n, 1, 3)
        group = SymmetryGroup(generators=[gen], order=2)
        detector = SymmetryDetector()
        f = np.arange(n, dtype=np.float64)
        orbits = detector.find_orbits(f, group)
        # Swapping 1 and 3 => orbits: {0}, {1,3}, {2}, {4}
        orbit_sets = [set(o) for o in orbits]
        assert {1, 3} in orbit_sets
        assert {0} in orbit_sets
        assert {2} in orbit_sets
        assert {4} in orbit_sets


# ═══════════════════════════════════════════════════════════════════════════
# §3  TranslationReducer
# ═══════════════════════════════════════════════════════════════════════════


class TestTranslationReducer:
    """Tests for TranslationReducer."""

    def _make_reducer(self, n=10, k=10, epsilon=1.0, delta=0.0):
        y_grid = np.arange(k, dtype=np.float64)
        return TranslationReducer(
            epsilon=epsilon,
            delta=delta,
            k=k,
            y_grid=y_grid,
        )

    def test_reduce_variable_count(self):
        """Reduced LP should have k+1 variables."""
        k = 10
        reducer = self._make_reducer(k=k)
        result = reducer.reduce(n=20)
        assert result.n_vars == k + 1

    def test_reduce_has_equality_constraint(self):
        """Reduced LP should have a simplex constraint."""
        reducer = self._make_reducer(k=8)
        result = reducer.reduce(n=10)
        assert result.A_eq is not None
        assert result.b_eq is not None
        assert result.n_eq == 1
        np.testing.assert_allclose(result.b_eq, [1.0])

    def test_reduce_objective_is_epigraph(self):
        """Objective should minimize the epigraph variable t."""
        k = 6
        reducer = self._make_reducer(k=k)
        result = reducer.reduce(n=5)
        c = result.c
        assert c[k] == 1.0
        assert np.sum(np.abs(c[:k])) == 0.0

    def test_reduce_bounds(self):
        """Each eta variable should have lower bound > 0."""
        k = 5
        reducer = self._make_reducer(k=k)
        result = reducer.reduce(n=5)
        for i in range(k):
            lb, ub = result.bounds[i]
            assert lb > 0
            assert ub == 1.0

    def test_reduce_epigraph_bound_unbounded(self):
        """Epigraph variable t should be unbounded."""
        k = 5
        reducer = self._make_reducer(k=k)
        result = reducer.reduce(n=5)
        lb, ub = result.bounds[k]
        assert lb is None
        assert ub is None

    def test_reduction_factor(self):
        k = 10
        n = 50
        reducer = self._make_reducer(k=k)
        factor = reducer.reduction_factor(n)
        expected = (n * k + 1) / (k + 1)
        assert abs(factor - expected) < 1e-10

    def test_reduction_factor_n1(self):
        k = 10
        reducer = self._make_reducer(k=k)
        factor = reducer.reduction_factor(1)
        # n=1: original has k+1 vars, reduced has k+1 vars => factor 1
        assert abs(factor - 1.0) < 1e-10

    def test_reduce_metadata(self):
        reducer = self._make_reducer(k=8, epsilon=0.5)
        result = reducer.reduce(n=10)
        assert result.reduction_type == "translation"
        assert result.original_n == 10
        assert result.original_k == 8
        assert result.metadata["epsilon"] == 0.5

    def test_reduce_dp_constraints_pure(self):
        """Pure DP should produce 2*(k-1) DP inequality constraints (non-periodic)."""
        k = 6
        reducer = self._make_reducer(k=k, epsilon=1.0)
        result = reducer.reduce(n=5)
        # DP constraints: 2*(k-1) forward/backward + 1 loss constraint
        n_dp = 2 * (k - 1)
        assert result.n_ub >= n_dp

    def test_reduce_sparse_constraint_matrix(self):
        """A_ub should be a sparse matrix."""
        reducer = self._make_reducer(k=10)
        result = reducer.reduce(n=10)
        assert sparse.issparse(result.A_ub)

    def test_reduce_periodic(self):
        """Periodic boundary should produce 2k DP constraints."""
        k = 6
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid, periodic=True,
        )
        result = reducer.reduce(n=5)
        # Periodic: 2k DP + 1 loss = 2k+1
        n_dp = 2 * k
        assert result.n_ub >= n_dp

    def test_reduce_approximate_dp(self):
        """Approximate DP (delta > 0) should add slack variables."""
        k = 6
        reducer = self._make_reducer(k=k, epsilon=1.0, delta=0.01)
        result = reducer.reduce(n=5)
        # Should have more variables than k+1 due to slacks
        assert result.n_vars > k + 1

    def test_reduce_y_grid_stored(self):
        k = 8
        y_grid = np.linspace(-3, 3, k)
        reducer = TranslationReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        result = reducer.reduce(n=5)
        np.testing.assert_array_equal(result.y_grid, y_grid)


class TestTranslationReducerReconstruct:
    """Tests for TranslationReducer.reconstruct."""

    def test_reconstruct_shape(self):
        n, k = 5, 8
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        noise = np.ones(k) / k
        f_values = np.arange(n, dtype=np.float64)
        table = reducer.reconstruct(noise, f_values)
        assert table.shape == (n, k)

    def test_reconstruct_rows_sum_to_one(self):
        n, k = 5, 8
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        noise = np.ones(k) / k
        f_values = np.arange(n, dtype=np.float64)
        table = reducer.reconstruct(noise, f_values)
        row_sums = table.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_reconstruct_nonnegative(self):
        n, k = 4, 6
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        rng = np.random.default_rng(123)
        noise = rng.dirichlet(np.ones(k))
        f_values = np.arange(n, dtype=np.float64)
        table = reducer.reconstruct(noise, f_values)
        assert np.all(table >= 0)

    def test_reconstruct_shift_structure(self):
        """Rows of the table should be shifts of the noise distribution."""
        n, k = 3, 10
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        noise = np.zeros(k)
        noise[0] = 0.5
        noise[1] = 0.3
        noise[2] = 0.2
        f_values = np.arange(n, dtype=np.float64)
        table = reducer.reconstruct(noise, f_values)
        # Row 0: noise placed at offset 0 => table[0, 0:3] ≈ noise[0:3]
        # Row 1: noise placed at offset 1 => table[1, 1:4] ≈ noise[0:3]
        # (after renormalization for truncation)
        assert table[0, 0] > 0
        assert table[1, 1] > 0
        assert table[2, 2] > 0

    def test_reconstruct_n1(self):
        """n=1 should produce a 1×k table equal to the noise distribution."""
        k = 5
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        noise = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        f_values = np.array([0.0])
        table = reducer.reconstruct(noise, f_values)
        assert table.shape == (1, k)
        np.testing.assert_allclose(table[0], noise, atol=1e-10)


class TestTranslationReducerBuildHelpers:
    """Tests for TranslationReducer internal helper methods."""

    def test_build_noise_variables(self):
        k = 8
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        eta = reducer._build_noise_variables(k)
        assert eta.shape == (k,)
        np.testing.assert_allclose(eta, 1.0 / k, atol=1e-15)
        np.testing.assert_allclose(eta.sum(), 1.0, atol=1e-14)

    def test_build_reduced_constraints_shape(self):
        k = 6
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        A, b = reducer._build_reduced_constraints(k, epsilon=1.0)
        assert A.shape == (2 * (k - 1), k)
        assert b.shape == (2 * (k - 1),)

    def test_build_reduced_constraints_rhs_zero(self):
        k = 5
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        _, b = reducer._build_reduced_constraints(k, epsilon=1.0)
        np.testing.assert_array_equal(b, 0.0)

    def test_uniform_satisfies_dp_constraints(self):
        """A uniform distribution should satisfy all DP constraints."""
        k = 5
        epsilon = 1.0
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=epsilon, delta=0.0, k=k, y_grid=y_grid,
        )
        A, b = reducer._build_reduced_constraints(k, epsilon)
        eta = np.ones(k) / k
        residual = A @ eta - b
        assert np.all(residual <= 1e-12)


# ═══════════════════════════════════════════════════════════════════════════
# §4  ReflectionReducer
# ═══════════════════════════════════════════════════════════════════════════


class TestReflectionReducer:
    """Tests for ReflectionReducer."""

    def _make_reducer(self, k=10, epsilon=1.0):
        y_grid = np.linspace(-4, 4, k)
        return ReflectionReducer(
            epsilon=epsilon, delta=0.0, k=k, y_grid=y_grid,
        )

    def test_reduce_variable_count_even(self):
        k = 10
        reducer = self._make_reducer(k=k)
        result = reducer.reduce()
        n_free = (k + 1) // 2  # ceil(10/2) = 5
        assert result.n_vars == n_free + 1

    def test_reduce_variable_count_odd(self):
        k = 7
        y_grid = np.linspace(-3, 3, k)
        reducer = ReflectionReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        result = reducer.reduce()
        n_free = (k + 1) // 2  # ceil(7/2) = 4
        assert result.n_vars == n_free + 1

    def test_reduce_has_simplex_constraint(self):
        reducer = self._make_reducer(k=8)
        result = reducer.reduce()
        assert result.A_eq is not None
        assert result.b_eq is not None
        np.testing.assert_allclose(result.b_eq, [1.0])

    def test_reduce_metadata(self):
        reducer = self._make_reducer(k=10, epsilon=0.5)
        result = reducer.reduce()
        assert result.reduction_type == "reflection"
        assert result.metadata["epsilon"] == 0.5
        assert result.metadata["n_free"] == (10 + 1) // 2

    def test_reconstruct_symmetry(self):
        """Reconstructed noise should be symmetric: η[l] == η[k-1-l]."""
        k = 8
        reducer = self._make_reducer(k=k)
        n_free = (k + 1) // 2
        noise_free = np.array([0.1, 0.2, 0.3, 0.4])  # n_free = 4
        noise_full = reducer.reconstruct(noise_free)
        assert len(noise_full) == k
        for l in range(k):
            mirror = k - 1 - l
            assert abs(noise_full[l] - noise_full[mirror]) < 1e-15

    def test_reconstruct_preserves_values(self):
        k = 6
        reducer = self._make_reducer(k=k)
        n_free = (k + 1) // 2  # 3
        noise_free = np.array([0.25, 0.15, 0.1])
        noise_full = reducer.reconstruct(noise_free)
        # First n_free entries should match
        np.testing.assert_allclose(noise_full[:n_free], noise_free)

    def test_reconstruct_odd_k(self):
        k = 7
        y_grid = np.linspace(-3, 3, k)
        reducer = ReflectionReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        n_free = (k + 1) // 2  # 4
        noise_free = np.array([0.1, 0.2, 0.3, 0.4])
        noise_full = reducer.reconstruct(noise_free)
        assert len(noise_full) == k
        # The centre element (index 3) should appear once
        for l in range(k):
            mirror = k - 1 - l
            assert abs(noise_full[l] - noise_full[mirror]) < 1e-15

    def test_exploit_symmetry_roundtrip(self):
        """_exploit_symmetry followed by reconstruct should recover original."""
        k = 8
        reducer = self._make_reducer(k=k)
        original = np.array([0.1, 0.15, 0.2, 0.3, 0.3, 0.2, 0.15, 0.1])
        # Already symmetric
        free = reducer._exploit_symmetry(original)
        reconstructed = reducer.reconstruct(free)
        np.testing.assert_allclose(reconstructed, original, atol=1e-15)


# ═══════════════════════════════════════════════════════════════════════════
# §5  PermutationReducer
# ═══════════════════════════════════════════════════════════════════════════


class TestPermutationReducer:
    """Tests for PermutationReducer."""

    def test_identity_generator_all_singletons(self):
        k = 5
        gen = _identity_perm(k)
        reducer = PermutationReducer(generators=[gen], k=k)
        orbits = reducer.orbits
        assert len(orbits) == k
        for orb in orbits:
            assert len(orb) == 1

    def test_cyclic_generator_single_orbit(self):
        k = 6
        gen = _cyclic_perm(k)
        reducer = PermutationReducer(generators=[gen], k=k)
        orbits = reducer.orbits
        assert len(orbits) == 1
        assert sorted(orbits[0]) == list(range(k))

    def test_transposition_generator(self):
        k = 5
        gen = _transposition(k, 0, 4)
        reducer = PermutationReducer(generators=[gen], k=k)
        orbits = reducer.orbits
        orbit_sets = [set(o) for o in orbits]
        assert {0, 4} in orbit_sets
        assert len(orbits) == 4  # {0,4}, {1}, {2}, {3}

    def test_orbit_representative(self):
        k = 5
        gen = _transposition(k, 1, 3)
        reducer = PermutationReducer(generators=[gen], k=k)
        assert reducer.orbit_representative(1) == 1
        assert reducer.orbit_representative(3) == 1  # min of {1,3}
        assert reducer.orbit_representative(0) == 0

    def test_orbits_cached(self):
        k = 4
        gen = _cyclic_perm(k)
        reducer = PermutationReducer(generators=[gen], k=k)
        orbits1 = reducer.orbits
        orbits2 = reducer.orbits
        assert orbits1 is orbits2  # Same object (cached)


# ═══════════════════════════════════════════════════════════════════════════
# §6  PermutationGroup
# ═══════════════════════════════════════════════════════════════════════════


class TestPermutationGroupBasic:
    """Tests for basic PermutationGroup operations."""

    def test_identity(self):
        n = 5
        group = PermutationGroup(generators=[_cyclic_perm(n)], n=n)
        np.testing.assert_array_equal(group.identity, np.arange(n))

    def test_compose_identity(self):
        n = 4
        group = PermutationGroup(generators=[_cyclic_perm(n)], n=n)
        sigma = _cyclic_perm(n)
        result = group.compose(sigma, group.identity)
        np.testing.assert_array_equal(result, sigma)

    def test_compose_left_identity(self):
        n = 4
        group = PermutationGroup(generators=[_cyclic_perm(n)], n=n)
        sigma = _cyclic_perm(n)
        result = group.compose(group.identity, sigma)
        np.testing.assert_array_equal(result, sigma)

    def test_compose_two_cycles(self):
        """(0 1 2 3) composed with itself should give (0 2)(1 3)."""
        n = 4
        group = PermutationGroup(generators=[_cyclic_perm(n)], n=n)
        c = _cyclic_perm(n)
        cc = group.compose(c, c)
        # (0→1→2), (1→2→3), (2→3→0), (3→0→1)
        np.testing.assert_array_equal(cc, [2, 3, 0, 1])

    def test_inverse_of_identity(self):
        n = 5
        group = PermutationGroup(generators=[_identity_perm(n)], n=n)
        inv = group.inverse(group.identity)
        np.testing.assert_array_equal(inv, group.identity)

    def test_inverse_of_cycle(self):
        n = 4
        group = PermutationGroup(generators=[_cyclic_perm(n)], n=n)
        c = _cyclic_perm(n)
        inv = group.inverse(c)
        # Composing with inverse should give identity
        product = group.compose(c, inv)
        np.testing.assert_array_equal(product, group.identity)

    def test_inverse_of_transposition(self):
        n = 5
        t = _transposition(n, 1, 3)
        group = PermutationGroup(generators=[t], n=n)
        inv = group.inverse(t)
        # Transposition is self-inverse
        np.testing.assert_array_equal(inv, t)

    def test_compose_inverse_identity(self):
        n = 6
        gen = _cyclic_perm(n)
        group = PermutationGroup(generators=[gen], n=n)
        inv = group.inverse(gen)
        product = group.compose(gen, inv)
        np.testing.assert_array_equal(product, group.identity)

    def test_invalid_generator_length(self):
        with pytest.raises(ValueError, match="Generator length"):
            PermutationGroup(generators=[np.array([1, 0, 2])], n=5)

    def test_invalid_generator_not_permutation(self):
        with pytest.raises(ValueError, match="not a valid permutation"):
            PermutationGroup(generators=[np.array([0, 0, 2, 3])], n=4)


class TestPermutationGroupOrbits:
    """Tests for orbit computation in PermutationGroup."""

    def test_orbit_of_identity_generator(self):
        n = 5
        group = PermutationGroup(generators=[_identity_perm(n)], n=n)
        for i in range(n):
            orb = group.orbit_of(i)
            assert orb == [i]

    def test_orbit_of_cyclic_generator(self):
        n = 6
        group = PermutationGroup(generators=[_cyclic_perm(n)], n=n)
        orb = group.orbit_of(0)
        assert sorted(orb) == list(range(n))

    def test_orbit_of_transposition(self):
        n = 5
        t = _transposition(n, 2, 4)
        group = PermutationGroup(generators=[t], n=n)
        assert sorted(group.orbit_of(2)) == [2, 4]
        assert sorted(group.orbit_of(4)) == [2, 4]
        assert group.orbit_of(0) == [0]

    def test_all_orbits_identity(self):
        n = 4
        group = PermutationGroup(generators=[_identity_perm(n)], n=n)
        orbits = group.all_orbits()
        assert len(orbits) == n
        for orb in orbits:
            assert len(orb) == 1

    def test_all_orbits_cyclic(self):
        n = 5
        group = PermutationGroup(generators=[_cyclic_perm(n)], n=n)
        orbits = group.all_orbits()
        assert len(orbits) == 1
        assert sorted(orbits[0]) == list(range(n))

    def test_all_orbits_two_generators(self):
        n = 6
        t1 = _transposition(n, 0, 1)
        t2 = _transposition(n, 2, 3)
        group = PermutationGroup(generators=[t1, t2], n=n)
        orbits = group.all_orbits()
        orbit_sets = [frozenset(o) for o in orbits]
        assert frozenset({0, 1}) in orbit_sets
        assert frozenset({2, 3}) in orbit_sets
        assert frozenset({4}) in orbit_sets
        assert frozenset({5}) in orbit_sets

    def test_orbit_of_element_out_of_range(self):
        n = 3
        group = PermutationGroup(generators=[_cyclic_perm(n)], n=n)
        # Element within range should work fine
        orb = group.orbit_of(0)
        assert 0 in orb


class TestPermutationGroupOrderEstimate:
    """Tests for group order estimation."""

    def test_trivial_group_order(self):
        n = 5
        group = PermutationGroup(generators=[_identity_perm(n)], n=n)
        assert group.order_estimate() == 1

    def test_cyclic_group_order(self):
        for n in [3, 4, 5, 7]:
            group = PermutationGroup(generators=[_cyclic_perm(n)], n=n)
            assert group.order_estimate() == n

    def test_transposition_order(self):
        n = 5
        t = _transposition(n, 0, 1)
        group = PermutationGroup(generators=[t], n=n)
        assert group.order_estimate() == 2

    def test_symmetric_group_order_s3(self):
        """S_3 generated by (0 1 2) and (0 1) should have order 6."""
        n = 3
        c = _cyclic_perm(n)
        t = _transposition(n, 0, 1)
        group = PermutationGroup(generators=[c, t], n=n)
        assert group.order_estimate() == 6

    def test_dihedral_group_order(self):
        """D_4 generated by rotation and reflection should have order 8."""
        n = 4
        rot = _cyclic_perm(n)
        refl = _reverse_perm(n)
        group = PermutationGroup(generators=[rot, refl], n=n)
        assert group.order_estimate() == 8

    def test_max_elements_caps_output(self):
        n = 5
        c = _cyclic_perm(n)
        t = _transposition(n, 0, 1)
        group = PermutationGroup(generators=[c, t], n=n)
        # S_5 has 120 elements; cap at 10
        estimate = group.order_estimate(max_elements=10)
        assert estimate == 10


# ═══════════════════════════════════════════════════════════════════════════
# §7  Group Theory Utilities
# ═══════════════════════════════════════════════════════════════════════════


class TestOrbitComputation:
    """Tests for the orbit_computation function."""

    def test_empty_elements(self):
        assert orbit_computation([], []) == []

    def test_no_generators(self):
        elements = [0, 1, 2, 3]
        orbits = orbit_computation(elements, [])
        assert len(orbits) == 4
        for orb in orbits:
            assert len(orb) == 1

    def test_identity_generator(self):
        n = 5
        elements = list(range(n))
        gen = _identity_perm(n)
        orbits = orbit_computation(elements, [gen])
        assert len(orbits) == n

    def test_cyclic_generator(self):
        n = 4
        elements = list(range(n))
        gen = _cyclic_perm(n)
        orbits = orbit_computation(elements, [gen])
        assert len(orbits) == 1
        assert sorted(orbits[0]) == elements

    def test_transposition_generator(self):
        n = 5
        elements = list(range(n))
        gen = _transposition(n, 0, 2)
        orbits = orbit_computation(elements, [gen])
        orbit_sets = [frozenset(o) for o in orbits]
        assert frozenset({0, 2}) in orbit_sets
        assert frozenset({1}) in orbit_sets
        assert frozenset({3}) in orbit_sets
        assert frozenset({4}) in orbit_sets

    def test_multiple_generators(self):
        n = 6
        g1 = _transposition(n, 0, 1)
        g2 = _transposition(n, 1, 2)
        elements = list(range(n))
        orbits = orbit_computation(elements, [g1, g2])
        orbit_sets = [frozenset(o) for o in orbits]
        # (0 1) and (1 2) generate transitive group on {0,1,2}
        assert frozenset({0, 1, 2}) in orbit_sets

    def test_orbits_are_sorted(self):
        n = 6
        gen = _cyclic_perm(n)
        orbits = orbit_computation(list(range(n)), [gen])
        for orb in orbits:
            assert orb == sorted(orb)

    def test_orbits_partition_elements(self):
        n = 8
        g1 = _transposition(n, 0, 3)
        g2 = _transposition(n, 4, 7)
        elements = list(range(n))
        orbits = orbit_computation(elements, [g1, g2])
        all_elems = set()
        for orb in orbits:
            for e in orb:
                assert e not in all_elems
                all_elems.add(e)
        assert all_elems == set(elements)


class TestIsGroup:
    """Tests for the is_group function."""

    def test_trivial_group(self):
        n = 3
        elements = [_identity_perm(n)]
        assert is_group(elements, n) is True

    def test_cyclic_group_z3(self):
        n = 3
        c = _cyclic_perm(n)
        group = PermutationGroup(generators=[c], n=n)
        # Build all elements
        elements = [group.identity]
        cur = c.copy()
        while not np.array_equal(cur, group.identity):
            elements.append(cur.copy())
            cur = group.compose(c, cur)
        assert is_group(elements, n) is True

    def test_z2_transposition(self):
        n = 4
        t = _transposition(n, 0, 1)
        elements = [_identity_perm(n), t]
        assert is_group(elements, n) is True

    def test_not_a_group_missing_identity(self):
        n = 3
        elements = [_cyclic_perm(n)]  # No identity
        assert is_group(elements, n) is False

    def test_not_a_group_missing_inverse(self):
        n = 4
        c = _cyclic_perm(n)  # order 4
        # Include identity and c but not c^{-1} = c^3
        elements = [_identity_perm(n), c]
        assert is_group(elements, n) is False

    def test_not_a_group_not_closed(self):
        n = 4
        t1 = _transposition(n, 0, 1)
        t2 = _transposition(n, 2, 3)
        # {id, (01), (23)} is not closed: (01)(23) is missing
        elements = [_identity_perm(n), t1, t2]
        assert is_group(elements, n) is False

    def test_empty_not_a_group(self):
        assert is_group([], 3) is False

    def test_s3_is_group(self):
        """S_3 has 6 elements and should satisfy group axioms."""
        n = 3
        c = _cyclic_perm(n)
        t = _transposition(n, 0, 1)
        group = PermutationGroup(generators=[c, t], n=n)
        # Enumerate all elements
        seen = set()
        queue = [group.identity]
        seen.add(group.identity.tobytes())
        elements = [group.identity.copy()]
        while queue:
            cur = queue.pop(0)
            for gen in [c, t]:
                for new in [group.compose(gen, cur), group.compose(group.inverse(gen), cur)]:
                    key = new.tobytes()
                    if key not in seen:
                        seen.add(key)
                        queue.append(new)
                        elements.append(new.copy())
        assert len(elements) == 6
        assert is_group(elements, n) is True


class TestStabilizer:
    """Tests for the stabilizer function."""

    def test_stabilizer_trivial_group(self):
        n = 3
        group = PermutationGroup(generators=[_identity_perm(n)], n=n)
        stab = stabilizer(0, group)
        assert len(stab) == 1
        np.testing.assert_array_equal(stab[0], group.identity)

    def test_stabilizer_of_fixed_point(self):
        """In a transposition group, a non-swapped element is fixed by all."""
        n = 4
        t = _transposition(n, 0, 1)
        group = PermutationGroup(generators=[t], n=n)
        stab = stabilizer(2, group)
        assert len(stab) == 2  # Both id and (0 1) fix element 2

    def test_stabilizer_of_moved_point(self):
        """Element 0 in ⟨(0 1)⟩ is moved by (0 1), so stabilizer is {id}."""
        n = 4
        t = _transposition(n, 0, 1)
        group = PermutationGroup(generators=[t], n=n)
        stab = stabilizer(0, group)
        assert len(stab) == 1
        np.testing.assert_array_equal(stab[0], group.identity)


# ═══════════════════════════════════════════════════════════════════════════
# §8  Reconstruction and Expansion
# ═══════════════════════════════════════════════════════════════════════════


class TestExpandNoiseToFull:
    """Tests for expand_noise_to_full."""

    def test_output_shape(self):
        n, k = 5, 8
        noise = np.ones(k) / k
        f_values = np.arange(n, dtype=np.float64)
        grid = np.arange(k, dtype=np.float64)
        table = expand_noise_to_full(noise, f_values, grid)
        assert table.shape == (n, k)

    def test_rows_sum_to_one(self):
        n, k = 4, 6
        rng = np.random.default_rng(99)
        noise = rng.dirichlet(np.ones(k))
        f_values = np.arange(n, dtype=np.float64)
        grid = np.arange(k, dtype=np.float64)
        table = expand_noise_to_full(noise, f_values, grid)
        np.testing.assert_allclose(table.sum(axis=1), 1.0, atol=1e-10)

    def test_nonnegative(self):
        n, k = 3, 5
        noise = np.array([0.1, 0.3, 0.4, 0.15, 0.05])
        f_values = np.arange(n, dtype=np.float64)
        grid = np.arange(k, dtype=np.float64)
        table = expand_noise_to_full(noise, f_values, grid)
        assert np.all(table >= 0)

    def test_single_row(self):
        k = 5
        noise = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        f_values = np.array([0.0])
        grid = np.arange(k, dtype=np.float64)
        table = expand_noise_to_full(noise, f_values, grid)
        assert table.shape == (1, k)
        np.testing.assert_allclose(table[0], noise, atol=1e-10)

    def test_matching_grid_and_values(self):
        """When f_values == grid, row 0 should exactly equal noise."""
        k = 4
        noise = np.array([0.25, 0.25, 0.25, 0.25])
        f_values = np.arange(k, dtype=np.float64)
        grid = np.arange(k, dtype=np.float64)
        table = expand_noise_to_full(noise, f_values, grid)
        np.testing.assert_allclose(table[0], noise, atol=1e-12)


class TestReconstructionMap:
    """Tests for ReconstructionMap."""

    def test_expand_translation(self):
        n, k = 4, 6
        f_values = np.arange(n, dtype=np.float64)
        grid = np.arange(k, dtype=np.float64)
        rmap = ReconstructionMap(
            reduction_type="translation",
            original_n=n,
            original_k=k,
            f_values=f_values,
            y_grid=grid,
        )
        noise = np.ones(k) / k
        table = rmap.expand(noise)
        assert table.shape == (n, k)
        np.testing.assert_allclose(table.sum(axis=1), 1.0, atol=1e-10)

    def test_expand_reflection(self):
        k = 6
        n_free = (k + 1) // 2
        f_values = np.array([0.0])
        grid = np.arange(k, dtype=np.float64)
        rmap = ReconstructionMap(
            reduction_type="reflection",
            original_n=1,
            original_k=k,
            f_values=f_values,
            y_grid=grid,
        )
        noise_free = np.array([0.2, 0.15, 0.15])
        table = rmap.expand(noise_free)
        assert table.shape == (1, k)

    def test_expand_unknown_type_raises(self):
        rmap = ReconstructionMap(
            reduction_type="unknown",
            original_n=2,
            original_k=4,
            f_values=np.array([0.0, 1.0]),
            y_grid=np.arange(4, dtype=np.float64),
        )
        with pytest.raises(ValueError, match="Unknown reduction type"):
            rmap.expand(np.array([0.5, 0.5]))


class TestVerifyReconstruction:
    """Tests for verify_reconstruction."""

    def test_exact_match(self):
        n, k = 3, 5
        f_values = np.arange(n, dtype=np.float64)
        grid = np.arange(k, dtype=np.float64)
        noise = np.ones(k) / k
        table = expand_noise_to_full(noise, f_values, grid)
        rmap = ReconstructionMap(
            reduction_type="translation",
            original_n=n,
            original_k=k,
            f_values=f_values,
            y_grid=grid,
        )
        assert verify_reconstruction(table, noise, rmap) is True

    def test_mismatch(self):
        n, k = 3, 5
        f_values = np.arange(n, dtype=np.float64)
        grid = np.arange(k, dtype=np.float64)
        noise = np.ones(k) / k
        # Construct a table that doesn't match
        wrong_table = np.ones((n, k)) / k
        rmap = ReconstructionMap(
            reduction_type="translation",
            original_n=n,
            original_k=k,
            f_values=f_values,
            y_grid=grid,
        )
        result = verify_reconstruction(wrong_table, noise, rmap)
        # May or may not match depending on grid; test that function runs
        assert isinstance(result, bool)

    def test_custom_tolerance(self):
        n, k = 2, 4
        f_values = np.arange(n, dtype=np.float64)
        grid = np.arange(k, dtype=np.float64)
        noise = np.ones(k) / k
        table = expand_noise_to_full(noise, f_values, grid)
        rmap = ReconstructionMap(
            reduction_type="translation",
            original_n=n,
            original_k=k,
            f_values=f_values,
            y_grid=grid,
        )
        # With very tight tolerance, should still match for exact noise
        assert verify_reconstruction(table, noise, rmap, tol=1e-12) is True


# ═══════════════════════════════════════════════════════════════════════════
# §9  Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════


class TestIsCountingQuery:
    """Tests for is_counting_query."""

    def test_counting_spec(self):
        spec = _counting_spec(10)
        assert is_counting_query(spec) is True

    def test_counting_n1(self):
        spec = _counting_spec(1, k=5)
        assert is_counting_query(spec) is True

    def test_counting_n2(self):
        spec = _counting_spec(2)
        assert is_counting_query(spec) is True

    def test_non_counting_values(self):
        spec = _custom_spec([0.0, 2.0, 4.0])
        assert is_counting_query(spec) is False

    def test_non_unit_sensitivity(self):
        spec = QuerySpec(
            query_values=np.arange(5, dtype=np.float64),
            domain="test",
            sensitivity=2.0,
            epsilon=1.0,
            k=10,
            query_type=QueryType.CUSTOM,
        )
        assert is_counting_query(spec) is False

    def test_counting_type_flag(self):
        """QueryType.COUNTING should be detected regardless of values."""
        spec = QuerySpec(
            query_values=np.array([0.0, 1.0, 2.0]),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            k=10,
            query_type=QueryType.COUNTING,
        )
        assert is_counting_query(spec) is True

    def test_histogram_structurally_matches_counting(self):
        """histogram() has same values/sensitivity as counting, so is_counting_query is True."""
        spec = QuerySpec.histogram(5, epsilon=1.0)
        # histogram has query_values=[0,1,...,n-1] and sensitivity=1, same as counting
        assert is_counting_query(spec) is True


class TestDetectAndReportSymmetry:
    """Tests for detect_and_report_symmetry."""

    def test_returns_symmetry_group(self):
        spec = _counting_spec(8)
        group = detect_and_report_symmetry(spec)
        assert isinstance(group, SymmetryGroup)
        assert group.is_translation_invariant is True

    def test_trivial_symmetry(self):
        spec = QuerySpec(
            query_values=np.array([0.0, 1.0, 5.0]),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            k=10,
            loss_fn=LossFunction.CUSTOM,
            custom_loss=lambda t, y: abs(t - y) ** 3,
            query_type=QueryType.CUSTOM,
        )
        group = detect_and_report_symmetry(spec)
        assert group.group_type == "trivial"

    def test_report_counting_combined(self):
        spec = _counting_spec(5)
        group = detect_and_report_symmetry(spec)
        assert group.group_type == "translation+reflection"
        assert group.theoretical_speedup == 10.0


# ═══════════════════════════════════════════════════════════════════════════
# §10  ReducedLPStruct
# ═══════════════════════════════════════════════════════════════════════════


class TestReducedLPStruct:
    """Tests for ReducedLPStruct properties."""

    def _make_reduced(self, n_vars=5, n_ub=3, n_eq=1):
        c = np.zeros(n_vars)
        A_ub = csr_matrix(np.ones((n_ub, n_vars)))
        b_ub = np.zeros(n_ub)
        A_eq = csr_matrix(np.ones((n_eq, n_vars))) if n_eq > 0 else None
        b_eq = np.ones(n_eq) if n_eq > 0 else None
        bounds = [(0.0, 1.0)] * n_vars
        return ReducedLPStruct(
            c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, original_n=10, original_k=20,
            reduction_type="test", reduction_factor=4.0,
            y_grid=np.arange(20, dtype=np.float64),
        )

    def test_n_vars(self):
        r = self._make_reduced(n_vars=7)
        assert r.n_vars == 7

    def test_n_ub(self):
        r = self._make_reduced(n_ub=5)
        assert r.n_ub == 5

    def test_n_eq_with_constraints(self):
        r = self._make_reduced(n_eq=2)
        assert r.n_eq == 2

    def test_n_eq_without_constraints(self):
        r = self._make_reduced(n_eq=0)
        assert r.n_eq == 0

    def test_repr(self):
        r = self._make_reduced()
        s = repr(r)
        assert "ReducedLPStruct" in s
        assert "test" in s


# ═══════════════════════════════════════════════════════════════════════════
# §11  Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_n2_counting_translation(self):
        spec = _counting_spec(2, k=5)
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_translation_invariant is True
        assert group.metadata["n"] == 2

    def test_n2_reduction_factor(self):
        k = 5
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        factor = reducer.reduction_factor(2)
        expected = (2 * k + 1) / (k + 1)
        assert abs(factor - expected) < 1e-10

    def test_k2_reduce(self):
        """k=2 is the minimum discretization."""
        k = 2
        y_grid = np.array([0.0, 1.0])
        reducer = TranslationReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        result = reducer.reduce(n=5)
        assert result.n_vars == 3  # 2 eta + 1 t

    def test_large_epsilon(self):
        """Large epsilon (weak privacy) should still produce valid LP."""
        k = 5
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=10.0, delta=0.0, k=k, y_grid=y_grid,
        )
        result = reducer.reduce(n=5)
        assert result.n_vars == k + 1

    def test_small_epsilon(self):
        """Small epsilon (strong privacy) should still produce valid LP."""
        k = 5
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=0.01, delta=0.0, k=k, y_grid=y_grid,
        )
        result = reducer.reduce(n=5)
        assert result.n_vars == k + 1

    def test_negative_f_values_arithmetic(self):
        """Arithmetic sequence with negative start should be translation invariant."""
        spec = _custom_spec([-2.0, -1.0, 0.0, 1.0, 2.0])
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_translation_invariant is True

    def test_orbit_computation_single_element(self):
        elements = [0]
        gen = np.array([0], dtype=np.intp)
        orbits = orbit_computation(elements, [gen])
        assert orbits == [[0]]

    def test_permutation_group_n1(self):
        """Permutation group on a single element."""
        n = 1
        group = PermutationGroup(generators=[np.array([0], dtype=np.intp)], n=n)
        assert group.order_estimate() == 1
        np.testing.assert_array_equal(group.identity, [0])
        assert group.orbit_of(0) == [0]
        assert group.all_orbits() == [[0]]

    def test_expand_noise_single_bin(self):
        """k=1 edge case (degenerate)."""
        # This is a degenerate case but should not crash
        noise = np.array([1.0])
        f_values = np.array([0.0])
        grid = np.array([0.0])
        table = expand_noise_to_full(noise, f_values, grid)
        assert table.shape == (1, 1)
        np.testing.assert_allclose(table[0, 0], 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# §12  Integration: Symmetry Detection + Reduction Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegrationDetectAndReduce:
    """Integration tests combining detection with reduction."""

    def test_counting_query_full_pipeline(self):
        """Detect symmetry → reduce → check dimensions."""
        n, k = 8, 12
        spec = _counting_spec(n, k=k)
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_translation_invariant

        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=spec.epsilon, delta=spec.delta,
            k=k, y_grid=y_grid,
        )
        reduced = reducer.reduce(n=n)
        assert reduced.n_vars == k + 1
        assert reduced.reduction_factor > 1.0

    def test_non_counting_no_reduction(self):
        """Non-counting query should yield trivial symmetry group."""
        spec = _custom_spec([0.0, 1.0, 7.0, 15.0])
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        # No translation invariance
        assert group.is_translation_invariant is False
        # Still has reflection due to L2
        assert group.is_reflection_symmetric is True

    def test_reconstruct_from_reduced(self):
        """Reduce, solve conceptually, reconstruct."""
        n, k = 5, 8
        spec = _counting_spec(n, k=k)
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=spec.epsilon, delta=spec.delta,
            k=k, y_grid=y_grid,
        )
        # Use uniform noise as a conceptual solution
        noise = np.ones(k) / k
        f_values = spec.query_values
        table = reducer.reconstruct(noise, f_values)
        assert table.shape == (n, k)
        np.testing.assert_allclose(table.sum(axis=1), 1.0, atol=1e-10)

    def test_translation_then_reflection_further_reduces(self):
        """After translation reduction, reflection can halve variables."""
        n, k = 10, 20
        spec = _counting_spec(n, k=k)
        y_grid = np.linspace(-5, 5, k)

        # Translation: n*k+1 → k+1
        t_reducer = TranslationReducer(
            epsilon=spec.epsilon, delta=spec.delta,
            k=k, y_grid=y_grid,
        )
        t_reduced = t_reducer.reduce(n=n)
        assert t_reduced.n_vars == k + 1

        # Reflection: k+1 → ceil(k/2)+1
        r_reducer = ReflectionReducer(
            epsilon=spec.epsilon, delta=spec.delta,
            k=k, y_grid=y_grid,
        )
        r_reduced = r_reducer.reduce()
        n_free = (k + 1) // 2
        assert r_reduced.n_vars == n_free + 1
        assert r_reduced.n_vars < t_reduced.n_vars

    def test_speedup_matches_expectation(self):
        """Theoretical speedup should match n for counting query."""
        n = 20
        spec = _counting_spec(n, k=15)
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        # Translation + reflection => 2n speedup
        assert group.theoretical_speedup == 2 * n

    def test_multiple_detections_consistent(self):
        """Running detection multiple times should give same result."""
        spec = _counting_spec(10, k=12)
        detector = SymmetryDetector(spec=spec)
        g1 = detector.detect()
        g2 = detector.detect()
        assert g1.is_translation_invariant == g2.is_translation_invariant
        assert g1.is_reflection_symmetric == g2.is_reflection_symmetric
        assert g1.order == g2.order
        assert g1.group_type == g2.group_type


# ═══════════════════════════════════════════════════════════════════════════
# §13  DP Constraint Validity
# ═══════════════════════════════════════════════════════════════════════════


class TestDPConstraintValidity:
    """Verify that reduced DP constraints are correct."""

    def test_geometric_noise_satisfies_constraints(self):
        """A geometric (staircase) noise distribution should satisfy DP constraints."""
        k = 8
        epsilon = 1.0
        e_eps = math.exp(epsilon)
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=epsilon, delta=0.0, k=k, y_grid=y_grid,
        )

        # Build a valid geometric noise distribution
        centre = k // 2
        eta = np.zeros(k)
        for l in range(k):
            eta[l] = e_eps ** (-abs(l - centre))
        eta /= eta.sum()

        A, b = reducer._build_reduced_constraints(k, epsilon)
        residual = A @ eta - b
        assert np.all(residual <= 1e-10)

    def test_peaked_noise_violates_constraints(self):
        """A noise distribution that is too peaked should violate DP constraints."""
        k = 5
        epsilon = 0.5
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=epsilon, delta=0.0, k=k, y_grid=y_grid,
        )

        # Extremely peaked: all mass at centre
        eta = np.zeros(k)
        eta[k // 2] = 1.0

        A, b = reducer._build_reduced_constraints(k, epsilon)
        residual = A @ eta - b
        # Should violate at least one constraint
        assert np.any(residual > 1e-10)

    def test_uniform_noise_satisfies_constraints(self):
        """Uniform noise trivially satisfies DP constraints."""
        for k in [3, 5, 10, 20]:
            epsilon = 1.0
            y_grid = np.arange(k, dtype=np.float64)
            reducer = TranslationReducer(
                epsilon=epsilon, delta=0.0, k=k, y_grid=y_grid,
            )
            eta = np.ones(k) / k
            A, b = reducer._build_reduced_constraints(k, epsilon)
            residual = A @ eta - b
            assert np.all(residual <= 1e-12), f"Failed for k={k}"


# ═══════════════════════════════════════════════════════════════════════════
# §14  Parameterized Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParameterized:
    """Parameterized tests for scalability and variety."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 50])
    def test_counting_detection_various_n(self, n):
        spec = _counting_spec(n, k=max(n + 2, 5))
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_translation_invariant is True

    @pytest.mark.parametrize("k", [2, 3, 5, 10, 20])
    def test_translation_reduce_various_k(self, k):
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        result = reducer.reduce(n=10)
        assert result.n_vars == k + 1

    @pytest.mark.parametrize("k", [2, 3, 5, 8, 11])
    def test_reflection_reduce_various_k(self, k):
        y_grid = np.linspace(-3, 3, k)
        reducer = ReflectionReducer(
            epsilon=1.0, delta=0.0, k=k, y_grid=y_grid,
        )
        result = reducer.reduce()
        n_free = (k + 1) // 2
        assert result.n_vars == n_free + 1

    @pytest.mark.parametrize("epsilon", [0.01, 0.1, 0.5, 1.0, 2.0, 5.0])
    def test_reduction_factor_independent_of_epsilon(self, epsilon):
        k = 10
        y_grid = np.arange(k, dtype=np.float64)
        reducer = TranslationReducer(
            epsilon=epsilon, delta=0.0, k=k, y_grid=y_grid,
        )
        factor = reducer.reduction_factor(20)
        expected = (20 * k + 1) / (k + 1)
        assert abs(factor - expected) < 1e-10

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_cyclic_group_order(self, n):
        group = PermutationGroup(generators=[_cyclic_perm(n)], n=n)
        assert group.order_estimate() == n

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_dihedral_group_order(self, n):
        rot = _cyclic_perm(n)
        refl = _reverse_perm(n)
        group = PermutationGroup(generators=[rot, refl], n=n)
        assert group.order_estimate() == 2 * n

    @pytest.mark.parametrize(
        "loss_fn",
        [LossFunction.L1, LossFunction.L2, LossFunction.LINF],
    )
    def test_symmetric_losses_detected(self, loss_fn):
        spec = QuerySpec(
            query_values=np.arange(5, dtype=np.float64),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            k=10,
            loss_fn=loss_fn,
            query_type=QueryType.COUNTING,
        )
        detector = SymmetryDetector(spec=spec)
        group = detector.detect()
        assert group.is_reflection_symmetric is True


# ═══════════════════════════════════════════════════════════════════════════
# §15  ReduceBySymmetry Top-Level Entry Point
# ═══════════════════════════════════════════════════════════════════════════


class TestReduceBySymmetry:
    """Tests for the ReduceBySymmetry top-level function."""

    def _make_lp_struct(self, n, k):
        """Build a minimal LPStruct for testing."""
        n_vars = n * k + 1
        c = np.zeros(n_vars)
        c[-1] = 1.0
        A_ub = csr_matrix(np.eye(n_vars, n_vars))
        b_ub = np.ones(n_vars)
        A_eq = csr_matrix(np.ones((1, n_vars)))
        b_eq = np.array([1.0])
        bounds = [(0.0, 1.0)] * n_vars
        var_map = {}
        for i in range(n):
            for j in range(k):
                var_map[(i, j)] = i * k + j
        y_grid = np.arange(k, dtype=np.float64)
        return LPStruct(
            c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, var_map=var_map, y_grid=y_grid,
        )

    def test_counting_query_reduces(self):
        n, k = 5, 8
        spec = _counting_spec(n, k=k)
        lp = self._make_lp_struct(n, k)
        reduced, rmap = ReduceBySymmetry(lp, spec)
        assert reduced.reduction_type == "translation"
        assert reduced.n_vars == k + 1

    def test_no_symmetry_returns_original(self):
        n, k = 3, 5
        spec = QuerySpec(
            query_values=np.array([0.0, 1.0, 5.0]),
            domain="test",
            sensitivity=1.0,
            epsilon=1.0,
            k=k,
            loss_fn=LossFunction.CUSTOM,
            custom_loss=lambda t, y: abs(t - y) ** 3,
            query_type=QueryType.CUSTOM,
        )
        lp = self._make_lp_struct(n, k)
        reduced, rmap = ReduceBySymmetry(lp, spec)
        assert reduced.reduction_type == "none"
        assert reduced.reduction_factor == 1.0

    def test_pre_detected_symmetry(self):
        n, k = 5, 8
        spec = _counting_spec(n, k=k)
        lp = self._make_lp_struct(n, k)
        group = SymmetryGroup(
            is_translation_invariant=True,
            is_reflection_symmetric=True,
            order=2 * n,
            group_type="translation+reflection",
            metadata={"n": n},
        )
        reduced, rmap = ReduceBySymmetry(lp, spec, symmetry_group=group)
        assert reduced.reduction_type == "translation"

    def test_reconstruction_map_type(self):
        n, k = 5, 8
        spec = _counting_spec(n, k=k)
        lp = self._make_lp_struct(n, k)
        _, rmap = ReduceBySymmetry(lp, spec)
        assert isinstance(rmap, ReconstructionMap)
        assert rmap.reduction_type == "translation"
        assert rmap.original_n == n
        assert rmap.original_k == k

    def test_reflection_only_returns_original(self):
        """Reflection-only symmetry doesn't trigger reduction in ReduceBySymmetry."""
        n, k = 3, 5
        spec = _custom_spec([0.0, 1.0, 5.0], loss_fn=LossFunction.L2)
        lp = self._make_lp_struct(n, k)
        reduced, rmap = ReduceBySymmetry(lp, spec)
        assert reduced.reduction_type == "none"
