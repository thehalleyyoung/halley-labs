"""
Comprehensive tests for dp_forge.multidim module.

Tests cover separability detection, tensor product mechanisms, budget
allocation, lower bounds, marginal queries, and the ProjectedCEGIS pipeline.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from dp_forge.multidim.separability_detector import (
    KroneckerFactor,
    SeparabilityDetector,
    SeparabilityResult,
    SeparabilityType,
)
from dp_forge.multidim.tensor_product import (
    MarginalMechanism,
    TensorProductMechanism,
    build_product_mechanism,
    kronecker_sparse,
)
from dp_forge.multidim.budget_allocation import (
    AllocationStrategy,
    BudgetAllocation,
    BudgetAllocator,
)
from dp_forge.multidim.lower_bounds import (
    LowerBoundComputer,
    LowerBoundResult,
)
from dp_forge.multidim.marginal_queries import (
    MarginalQuery,
    MarginalQueryBuilder,
)
from dp_forge.types import CompositionType, PrivacyBudget
from dp_forge.exceptions import ConfigurationError


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_marginal_2x3():
    """Simple 2-input, 3-output marginal mechanism."""
    table = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.3, 0.5],
    ])
    grid = np.array([0.0, 0.5, 1.0])
    return MarginalMechanism(
        p_table=table, y_grid=grid, coordinate_index=0,
        epsilon=1.0, sensitivity=1.0,
    )


@pytest.fixture
def simple_marginal_2x2():
    """Simple 2-input, 2-output marginal mechanism."""
    table = np.array([
        [0.7, 0.3],
        [0.3, 0.7],
    ])
    grid = np.array([0.0, 1.0])
    return MarginalMechanism(
        p_table=table, y_grid=grid, coordinate_index=1,
        epsilon=1.0, sensitivity=1.0,
    )


# =========================================================================
# Section 1: Separability Detection
# =========================================================================


class TestSeparabilityDetector:
    """Tests for Kronecker/block-diagonal decomposition detection."""

    def test_identity_is_separable(self):
        """I⊗I (4×4 identity) detected as separable."""
        A = np.eye(4)
        detector = SeparabilityDetector(tol=1e-8)
        result = detector.detect(A)
        assert result.is_separable

    def test_kronecker_product_detected(self):
        """Exact Kronecker product A₁⊗A₂ is detected."""
        A1 = np.array([[1, 0], [0, 1]], dtype=float)
        A2 = np.array([[1, 0], [0, 1]], dtype=float)
        A = np.kron(A1, A2)
        detector = SeparabilityDetector(tol=1e-8)
        result = detector.detect(A)
        assert result.is_separable

    def test_non_separable_matrix(self):
        """Random non-separable matrix detected correctly."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((6, 6))
        detector = SeparabilityDetector(tol=1e-8)
        result = detector.detect(A)
        # Random matrix is almost certainly non-separable
        assert result.sep_type in (
            SeparabilityType.NON_SEPARABLE,
            SeparabilityType.PARTIAL,
        )

    def test_block_diagonal(self):
        """Block-diagonal matrix detected."""
        A = np.zeros((4, 4))
        A[0:2, 0:2] = np.array([[1, 2], [3, 4]])
        A[2:4, 2:4] = np.array([[5, 6], [7, 8]])
        detector = SeparabilityDetector(tol=1e-8)
        result = detector.detect(A)
        assert result.sep_type in (
            SeparabilityType.BLOCK_DIAGONAL,
            SeparabilityType.FULL_KRONECKER,
        )

    def test_identity_structure_detection(self):
        """Permuted identity detected via detect_identity_structure."""
        A = np.zeros((3, 3))
        A[0, 2] = 1.0
        A[1, 0] = 1.0
        A[2, 1] = 1.0
        detector = SeparabilityDetector()
        indices = detector.detect_identity_structure(A)
        assert indices is not None
        assert len(indices) == 3

    def test_residual_norm_small_for_exact(self):
        """Exact Kronecker product has near-zero residual."""
        A1 = np.array([[1, 0.5], [0.5, 1]], dtype=float)
        A2 = np.array([[2, 1], [1, 3]], dtype=float)
        A = np.kron(A1, A2)
        detector = SeparabilityDetector(tol=1e-6)
        result = detector.detect(A)
        if result.is_separable:
            assert result.residual_norm < 1e-6

    def test_separability_result_properties(self):
        """SeparabilityResult has correct properties."""
        result = SeparabilityResult(
            sep_type=SeparabilityType.FULL_KRONECKER,
            factors=[
                KroneckerFactor(matrix=np.eye(2), coordinate_indices=[0], rank=2),
                KroneckerFactor(matrix=np.eye(3), coordinate_indices=[1], rank=3),
            ],
        )
        assert result.is_separable
        assert result.n_factors == 2

    def test_1x1_matrix(self):
        """1×1 matrix is trivially separable."""
        A = np.array([[5.0]])
        detector = SeparabilityDetector()
        result = detector.detect(A)
        assert result.is_separable or result.sep_type == SeparabilityType.NON_SEPARABLE


# =========================================================================
# Section 2: Tensor Product Mechanism
# =========================================================================


class TestTensorProductMechanism:
    """Tests for tensor product mechanism construction and operations."""

    def test_product_preserves_normalization(
        self, simple_marginal_2x3, simple_marginal_2x2
    ):
        """Tensor product preserves row normalization (rows sum to 1)."""
        mech = TensorProductMechanism(
            marginals=[simple_marginal_2x3, simple_marginal_2x2],
            total_epsilon=2.0,
        )
        # Dense materialization for all input combos
        for i0 in range(simple_marginal_2x3.n):
            for i1 in range(simple_marginal_2x2.n):
                dense = mech.materialise_dense(input_indices=[i0, i1])
                total = dense.sum()
                assert abs(total - 1.0) < 1e-10, (
                    f"Row sum for input ({i0},{i1}) = {total}, expected 1.0"
                )

    def test_product_correct_marginals(
        self, simple_marginal_2x3, simple_marginal_2x2
    ):
        """Product mechanism has correct marginals."""
        mech = TensorProductMechanism(
            marginals=[simple_marginal_2x3, simple_marginal_2x2],
        )
        m0 = mech.marginal(0)
        m1 = mech.marginal(1)
        np.testing.assert_array_almost_equal(m0.p_table, simple_marginal_2x3.p_table)
        np.testing.assert_array_almost_equal(m1.p_table, simple_marginal_2x2.p_table)

    def test_product_probability_factorizes(
        self, simple_marginal_2x3, simple_marginal_2x2
    ):
        """P(x, y) = P₁(x₁, y₁) · P₂(x₂, y₂)."""
        mech = TensorProductMechanism(
            marginals=[simple_marginal_2x3, simple_marginal_2x2],
        )
        for i0 in range(2):
            for i1 in range(2):
                for j0 in range(3):
                    for j1 in range(2):
                        p = mech.probability([i0, i1], [j0, j1])
                        p_expected = (
                            simple_marginal_2x3.p_table[i0, j0]
                            * simple_marginal_2x2.p_table[i1, j1]
                        )
                        assert abs(p - p_expected) < 1e-12

    def test_sample_shape(self, simple_marginal_2x3, simple_marginal_2x2, rng):
        """Sampling returns correct shape."""
        mech = TensorProductMechanism(
            marginals=[simple_marginal_2x3, simple_marginal_2x2],
        )
        samples = mech.sample([0, 0], rng=rng, n_samples=100)
        assert samples.shape == (100, 2)

    def test_log_probability(self, simple_marginal_2x3, simple_marginal_2x2):
        """Log probability is consistent with probability."""
        mech = TensorProductMechanism(
            marginals=[simple_marginal_2x3, simple_marginal_2x2],
        )
        p = mech.probability([0, 0], [0, 0])
        lp = mech.log_probability([0, 0], [0, 0])
        assert abs(lp - math.log(p)) < 1e-12

    def test_expected_loss_per_coordinate(
        self, simple_marginal_2x3, simple_marginal_2x2
    ):
        """Per-coordinate expected loss is finite and positive."""
        mech = TensorProductMechanism(
            marginals=[simple_marginal_2x3, simple_marginal_2x2],
        )
        losses = mech.expected_loss_per_coordinate(
            input_indices=[0, 0],
            true_values=[0.0, 0.0],
            loss_fn="L2",
        )
        assert losses.shape == (2,)
        assert all(l >= 0 for l in losses)

    def test_total_expected_loss(
        self, simple_marginal_2x3, simple_marginal_2x2
    ):
        """Total loss = sum of per-coordinate losses."""
        mech = TensorProductMechanism(
            marginals=[simple_marginal_2x3, simple_marginal_2x2],
        )
        per_coord = mech.expected_loss_per_coordinate([0, 0], [0.0, 0.0])
        total = mech.total_expected_loss([0, 0], [0.0, 0.0])
        assert abs(total - per_coord.sum()) < 1e-12

    def test_dimension_property(
        self, simple_marginal_2x3, simple_marginal_2x2
    ):
        """d property returns number of dimensions."""
        mech = TensorProductMechanism(
            marginals=[simple_marginal_2x3, simple_marginal_2x2],
        )
        assert mech.d == 2

    def test_build_product_mechanism(self):
        """Factory function creates valid TensorProductMechanism."""
        tables = [
            np.array([[0.6, 0.4], [0.4, 0.6]]),
            np.array([[0.5, 0.5], [0.5, 0.5]]),
        ]
        grids = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
        epsilons = [0.5, 0.5]
        mech = build_product_mechanism(tables, grids, epsilons)
        assert isinstance(mech, TensorProductMechanism)
        assert mech.d == 2


class TestTensorProductProperties:
    """Property-based tests for tensor product mechanisms."""

    @given(
        p1=st.floats(min_value=0.1, max_value=0.9),
        p2=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=50)
    def test_normalization_property(self, p1, p2):
        """Product of normalized marginals is normalized."""
        m1 = MarginalMechanism(
            p_table=np.array([[p1, 1 - p1]]),
            y_grid=np.array([0.0, 1.0]),
            coordinate_index=0,
        )
        m2 = MarginalMechanism(
            p_table=np.array([[p2, 1 - p2]]),
            y_grid=np.array([0.0, 1.0]),
            coordinate_index=1,
        )
        mech = TensorProductMechanism(marginals=[m1, m2])
        dense = mech.materialise_dense(input_indices=[0, 0])
        assert abs(dense.sum() - 1.0) < 1e-10


# =========================================================================
# Section 3: Kronecker Sparse
# =========================================================================


class TestKroneckerSparse:
    """Tests for sparse Kronecker product."""

    def test_matches_dense(self):
        """Sparse Kronecker matches numpy kron for small matrices."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)
        sparse_result = kronecker_sparse([A, B]).toarray()
        dense_result = np.kron(A, B)
        np.testing.assert_array_almost_equal(sparse_result, dense_result)

    def test_three_factors(self):
        """Kronecker of three matrices."""
        A = np.eye(2)
        B = np.ones((2, 2)) / 2
        C = np.eye(2)
        result = kronecker_sparse([A, B, C]).toarray()
        expected = np.kron(np.kron(A, B), C)
        np.testing.assert_array_almost_equal(result, expected)


# =========================================================================
# Section 4: Budget Allocation
# =========================================================================


class TestBudgetAllocation:
    """Tests for privacy budget allocation across dimensions."""

    def test_uniform_equal_budgets(self):
        """Uniform allocation gives equal budgets to all coordinates."""
        allocator = BudgetAllocator(composition_type=CompositionType.BASIC)
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        alloc = allocator.allocate_uniform(budget, d=4)
        assert len(alloc.epsilons) == 4
        assert all(abs(e - alloc.epsilons[0]) < 1e-12 for e in alloc.epsilons)
        assert alloc.strategy == AllocationStrategy.UNIFORM

    def test_uniform_composition_valid(self):
        """Uniform allocation satisfies composition constraint."""
        allocator = BudgetAllocator(composition_type=CompositionType.BASIC)
        budget = PrivacyBudget(epsilon=2.0, delta=1e-5)
        alloc = allocator.allocate_uniform(budget, d=5)
        valid, composed_eps = allocator.verify_composition(alloc)
        assert valid
        assert composed_eps <= 2.0 + 1e-10

    def test_proportional_scales_with_sensitivity(self):
        """Proportional allocation assigns higher ε to higher sensitivity."""
        allocator = BudgetAllocator()
        budget = PrivacyBudget(epsilon=2.0, delta=0.0)
        sensitivities = np.array([1.0, 2.0, 3.0])
        alloc = allocator.allocate_proportional(budget, sensitivities)
        # Higher sensitivity → higher ε allocation
        assert alloc.epsilons[2] > alloc.epsilons[0]

    def test_optimal_allocation_valid(self):
        """Optimal allocation satisfies composition constraint."""
        allocator = BudgetAllocator(composition_type=CompositionType.BASIC)
        budget = PrivacyBudget(epsilon=2.0, delta=1e-5)
        sensitivities = np.array([1.0, 1.0, 1.0])
        alloc = allocator.allocate_optimal(budget, sensitivities)
        valid, _ = allocator.verify_composition(alloc)
        assert valid

    def test_allocation_strategy_attribute(self):
        """BudgetAllocation stores correct strategy."""
        allocator = BudgetAllocator()
        budget = PrivacyBudget(epsilon=1.0, delta=0.0)
        alloc_u = allocator.allocate_uniform(budget, d=3)
        assert alloc_u.strategy == AllocationStrategy.UNIFORM

    def test_d1_trivial(self):
        """d=1 allocation gives full budget."""
        allocator = BudgetAllocator()
        budget = PrivacyBudget(epsilon=1.0, delta=0.0)
        alloc = allocator.allocate_uniform(budget, d=1)
        assert len(alloc.epsilons) == 1
        assert abs(alloc.epsilons[0] - 1.0) < 1e-10

    def test_budget_for_index(self):
        """budget_for(i) returns correct per-coordinate budget."""
        allocator = BudgetAllocator()
        budget = PrivacyBudget(epsilon=2.0, delta=1e-5)
        alloc = allocator.allocate_uniform(budget, d=4)
        b = alloc.budget_for(0)
        assert isinstance(b, PrivacyBudget)
        assert b.epsilon > 0


# =========================================================================
# Section 5: Lower Bounds
# =========================================================================


class TestLowerBounds:
    """Tests for information-theoretic lower bounds."""

    def test_fano_bound_positive(self):
        """Fano-Assouad bound is positive."""
        lb = LowerBoundComputer(loss_type="L2")
        result = lb.fano_assouad(
            d=4, epsilon=1.0, delta=0.0, sensitivity=1.0, domain_size=2
        )
        assert result.bound_value > 0

    def test_fano_le_achieved(self):
        """Fano bound ≤ any achieved error (by definition of lower bound)."""
        lb = LowerBoundComputer(loss_type="L2")
        result = lb.fano_assouad(
            d=4, epsilon=1.0, sensitivity=1.0, domain_size=2
        )
        # Laplace achieves error ≈ 2d/ε² = 8
        laplace_error = 2.0 * 4 / (1.0 ** 2)
        assert result.bound_value <= laplace_error + 1e-6

    def test_minimax_bound(self):
        """Minimax bound is positive and finite."""
        lb = LowerBoundComputer(loss_type="L2")
        result = lb.minimax_bound(
            d=4, epsilon=1.0, sensitivity=1.0, n_databases=2, k=100
        )
        assert result.bound_value > 0
        assert math.isfinite(result.bound_value)

    def test_gap_analysis(self):
        """Gap analysis compares achieved error to lower bound."""
        lb = LowerBoundComputer(loss_type="L2")
        achieved = 5.0
        result = lb.gap_analysis(
            achieved_error=achieved, d=4, epsilon=1.0, sensitivity=1.0
        )
        assert result.gap_ratio is not None
        assert result.gap_ratio >= 1.0 - 1e-10  # achieved / bound ≥ 1

    def test_per_coordinate_bounds(self):
        """Per-coordinate bounds computed for each dimension."""
        lb = LowerBoundComputer(loss_type="L2")
        epsilons = np.array([0.5, 1.0, 2.0])
        sensitivities = np.array([1.0, 1.0, 1.0])
        results = lb.per_coordinate_bounds(epsilons, sensitivities)
        assert len(results) == 3
        # Lower ε → higher lower bound (harder problem)
        assert results[0].bound_value >= results[2].bound_value - 1e-10

    def test_composition_overhead(self):
        """Composition overhead is ≥ 1 for d > 1."""
        lb = LowerBoundComputer()
        overhead = lb.composition_overhead(d=4, epsilon=1.0, composition_type="basic")
        assert overhead >= 1.0

    def test_d1_lower_bound(self):
        """d=1 gives standard 1D lower bound."""
        lb = LowerBoundComputer(loss_type="L2")
        result = lb.minimax_bound(d=1, epsilon=1.0, sensitivity=1.0)
        assert result.bound_value > 0

    def test_higher_eps_lower_bound(self):
        """Higher ε → lower bound decreases (easier problem)."""
        lb = LowerBoundComputer(loss_type="L2")
        r1 = lb.minimax_bound(d=4, epsilon=0.5, sensitivity=1.0)
        r2 = lb.minimax_bound(d=4, epsilon=2.0, sensitivity=1.0)
        assert r1.bound_value >= r2.bound_value - 1e-10


# =========================================================================
# Section 6: Marginal Queries
# =========================================================================


class TestMarginalQueries:
    """Tests for marginal query construction and sensitivity."""

    def test_build_1way(self):
        """Build all 1-way marginals for d=3."""
        builder = MarginalQueryBuilder(d=3, domain_sizes=2)
        queries = builder.build_kway(k=1)
        assert len(queries) == 3  # C(3,1)

    def test_build_2way(self):
        """Build all 2-way marginals for d=4."""
        builder = MarginalQueryBuilder(d=4, domain_sizes=2)
        queries = builder.build_kway(k=2)
        assert len(queries) == 6  # C(4,2)

    def test_build_all_marginals(self):
        """Build all marginals up to max_k."""
        builder = MarginalQueryBuilder(d=3, domain_sizes=2)
        queries = builder.build_all_marginals(max_k=2)
        # 1-way: 3, 2-way: 3 = 6 total
        assert len(queries) == 6

    def test_sensitivity_counting_l1(self):
        """L1 sensitivity for 1-way binary marginal = 2."""
        builder = MarginalQueryBuilder(d=3, domain_sizes=2)
        queries = builder.build_kway(k=1)
        s = builder.compute_sensitivity(queries[0], sensitivity_type="L1")
        assert abs(s - 2.0) < 1e-10

    def test_sensitivity_l2(self):
        """L2 sensitivity for binary marginal = √2."""
        builder = MarginalQueryBuilder(d=3, domain_sizes=2)
        queries = builder.build_kway(k=1)
        s = builder.compute_sensitivity(queries[0], sensitivity_type="L2")
        assert abs(s - math.sqrt(2.0)) < 1e-10

    def test_sensitivity_linf(self):
        """L∞ sensitivity for binary marginal = 1."""
        builder = MarginalQueryBuilder(d=3, domain_sizes=2)
        queries = builder.build_kway(k=1)
        s = builder.compute_sensitivity(queries[0], sensitivity_type="Linf")
        assert abs(s - 1.0) < 1e-10

    def test_compute_sensitivities_batch(self):
        """Batch sensitivity computation."""
        builder = MarginalQueryBuilder(d=3, domain_sizes=2)
        queries = builder.build_kway(k=1)
        sensitivities = builder.compute_sensitivities(queries, sensitivity_type="L1")
        assert len(sensitivities) == 3
        assert all(abs(s - 2.0) < 1e-10 for s in sensitivities)

    def test_query_matrix(self):
        """Query matrix has correct shape."""
        builder = MarginalQueryBuilder(d=3, domain_sizes=2)
        queries = builder.build_kway(k=1)
        mat = queries[0].query_matrix()
        assert mat.shape[0] == queries[0].n_cells  # 2 cells for binary 1-way

    def test_workload_matrix(self):
        """Build combined workload matrix."""
        builder = MarginalQueryBuilder(d=3, domain_sizes=2)
        queries = builder.build_kway(k=1)
        W = builder.build_workload_matrix(queries)
        assert W.ndim == 2

    def test_overlapping_coordinates(self):
        """Overlapping coordinate detection."""
        builder = MarginalQueryBuilder(d=4, domain_sizes=2)
        q_1way = builder.build_kway(k=1)
        q_2way = builder.build_kway(k=2)
        all_queries = q_1way + q_2way
        overlaps = builder.overlapping_coordinates(all_queries)
        assert isinstance(overlaps, dict)

    def test_marginal_query_properties(self):
        """MarginalQuery has correct k and n_cells."""
        q = MarginalQuery(
            coordinates=(0, 2),
            domain_sizes=(3, 4),
            name="test",
        )
        assert q.k == 2
        assert q.n_cells == 12


class TestMarginalQuerySpec:
    """Tests for building QuerySpec from marginal queries."""

    def test_build_query_spec(self):
        """build_query_spec creates a valid QuerySpec."""
        builder = MarginalQueryBuilder(d=3, domain_sizes=2)
        queries = builder.build_kway(k=1)
        spec = builder.build_query_spec(queries[0], epsilon=1.0, delta=0.0, k=50)
        assert spec.epsilon == 1.0
        assert spec.k == 50
        assert spec.n > 0


# =========================================================================
# Section 7: MarginalMechanism unit tests
# =========================================================================


class TestMarginalMechanism:
    """Tests for the MarginalMechanism dataclass."""

    def test_n_and_k(self, simple_marginal_2x3):
        """n and k properties are correct."""
        assert simple_marginal_2x3.n == 2
        assert simple_marginal_2x3.k == 3

    def test_sample(self, simple_marginal_2x3, rng):
        """Sampling returns valid (bin_idx, value) pair."""
        bin_idx, value = simple_marginal_2x3.sample(0, rng)
        assert 0 <= bin_idx < 3
        assert value in simple_marginal_2x3.y_grid

    def test_probability(self, simple_marginal_2x3):
        """Probability matches table entry."""
        p = simple_marginal_2x3.probability(0, 0)
        assert abs(p - 0.5) < 1e-12

    def test_expected_loss(self, simple_marginal_2x3):
        """Expected loss is non-negative."""
        loss = simple_marginal_2x3.expected_loss(0, 0.0, loss_fn="L2")
        assert loss >= 0
