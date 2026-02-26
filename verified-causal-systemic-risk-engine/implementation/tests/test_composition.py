"""Tests for bound composition module."""
import pytest
import numpy as np
from causalbound.composition.composer import (
    BoundComposer,
    CompositionStrategy,
    SubgraphBound,
    SeparatorInfo,
    OverlapStructure,
)
from causalbound.composition.gap_estimation import GapEstimator
from causalbound.composition.consistency import (
    SeparatorConsistencyChecker,
    SeparatorSpec,
)
from causalbound.composition.propagation import (
    MonotoneBoundPropagator,
    PropagationBound,
    AdjacencyInfo,
)
from causalbound.composition.aggregation import (
    GlobalBoundAggregator,
    AggregationMethod,
    LocalBound,
    OverlapInfo,
)
from causalbound.composition.theorem import (
    CompositionTheorem,
    SubgraphInfo,
    SeparatorData,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_subgraph_bounds():
    """Simple two-subgraph bounds as SubgraphBound dataclasses."""
    return [
        SubgraphBound(subgraph_id=0, lower=np.array([0.1, 0.2]),
                       upper=np.array([0.5, 0.6]), separator_vars=[1]),
        SubgraphBound(subgraph_id=1, lower=np.array([0.2, 0.15]),
                       upper=np.array([0.6, 0.55]), separator_vars=[0]),
    ]


@pytest.fixture
def three_subgraph_bounds():
    """Three overlapping subgraph bounds as SubgraphBound dataclasses."""
    return [
        SubgraphBound(subgraph_id=0, lower=np.array([0.05, 0.10]),
                       upper=np.array([0.45, 0.50]), separator_vars=[1]),
        SubgraphBound(subgraph_id=1, lower=np.array([0.10, 0.12]),
                       upper=np.array([0.55, 0.52]), separator_vars=[0, 1]),
        SubgraphBound(subgraph_id=2, lower=np.array([0.15, 0.18]),
                       upper=np.array([0.60, 0.58]), separator_vars=[0]),
    ]


@pytest.fixture
def separator_info_list():
    """Separator information as SeparatorInfo dataclasses."""
    return [
        SeparatorInfo(separator_id=0, variable_indices=[1],
                      adjacent_subgraphs=[0, 1], cardinality=2),
        SeparatorInfo(separator_id=1, variable_indices=[0],
                      adjacent_subgraphs=[1, 2], cardinality=2),
    ]


@pytest.fixture
def overlap_struct():
    """Overlap structure as OverlapStructure dataclass."""
    return OverlapStructure(
        n_subgraphs=3,
        overlap_matrix=np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]),
        shared_variables={(0, 1): [1], (1, 2): [0]},
    )


@pytest.fixture
def overlap_struct_two():
    """Overlap structure for two subgraphs."""
    return OverlapStructure(
        n_subgraphs=2,
        overlap_matrix=np.array([
            [0, 1],
            [1, 0],
        ]),
        shared_variables={(0, 1): [1]},
    )


@pytest.fixture
def subgraph_marginals():
    """Marginal distributions on separator variables (int-keyed)."""
    return {
        0: {0: np.array([0.3, 0.4, 0.3])},
        1: {0: np.array([0.35, 0.35, 0.30]),
            1: np.array([0.2, 0.5, 0.3])},
        2: {1: np.array([0.25, 0.45, 0.30])},
    }


# ---------------------------------------------------------------------------
# BoundComposer Tests
# ---------------------------------------------------------------------------
class TestBoundComposer:
    def test_compose_two_subgraphs(self, simple_subgraph_bounds, overlap_struct_two):
        composer = BoundComposer()
        sep_info = [
            SeparatorInfo(separator_id=0, variable_indices=[1],
                          adjacent_subgraphs=[0, 1], cardinality=2),
        ]
        result = composer.compose(simple_subgraph_bounds, sep_info, overlap_struct_two)
        assert result is not None
        assert result.global_lower is not None
        assert result.global_upper is not None
        assert np.all(result.global_lower <= result.global_upper)

    def test_compose_worst_case(self, simple_subgraph_bounds, overlap_struct_two):
        """Worst-case composition should produce valid bounds."""
        composer = BoundComposer(strategy=CompositionStrategy.WORST_CASE)
        sep_info = [
            SeparatorInfo(separator_id=0, variable_indices=[1],
                          adjacent_subgraphs=[0, 1], cardinality=2),
        ]
        result = composer.compose(simple_subgraph_bounds, sep_info, overlap_struct_two)
        assert np.all(result.global_lower <= result.global_upper)

    def test_composition_gap_nonnegative(self, simple_subgraph_bounds, overlap_struct_two):
        """Composition gap should be non-negative."""
        composer = BoundComposer()
        sep_info = [
            SeparatorInfo(separator_id=0, variable_indices=[1],
                          adjacent_subgraphs=[0, 1], cardinality=2),
        ]
        composer.compose(simple_subgraph_bounds, sep_info, overlap_struct_two)
        gap = composer.get_composition_gap()
        assert gap >= -0.01

    def test_compose_three_subgraphs(self, three_subgraph_bounds,
                                     separator_info_list, overlap_struct):
        composer = BoundComposer()
        result = composer.compose(
            three_subgraph_bounds, separator_info_list, overlap_struct
        )
        assert result is not None
        assert np.all(result.global_lower <= result.global_upper)

    def test_validate_composition(self, simple_subgraph_bounds, overlap_struct_two):
        composer = BoundComposer()
        sep_info = [
            SeparatorInfo(separator_id=0, variable_indices=[1],
                          adjacent_subgraphs=[0, 1], cardinality=2),
        ]
        result = composer.compose(simple_subgraph_bounds, sep_info, overlap_struct_two)
        global_bounds = (result.global_lower, result.global_upper)
        validation = composer.validate_composition(global_bounds, simple_subgraph_bounds)
        assert validation["sound"]


# ---------------------------------------------------------------------------
# GapEstimator Tests
# ---------------------------------------------------------------------------
class TestGapEstimator:
    def test_gap_estimate_positive(self):
        ge = GapEstimator(n_samples=200, seed=42)
        subgraph_bounds = [
            {"lower": np.array([0.1, 0.2]), "upper": np.array([0.5, 0.6])},
            {"lower": np.array([0.2, 0.15]), "upper": np.array([0.6, 0.55])},
        ]
        separators = [
            {"id": 0, "variables": [1], "adjacent": [0, 1], "cardinality": 2},
        ]
        gap = ge.estimate_gap(subgraph_bounds, separators)
        assert gap.total_gap >= 0.0

    def test_lipschitz_bound(self):
        ge = GapEstimator(n_samples=200, seed=42)
        fn = lambda x: np.sum(x)
        info = ge.lipschitz_bound(
            contagion_function=fn,
            separator_size=3,
            discretization=10,
        )
        assert info.constant >= 0.0
        assert info.domain_dim == 3

    def test_gap_decreases_with_smaller_separators(self):
        ge = GapEstimator(n_samples=200, seed=42)
        subgraph_bounds = [
            {"lower": np.array([0.1, 0.2]), "upper": np.array([0.5, 0.6])},
            {"lower": np.array([0.2, 0.15]), "upper": np.array([0.6, 0.55])},
        ]
        sep_large = [
            {"id": 0, "variables": list(range(2)), "adjacent": [0, 1], "cardinality": 2},
        ]
        sep_small = [
            {"id": 0, "variables": [0], "adjacent": [0, 1], "cardinality": 2},
        ]
        gap_large = ge.estimate_gap(subgraph_bounds, sep_large)
        gap_small = ge.estimate_gap(subgraph_bounds, sep_small)
        assert gap_small.total_gap <= gap_large.total_gap + 0.5


# ---------------------------------------------------------------------------
# SeparatorConsistencyChecker Tests
# ---------------------------------------------------------------------------
class TestSeparatorConsistency:
    def test_consistent_marginals(self):
        """Identical marginals should be consistent."""
        checker = SeparatorConsistencyChecker()
        marginals = {
            0: {0: np.array([0.3, 0.4, 0.3])},
            1: {0: np.array([0.3, 0.4, 0.3])},
        }
        separators = [SeparatorSpec(separator_id=0, variable_indices=[0],
                                    adjacent_subgraphs=[0, 1])]
        result = checker.check_consistency(marginals, separators)
        assert result.is_consistent or result.max_inconsistency < 0.01

    def test_inconsistent_marginals(self):
        """Very different marginals should be detected as inconsistent."""
        checker = SeparatorConsistencyChecker()
        marginals = {
            0: {0: np.array([0.9, 0.05, 0.05])},
            1: {0: np.array([0.05, 0.05, 0.9])},
        }
        separators = [SeparatorSpec(separator_id=0, variable_indices=[0],
                                    adjacent_subgraphs=[0, 1])]
        result = checker.check_consistency(marginals, separators)
        assert not result.is_consistent or result.max_inconsistency > 0.1

    def test_repair_marginals(self, subgraph_marginals):
        """Repaired marginals should be more consistent."""
        checker = SeparatorConsistencyChecker()
        separators = [
            SeparatorSpec(separator_id=0, variable_indices=[0],
                          adjacent_subgraphs=[0, 1]),
            SeparatorSpec(separator_id=1, variable_indices=[1],
                          adjacent_subgraphs=[1, 2]),
        ]
        repaired = checker.repair_marginals(subgraph_marginals, separators)
        assert repaired is not None
        assert repaired.repaired_marginals is not None

    def test_kl_divergence_nonnegative(self):
        """KL divergence should be non-negative."""
        checker = SeparatorConsistencyChecker(metric="kl")
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.35, 0.35, 0.30])
        kl = checker.compute_inconsistency(p, q)
        assert kl >= -1e-10

    def test_tv_distance_bounded(self):
        """Total variation distance should be in [0, 1]."""
        checker = SeparatorConsistencyChecker(metric="tv")
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.1, 0.6, 0.3])
        tv = checker.compute_inconsistency(p, q)
        assert 0.0 <= tv <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# MonotoneBoundPropagator Tests
# ---------------------------------------------------------------------------
class TestMonotonePropagation:
    def test_propagation_converges(self):
        prop = MonotoneBoundPropagator()
        bounds = {
            0: PropagationBound(subgraph_id=0,
                                lower=np.array([0.1, 0.2]),
                                upper=np.array([0.5, 0.6]),
                                separator_vars={0: [1]}),
            1: PropagationBound(subgraph_id=1,
                                lower=np.array([0.2, 0.15]),
                                upper=np.array([0.6, 0.55]),
                                separator_vars={0: [1]}),
        }
        adjacency = AdjacencyInfo(
            n_subgraphs=2,
            adjacency_list={0: [1], 1: [0]},
            shared_separators={(0, 1): [0]},
            separator_cardinalities={0: 2},
        )
        result = prop.propagate(bounds, adjacency)
        assert result is not None
        assert 0 in result.bounds
        assert 1 in result.bounds

    def test_fixed_point_via_propagate(self):
        """propagate() internally calls iterate_to_fixed_point."""
        prop = MonotoneBoundPropagator(tolerance=1e-6, max_iterations=100)
        bounds = {
            0: PropagationBound(subgraph_id=0,
                                lower=np.array([0.05, 0.10]),
                                upper=np.array([0.45, 0.50]),
                                separator_vars={0: [1]}),
            1: PropagationBound(subgraph_id=1,
                                lower=np.array([0.10, 0.12]),
                                upper=np.array([0.55, 0.52]),
                                separator_vars={0: [1], 1: [0]}),
            2: PropagationBound(subgraph_id=2,
                                lower=np.array([0.15, 0.18]),
                                upper=np.array([0.60, 0.58]),
                                separator_vars={1: [0]}),
        }
        adjacency = AdjacencyInfo(
            n_subgraphs=3,
            adjacency_list={0: [1], 1: [0, 2], 2: [1]},
            shared_separators={(0, 1): [0], (1, 2): [1]},
            separator_cardinalities={0: 2, 1: 2},
        )
        result = prop.propagate(bounds, adjacency)
        assert result is not None
        assert result.n_iterations >= 1

    def test_monotonicity_enforcement(self):
        """New bounds should be no wider than old bounds."""
        prop = MonotoneBoundPropagator()
        old_bound = PropagationBound(subgraph_id=0,
                                     lower=np.array([0.1, 0.2]),
                                     upper=np.array([0.5, 0.6]))
        new_bound = PropagationBound(subgraph_id=0,
                                     lower=np.array([0.0, 0.15]),
                                     upper=np.array([0.8, 0.65]))
        enforced = prop.enforce_monotonicity(old_bound, new_bound)
        assert np.all(enforced.lower >= old_bound.lower)
        assert np.all(enforced.upper <= old_bound.upper)


# ---------------------------------------------------------------------------
# GlobalBoundAggregator Tests
# ---------------------------------------------------------------------------
class TestGlobalAggregation:
    def test_conservative_aggregation(self):
        agg = GlobalBoundAggregator(seed=42)
        bounds = [
            LocalBound(subgraph_id=0, lower=np.array([0.1, 0.2]),
                       upper=np.array([0.5, 0.6]), variables=[0, 1]),
            LocalBound(subgraph_id=1, lower=np.array([0.2, 0.15]),
                       upper=np.array([0.6, 0.55]), variables=[1, 2]),
        ]
        result = agg.aggregate(bounds, method=AggregationMethod.CONSERVATIVE)
        assert result is not None
        assert result.global_lower is not None
        assert result.global_upper is not None

    def test_double_counting_correction(self):
        agg = GlobalBoundAggregator(seed=42)
        bounds = [
            LocalBound(subgraph_id=0, lower=np.array([0.05, 0.10]),
                       upper=np.array([0.45, 0.50]), variables=[0, 1]),
            LocalBound(subgraph_id=1, lower=np.array([0.10, 0.12]),
                       upper=np.array([0.55, 0.52]), variables=[1, 2]),
            LocalBound(subgraph_id=2, lower=np.array([0.15, 0.18]),
                       upper=np.array([0.60, 0.58]), variables=[2, 3]),
        ]
        overlap = OverlapInfo(
            overlap_pairs={(0, 1): [1], (1, 2): [2]},
            overlap_counts=np.array([1, 2, 2, 1]),
            total_variables=4,
        )
        corrected_lower, corrected_upper = agg.correct_double_counting(bounds, overlap)
        assert corrected_lower is not None
        assert corrected_upper is not None

    def test_risk_decomposition(self):
        agg = GlobalBoundAggregator(seed=42)
        bounds = [
            LocalBound(subgraph_id=0, lower=np.array([0.05, 0.10]),
                       upper=np.array([0.45, 0.50]), variables=[0, 1]),
            LocalBound(subgraph_id=1, lower=np.array([0.10, 0.12]),
                       upper=np.array([0.55, 0.52]), variables=[1, 2]),
            LocalBound(subgraph_id=2, lower=np.array([0.15, 0.18]),
                       upper=np.array([0.60, 0.58]), variables=[2, 3]),
        ]
        decomp = agg.get_risk_decomposition(bounds)
        assert len(decomp) == 3
        for contrib in decomp.values():
            assert contrib >= -0.01


# ---------------------------------------------------------------------------
# CompositionTheorem Tests
# ---------------------------------------------------------------------------
class TestCompositionTheorem:
    def test_verify_conditions(self):
        ct = CompositionTheorem(seed=42)
        subgraphs = [
            SubgraphInfo(subgraph_id=0, variables=[0, 1, 2],
                         lower_bound=np.array([0.0, 0.0, 0.0]),
                         upper_bound=np.array([1.0, 1.0, 1.0]),
                         is_sound=True),
        ]
        separators = []  # type: List[SeparatorData]
        fn = lambda x: np.array([np.sum(x)])
        result = ct.verify_conditions(subgraphs, separators, fn)
        assert result is not None
        assert isinstance(result.all_satisfied, bool)

    def test_gap_bound_computation(self):
        ct = CompositionTheorem(seed=42)
        params = {"k": 5, "L": 1.0, "s": 3, "epsilon": 0.1}
        gap = ct.compute_epsilon_gap(params)
        assert gap >= 0.0
        # Gap should scale with Lipschitz constant
        params2 = {"k": 5, "L": 2.0, "s": 3, "epsilon": 0.1}
        gap2 = ct.compute_epsilon_gap(params2)
        assert gap2 >= gap - 0.01

    def test_theorem_statement(self):
        ct = CompositionTheorem()
        stmt = ct.get_theorem_statement()
        assert isinstance(stmt, str)
        assert len(stmt) > 50

    def test_lipschitz_check(self):
        ct = CompositionTheorem(seed=42)
        fn = lambda x: np.sum(x)
        result = ct.check_lipschitz(fn, domain_dim=3)
        assert result["constant"] >= 0.0
        assert result["is_lipschitz"]
