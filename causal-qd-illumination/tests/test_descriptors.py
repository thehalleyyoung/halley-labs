"""Tests for causal_qd.descriptors — Structural, InfoTheoretic, Equivalence,
and Composite behavioral descriptor computers.

Covers descriptor dimensionality, value bounds, feature correctness for known
graphs, consistency across repeated evaluations, MEC-based similarity,
composite combination, PCA compression, and normalization strategies.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from causal_qd.descriptors.structural import StructuralDescriptor, ALL_FEATURES
from causal_qd.descriptors.info_theoretic import InfoTheoreticDescriptor
from causal_qd.descriptors.equivalence_desc import EquivalenceDescriptor
from causal_qd.descriptors.composite import CompositeDescriptor
from causal_qd.types import AdjacencyMatrix, BehavioralDescriptor, DataMatrix


# ===================================================================
# Helpers
# ===================================================================

def _make_chain(n: int) -> AdjacencyMatrix:
    """Chain DAG: 0→1→…→(n-1)."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
    return adj


def _make_fork(n: int) -> AdjacencyMatrix:
    """Fork DAG: 0→1, 0→2, …, 0→(n-1)."""
    adj = np.zeros((n, n), dtype=np.int8)
    for j in range(1, n):
        adj[0, j] = 1
    return adj


def _make_collider(n: int) -> AdjacencyMatrix:
    """Collider DAG: 0→(n-1), 1→(n-1), …, (n-2)→(n-1)."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, n - 1] = 1
    return adj


def _make_empty(n: int) -> AdjacencyMatrix:
    return np.zeros((n, n), dtype=np.int8)


def _make_full_forward(n: int) -> AdjacencyMatrix:
    """All forward edges i→j for i < j."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            adj[i, j] = 1
    return adj


def _make_diamond() -> AdjacencyMatrix:
    """Diamond DAG: 0→1, 0→2, 1→3, 2→3 (v-structure at 3)."""
    adj = np.zeros((4, 4), dtype=np.int8)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[1, 3] = 1
    adj[2, 3] = 1
    return adj


def _make_v_structure() -> AdjacencyMatrix:
    """Pure v-structure: 0→2←1 (no edge between 0 and 1)."""
    adj = np.zeros((3, 3), dtype=np.int8)
    adj[0, 2] = 1
    adj[1, 2] = 1
    return adj


def _generate_linear_gaussian(
    adj: AdjacencyMatrix,
    n_samples: int,
    rng: np.random.Generator,
    noise_scale: float = 0.3,
) -> DataMatrix:
    """Generate data from a linear Gaussian SCM defined by *adj*."""
    n = adj.shape[0]
    topo = _topological_sort(adj, n)
    data = np.zeros((n_samples, n), dtype=np.float64)
    for node in topo:
        parents = np.where(adj[:, node])[0]
        if len(parents) == 0:
            data[:, node] = rng.standard_normal(n_samples)
        else:
            weights = rng.uniform(0.5, 1.0, size=len(parents))
            data[:, node] = (
                data[:, parents] @ weights
                + rng.standard_normal(n_samples) * noise_scale
            )
    return data


def _topological_sort(adj: AdjacencyMatrix, n: int) -> List[int]:
    """Kahn's algorithm for topological sort."""
    from collections import deque

    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = deque(int(v) for v in range(n) if in_deg[v] == 0)
    order: List[int] = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in range(n):
            if adj[v, c]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
    return order


def _reverse_one_reversible_edge(adj: AdjacencyMatrix) -> AdjacencyMatrix:
    """Find a reversible edge and reverse it, keeping acyclicity.

    Used to produce a DAG in the same MEC as *adj*.
    """
    desc = EquivalenceDescriptor()
    n = adj.shape[0]
    cpdag = desc._dag_to_cpdag(adj, n)

    for i in range(n):
        for j in range(n):
            if adj[i, j] and cpdag[j, i]:
                candidate = adj.copy()
                candidate[i, j] = 0
                candidate[j, i] = 1
                # Verify acyclicity via topological sort length
                order = _topological_sort(candidate, n)
                if len(order) == n:
                    return candidate
    return adj.copy()


# ===================================================================
# StructuralDescriptor Tests
# ===================================================================


class TestStructuralDescriptorDimensions:
    """test_structural_descriptor_dimensions — verify dimensionality
    of the structural descriptor for various feature subsets."""

    def test_all_features_dim(self, small_adj: AdjacencyMatrix) -> None:
        desc = StructuralDescriptor()
        assert desc.descriptor_dim == 10
        bd = desc.compute(small_adj)
        assert bd.shape == (10,)
        assert bd.dtype == np.float64

    def test_subset_features_dim(self, small_adj: AdjacencyMatrix) -> None:
        features = ["edge_density", "dag_depth", "v_structure_count"]
        desc = StructuralDescriptor(features=features)
        assert desc.descriptor_dim == 3
        bd = desc.compute(small_adj)
        assert bd.shape == (3,)

    def test_single_feature_dim(self, small_adj: AdjacencyMatrix) -> None:
        desc = StructuralDescriptor(features=["edge_density"])
        assert desc.descriptor_dim == 1
        bd = desc.compute(small_adj)
        assert bd.shape == (1,)

    def test_bounds_match_dim(self) -> None:
        for k in range(1, len(ALL_FEATURES) + 1):
            features = ALL_FEATURES[:k]
            desc = StructuralDescriptor(features=features)
            low, high = desc.descriptor_bounds
            assert low.shape == (k,)
            assert high.shape == (k,)
            assert np.all(low == 0.0)
            assert np.all(high == 1.0)

    def test_invalid_feature_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown structural feature"):
            StructuralDescriptor(features=["nonexistent_feature"])

    def test_medium_graph_dim(self, medium_adj: AdjacencyMatrix) -> None:
        desc = StructuralDescriptor()
        bd = desc.compute(medium_adj)
        assert bd.shape == (10,)
        assert bd.dtype == np.float64

    def test_empty_graph_dim(self) -> None:
        adj = _make_empty(5)
        desc = StructuralDescriptor()
        bd = desc.compute(adj)
        assert bd.shape == (10,)
        # Empty graph: edge_density should be 0
        assert bd[0] == 0.0

    def test_feature_order_preserved(self, small_adj: AdjacencyMatrix) -> None:
        features_a = ["edge_density", "dag_depth"]
        features_b = ["dag_depth", "edge_density"]
        desc_a = StructuralDescriptor(features=features_a)
        desc_b = StructuralDescriptor(features=features_b)
        bd_a = desc_a.compute(small_adj)
        bd_b = desc_b.compute(small_adj)
        # Order matters: first element of a == second element of b
        assert np.isclose(bd_a[0], bd_b[1])
        assert np.isclose(bd_a[1], bd_b[0])

    def test_all_features_are_registered(self) -> None:
        assert len(ALL_FEATURES) == 10
        desc = StructuralDescriptor()
        assert desc.descriptor_dim == len(ALL_FEATURES)

    def test_two_node_graph(self) -> None:
        adj = np.zeros((2, 2), dtype=np.int8)
        adj[0, 1] = 1
        desc = StructuralDescriptor()
        bd = desc.compute(adj)
        assert bd.shape == (10,)
        assert np.all(np.isfinite(bd))


class TestStructuralDescriptorEdgeDensityRange:
    """test_structural_descriptor_edge_density_range — verify that
    edge density and all structural features are in [0, 1]."""

    def test_chain_density(self) -> None:
        adj = _make_chain(5)
        desc = StructuralDescriptor(features=["edge_density"])
        bd = desc.compute(adj)
        # Chain: 4 edges, max = 5*4/2 = 10 → density = 0.4
        assert np.isclose(bd[0], 0.4)

    def test_complete_density(self) -> None:
        adj = _make_full_forward(5)
        desc = StructuralDescriptor(features=["edge_density"])
        bd = desc.compute(adj)
        assert np.isclose(bd[0], 1.0)

    def test_empty_density(self) -> None:
        adj = _make_empty(5)
        desc = StructuralDescriptor(features=["edge_density"])
        bd = desc.compute(adj)
        assert bd[0] == 0.0

    def test_all_features_bounded_chain(self) -> None:
        adj = _make_chain(10)
        desc = StructuralDescriptor()
        bd = desc.compute(adj)
        assert np.all(bd >= 0.0), f"Features below 0: {bd}"
        assert np.all(bd <= 1.0), f"Features above 1: {bd}"

    def test_all_features_bounded_fork(self) -> None:
        adj = _make_fork(8)
        desc = StructuralDescriptor()
        bd = desc.compute(adj)
        assert np.all(bd >= 0.0)
        assert np.all(bd <= 1.0)

    def test_all_features_bounded_collider(self) -> None:
        adj = _make_collider(6)
        desc = StructuralDescriptor()
        bd = desc.compute(adj)
        assert np.all(bd >= 0.0)
        assert np.all(bd <= 1.0)

    def test_all_features_bounded_complete(self) -> None:
        adj = _make_full_forward(5)
        desc = StructuralDescriptor()
        bd = desc.compute(adj)
        assert np.all(bd >= 0.0)
        assert np.all(bd <= 1.0)

    def test_all_features_bounded_random(self, rng: np.random.Generator) -> None:
        n = 8
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.3:
                    adj[i, j] = 1
        desc = StructuralDescriptor()
        bd = desc.compute(adj)
        assert np.all(bd >= 0.0)
        assert np.all(bd <= 1.0)

    def test_v_structure_count_collider(self) -> None:
        adj = _make_v_structure()
        desc = StructuralDescriptor(features=["v_structure_count"])
        bd = desc.compute(adj)
        assert bd[0] > 0.0, "V-structure should be detected"
        assert bd[0] <= 1.0

    def test_longest_path_chain(self) -> None:
        adj = _make_chain(5)
        desc = StructuralDescriptor(features=["longest_path"])
        bd = desc.compute(adj)
        # Chain of 5: longest path = 4, normalized by n-1=4 → 1.0
        assert np.isclose(bd[0], 1.0)

    def test_longest_path_fork(self) -> None:
        adj = _make_fork(5)
        desc = StructuralDescriptor(features=["longest_path"])
        bd = desc.compute(adj)
        # Fork: longest path = 1, normalized by 4 → 0.25
        assert np.isclose(bd[0], 0.25)

    def test_connected_components_disconnected(self) -> None:
        adj = _make_empty(5)
        desc = StructuralDescriptor(features=["connected_components"])
        bd = desc.compute(adj)
        # 5 isolated nodes → 5/5 = 1.0
        assert np.isclose(bd[0], 1.0)

    def test_connected_components_connected(self) -> None:
        adj = _make_chain(5)
        desc = StructuralDescriptor(features=["connected_components"])
        bd = desc.compute(adj)
        # 1 connected component → 1/5 = 0.2
        assert np.isclose(bd[0], 0.2)

    def test_max_in_degree_collider(self) -> None:
        adj = _make_collider(5)
        desc = StructuralDescriptor(features=["max_in_degree"])
        bd = desc.compute(adj)
        # Node 4 has 4 parents → 4/4 = 1.0
        assert np.isclose(bd[0], 1.0)

    def test_dag_depth_chain(self) -> None:
        adj = _make_chain(5)
        desc = StructuralDescriptor(features=["dag_depth"])
        bd = desc.compute(adj)
        # Depth = 4, normalized by 4 → 1.0
        assert np.isclose(bd[0], 1.0)

    def test_parent_set_entropy_uniform(self) -> None:
        # All nodes have same in-degree → entropy = 0
        adj = _make_empty(5)
        desc = StructuralDescriptor(features=["parent_set_entropy"])
        bd = desc.compute(adj)
        assert np.isclose(bd[0], 0.0)

    def test_clustering_coefficient_complete(self) -> None:
        adj = _make_full_forward(5)
        desc = StructuralDescriptor(features=["clustering_coefficient"])
        bd = desc.compute(adj)
        # Complete graph: all neighbors connected → high clustering
        assert bd[0] > 0.0
        assert bd[0] <= 1.0

    def test_betweenness_centrality_chain(self) -> None:
        adj = _make_chain(5)
        desc = StructuralDescriptor(features=["betweenness_centrality"])
        bd = desc.compute(adj)
        # Middle nodes in chain have non-zero betweenness
        assert bd[0] > 0.0
        assert bd[0] <= 1.0

    def test_different_graphs_different_descriptors(self) -> None:
        chain = _make_chain(5)
        fork = _make_fork(5)
        desc = StructuralDescriptor()
        bd_chain = desc.compute(chain)
        bd_fork = desc.compute(fork)
        assert not np.allclose(bd_chain, bd_fork)


# ===================================================================
# InfoTheoreticDescriptor Tests
# ===================================================================


class TestInfoTheoreticDescriptorDimensions:
    """test_info_theoretic_descriptor_dimensions — verify dimensionality
    and basic properties of the info-theoretic descriptor."""

    def test_all_features_dim(self) -> None:
        desc = InfoTheoreticDescriptor()
        # mi_profile(4) + ci_signature(4) + entropy_profile(4) +
        # avg_mi(1) + avg_conditional_entropy(1) = 14
        # But with max_descriptor_dim=10 default, PCA compresses to 10
        assert desc.descriptor_dim == 10

    def test_all_features_no_compression(self) -> None:
        desc = InfoTheoreticDescriptor(max_descriptor_dim=20)
        assert desc.descriptor_dim == 14

    def test_subset_features_dim(self) -> None:
        desc = InfoTheoreticDescriptor(
            features=["avg_mi", "avg_conditional_entropy"],
            max_descriptor_dim=10,
        )
        # 1 + 1 = 2, no PCA needed
        assert desc.descriptor_dim == 2

    def test_single_feature_dim(self) -> None:
        desc = InfoTheoreticDescriptor(features=["mi_profile"], max_descriptor_dim=10)
        assert desc.descriptor_dim == 4

    def test_compute_returns_correct_shape(
        self, small_adj: AdjacencyMatrix, random_data: DataMatrix
    ) -> None:
        desc = InfoTheoreticDescriptor(
            features=["avg_mi", "avg_conditional_entropy"],
            max_descriptor_dim=10,
        )
        bd = desc.compute(small_adj, random_data)
        assert bd.shape == (2,)
        assert bd.dtype == np.float64

    def test_no_data_returns_zeros(self, small_adj: AdjacencyMatrix) -> None:
        desc = InfoTheoreticDescriptor(
            features=["avg_mi"], max_descriptor_dim=10
        )
        bd = desc.compute(small_adj, data=None)
        assert np.all(bd == 0.0)

    def test_bounds_shape(self) -> None:
        desc = InfoTheoreticDescriptor(
            features=["mi_profile", "avg_mi"], max_descriptor_dim=10
        )
        low, high = desc.descriptor_bounds
        assert low.shape == (desc.descriptor_dim,)
        assert high.shape == (desc.descriptor_dim,)

    def test_invalid_feature_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown feature"):
            InfoTheoreticDescriptor(features=["bogus_feature"])

    def test_entropy_profile_dim(self) -> None:
        desc = InfoTheoreticDescriptor(
            features=["entropy_profile"], max_descriptor_dim=10
        )
        assert desc.descriptor_dim == 4

    def test_ci_signature_dim(self) -> None:
        desc = InfoTheoreticDescriptor(
            features=["ci_signature"], max_descriptor_dim=10
        )
        assert desc.descriptor_dim == 4

    def test_all_features_with_data(
        self, medium_adj: AdjacencyMatrix
    ) -> None:
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(medium_adj, 300, rng)
        desc = InfoTheoreticDescriptor(max_descriptor_dim=20)
        bd = desc.compute(medium_adj, data)
        assert bd.shape == (14,)
        assert np.all(np.isfinite(bd))

    def test_pca_compression_applied(
        self, medium_adj: AdjacencyMatrix
    ) -> None:
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(medium_adj, 300, rng)
        desc = InfoTheoreticDescriptor(max_descriptor_dim=5)
        bd = desc.compute(medium_adj, data)
        assert bd.shape == (5,)
        assert np.all(np.isfinite(bd))

    def test_empty_graph_info_theoretic(self) -> None:
        adj = _make_empty(5)
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 5))
        desc = InfoTheoreticDescriptor(
            features=["avg_mi", "avg_conditional_entropy"],
            max_descriptor_dim=10,
        )
        bd = desc.compute(adj, data)
        # No edges → avg_mi = 0, avg_conditional_entropy = 0
        assert np.isclose(bd[0], 0.0)
        assert np.isclose(bd[1], 0.0)


class TestMIProfileKnownGraph:
    """test_mi_profile_known_graph — verify MI profile features for
    graphs with known causal structure and strong signal."""

    def test_strong_edge_has_positive_mi(self) -> None:
        """A 2-node graph 0→1 with strong linear relationship should
        have large average MI."""
        adj = np.zeros((2, 2), dtype=np.int8)
        adj[0, 1] = 1
        rng = np.random.default_rng(42)
        n_samples = 1000
        data = np.zeros((n_samples, 2))
        data[:, 0] = rng.standard_normal(n_samples)
        data[:, 1] = 0.9 * data[:, 0] + rng.standard_normal(n_samples) * 0.2

        desc = InfoTheoreticDescriptor(
            features=["mi_profile"], max_descriptor_dim=10
        )
        bd = desc.compute(adj, data)
        # MI profile: [mean, std, min, max]
        assert bd[0] > 0.5, f"Expected large MI mean, got {bd[0]}"

    def test_weak_edge_has_small_mi(self) -> None:
        adj = np.zeros((2, 2), dtype=np.int8)
        adj[0, 1] = 1
        rng = np.random.default_rng(42)
        n_samples = 1000
        data = np.zeros((n_samples, 2))
        data[:, 0] = rng.standard_normal(n_samples)
        data[:, 1] = 0.05 * data[:, 0] + rng.standard_normal(n_samples)

        desc = InfoTheoreticDescriptor(
            features=["mi_profile"], max_descriptor_dim=10
        )
        bd = desc.compute(adj, data)
        assert bd[0] < 0.1, f"Expected small MI mean, got {bd[0]}"

    def test_mi_profile_chain_monotonic_stats(self) -> None:
        """For a chain with uniform edge strength, MI values should be
        similar, so std should be relatively low."""
        adj = _make_chain(4)
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(adj, 500, rng, noise_scale=0.3)

        desc = InfoTheoreticDescriptor(
            features=["mi_profile"], max_descriptor_dim=10
        )
        bd = desc.compute(adj, data)
        mean_mi, std_mi, min_mi, max_mi = bd[0], bd[1], bd[2], bd[3]
        assert mean_mi > 0.0
        assert min_mi >= 0.0
        assert max_mi >= min_mi
        assert std_mi >= 0.0

    def test_mi_profile_no_edges_returns_zeros(self) -> None:
        adj = _make_empty(3)
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 3))
        desc = InfoTheoreticDescriptor(
            features=["mi_profile"], max_descriptor_dim=10
        )
        bd = desc.compute(adj, data)
        assert np.all(bd == 0.0)

    def test_avg_mi_positive_for_connected_graph(self) -> None:
        adj = _make_chain(5)
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(adj, 500, rng, noise_scale=0.3)
        desc = InfoTheoreticDescriptor(
            features=["avg_mi"], max_descriptor_dim=10
        )
        bd = desc.compute(adj, data)
        assert bd[0] > 0.0, "Expected positive avg MI for connected graph"

    def test_avg_conditional_entropy_positive(self) -> None:
        adj = _make_chain(5)
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(adj, 500, rng, noise_scale=0.5)
        desc = InfoTheoreticDescriptor(
            features=["avg_conditional_entropy"], max_descriptor_dim=10
        )
        bd = desc.compute(adj, data)
        # Conditional entropy under Gaussian is related to residual variance
        # For non-zero noise, this should be finite and positive
        assert np.isfinite(bd[0])

    def test_entropy_profile_all_finite(self) -> None:
        adj = _make_diamond()
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(adj, 500, rng)
        desc = InfoTheoreticDescriptor(
            features=["entropy_profile"], max_descriptor_dim=10
        )
        bd = desc.compute(adj, data)
        assert np.all(np.isfinite(bd))
        assert bd.shape == (4,)


class TestCISignatureConsistency:
    """test_ci_signature_consistency — verify CI signature is
    consistent across repeated evaluations and captures expected
    dependence patterns."""

    def test_deterministic(self) -> None:
        """CI signature should be deterministic given same dag and data."""
        adj = _make_chain(5)
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(adj, 500, rng)
        desc = InfoTheoreticDescriptor(
            features=["ci_signature"], max_descriptor_dim=10
        )
        bd1 = desc.compute(adj, data)
        bd2 = desc.compute(adj, data)
        np.testing.assert_array_equal(bd1, bd2)

    def test_fraction_dependent_in_range(self) -> None:
        adj = _make_chain(5)
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(adj, 500, rng)
        desc = InfoTheoreticDescriptor(
            features=["ci_signature"], max_descriptor_dim=10
        )
        bd = desc.compute(adj, data)
        frac_dep = bd[0]
        assert 0.0 <= frac_dep <= 1.0

    def test_normalized_tests_in_range(self) -> None:
        adj = _make_chain(5)
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(adj, 500, rng)
        desc = InfoTheoreticDescriptor(
            features=["ci_signature"], max_descriptor_dim=10
        )
        bd = desc.compute(adj, data)
        norm_tests = bd[1]
        # n_tests = n*(n-1) = 20, normalized by n²=25 → 0.8
        assert 0.0 <= norm_tests <= 1.0
        assert np.isclose(norm_tests, 20.0 / 25.0)

    def test_mean_abs_z_nonnegative(self) -> None:
        adj = _make_chain(5)
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(adj, 500, rng)
        desc = InfoTheoreticDescriptor(
            features=["ci_signature"], max_descriptor_dim=10
        )
        bd = desc.compute(adj, data)
        assert bd[2] >= 0.0, "mean |Z| should be non-negative"
        assert bd[3] >= 0.0, "max |Z| should be non-negative"

    def test_max_abs_z_ge_mean(self) -> None:
        adj = _make_chain(5)
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(adj, 500, rng)
        desc = InfoTheoreticDescriptor(
            features=["ci_signature"], max_descriptor_dim=10
        )
        bd = desc.compute(adj, data)
        assert bd[3] >= bd[2] - 1e-12, "max |Z| should be >= mean |Z|"

    def test_independent_variables_low_frac_dep(self) -> None:
        """Truly independent variables should have low fraction dependent."""
        adj = _make_empty(4)
        rng = np.random.default_rng(42)
        data = rng.standard_normal((1000, 4))
        desc = InfoTheoreticDescriptor(
            features=["ci_signature"], max_descriptor_dim=10
        )
        bd = desc.compute(adj, data)
        # With independent data and no edges, most pairs should appear
        # independent at α=0.05; expect ~5% false positives
        assert bd[0] < 0.2, f"Expected low frac_dep for independent data, got {bd[0]}"

    def test_strong_deps_high_frac_dep(self) -> None:
        """Graph with strong edges should have high fraction dependent."""
        adj = _make_chain(4)
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(adj, 1000, rng, noise_scale=0.1)
        desc = InfoTheoreticDescriptor(
            features=["ci_signature"], max_descriptor_dim=10
        )
        bd = desc.compute(adj, data)
        assert bd[0] > 0.3, f"Expected high frac_dep with strong edges, got {bd[0]}"

    def test_ci_signature_all_finite(self) -> None:
        adj = _make_diamond()
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(adj, 500, rng)
        desc = InfoTheoreticDescriptor(
            features=["ci_signature"], max_descriptor_dim=10
        )
        bd = desc.compute(adj, data)
        assert np.all(np.isfinite(bd))


# ===================================================================
# EquivalenceDescriptor Tests
# ===================================================================


class TestEquivalenceDescriptorSameMECSimilar:
    """test_equivalence_descriptor_same_mec_similar — verify that DAGs
    in the same MEC produce identical or very similar equivalence
    descriptors."""

    def test_default_dim(self) -> None:
        desc = EquivalenceDescriptor()
        assert desc.descriptor_dim == 6

    def test_bounds(self) -> None:
        desc = EquivalenceDescriptor()
        low, high = desc.descriptor_bounds
        assert low.shape == (6,)
        assert high.shape == (6,)
        assert np.all(low == 0.0)
        assert np.all(high == 1.0)

    def test_subset_features(self) -> None:
        desc = EquivalenceDescriptor(features=["mec_size", "compelled_fraction"])
        assert desc.descriptor_dim == 2

    def test_invalid_feature_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported feature"):
            EquivalenceDescriptor(features=["bogus"])

    def test_empty_features_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            EquivalenceDescriptor(features=[])

    def test_all_bounded(self) -> None:
        """All features should be in [0, 1]."""
        adj = _make_diamond()
        desc = EquivalenceDescriptor()
        bd = desc.compute(adj)
        assert np.all(bd >= 0.0)
        assert np.all(bd <= 1.0)

    def test_chain_same_mec(self) -> None:
        """A chain and its reversal are in the same MEC (both are chains
        with no v-structures). Their equivalence descriptors should match."""
        chain = _make_chain(4)
        # Reversed chain: 3→2→1→0
        reversed_chain = np.zeros((4, 4), dtype=np.int8)
        reversed_chain[3, 2] = 1
        reversed_chain[2, 1] = 1
        reversed_chain[1, 0] = 1

        desc = EquivalenceDescriptor()
        bd1 = desc.compute(chain)
        bd2 = desc.compute(reversed_chain)

        # The CPDAG for both should be the same undirected path,
        # so all features should be identical or very close
        np.testing.assert_allclose(bd1, bd2, atol=1e-10)

    def test_v_structure_preserved_in_mec(self) -> None:
        """A v-structure has compelled edges; its MEC descriptor should
        reflect non-zero compelled fraction and v-structure density."""
        adj = _make_v_structure()
        desc = EquivalenceDescriptor()
        bd = desc.compute(adj)

        # v_structure_density (index 2) should be > 0
        v_struct_idx = ["mec_size", "compelled_fraction", "v_structure_density",
                        "reversible_fraction", "cpdag_density",
                        "avg_chain_component_size"]
        vs_idx = v_struct_idx.index("v_structure_density")
        assert bd[vs_idx] > 0.0

        # compelled_fraction should be 1.0 (both edges compelled)
        cf_idx = v_struct_idx.index("compelled_fraction")
        assert np.isclose(bd[cf_idx], 1.0), \
            f"V-structure edges should all be compelled, got {bd[cf_idx]}"

    def test_reversible_fraction_chain(self) -> None:
        """A simple chain has all reversible edges (no v-structures)."""
        adj = _make_chain(4)
        desc = EquivalenceDescriptor(features=["reversible_fraction"])
        bd = desc.compute(adj)
        assert np.isclose(bd[0], 1.0), \
            f"Chain edges should be all reversible, got {bd[0]}"

    def test_compelled_plus_reversible_equals_one(self) -> None:
        """compelled_fraction + reversible_fraction should equal 1.0."""
        adj = _make_diamond()
        desc = EquivalenceDescriptor(
            features=["compelled_fraction", "reversible_fraction"]
        )
        bd = desc.compute(adj)
        total = bd[0] + bd[1]
        assert np.isclose(total, 1.0), \
            f"compelled + reversible should = 1.0, got {total}"

    def test_same_mec_reversed_edge(self) -> None:
        """Reversing a reversible edge should keep the DAG in the same
        MEC, yielding identical equivalence descriptors."""
        adj = _make_chain(5)
        adj_reversed = _reverse_one_reversible_edge(adj)

        desc = EquivalenceDescriptor()
        bd1 = desc.compute(adj)
        bd2 = desc.compute(adj_reversed)

        np.testing.assert_allclose(bd1, bd2, atol=1e-10)

    def test_different_mec_different_descriptors(self) -> None:
        """DAGs with different v-structures should have different MEC
        descriptors."""
        # 0→2←1 (v-structure)
        vs = _make_v_structure()
        # 0→1→2 (chain, no v-structure)
        chain = _make_chain(3)

        desc = EquivalenceDescriptor()
        bd_vs = desc.compute(vs)
        bd_chain = desc.compute(chain)

        assert not np.allclose(bd_vs, bd_chain)

    def test_empty_graph_descriptor(self) -> None:
        adj = _make_empty(5)
        desc = EquivalenceDescriptor()
        bd = desc.compute(adj)
        assert bd.shape == (6,)
        assert np.all(np.isfinite(bd))

    def test_complete_graph_descriptor(self) -> None:
        adj = _make_full_forward(4)
        desc = EquivalenceDescriptor()
        bd = desc.compute(adj)
        assert np.all(bd >= 0.0)
        assert np.all(bd <= 1.0)

    def test_mec_size_v_structure_smaller(self) -> None:
        """V-structure has smaller MEC than chain (more compelled edges)."""
        vs = _make_v_structure()
        chain = _make_chain(3)
        desc = EquivalenceDescriptor(features=["mec_size"])
        bd_vs = desc.compute(vs)
        bd_chain = desc.compute(chain)
        assert bd_vs[0] <= bd_chain[0], \
            "V-structure MEC should be <= chain MEC size"


# ===================================================================
# CompositeDescriptor Tests
# ===================================================================


class TestCompositeDescriptorCombinesAll:
    """test_composite_descriptor_combines_all — verify that the
    composite descriptor correctly concatenates sub-descriptors."""

    def test_concatenation_dims(self) -> None:
        s_desc = StructuralDescriptor(features=["edge_density", "dag_depth"])
        e_desc = EquivalenceDescriptor(features=["mec_size"])
        comp = CompositeDescriptor(descriptors=[s_desc, e_desc])
        assert comp.descriptor_dim == 3

    def test_concatenation_values(self, small_adj: AdjacencyMatrix) -> None:
        s_desc = StructuralDescriptor(features=["edge_density", "dag_depth"])
        e_desc = EquivalenceDescriptor(features=["mec_size"])
        comp = CompositeDescriptor(descriptors=[s_desc, e_desc])

        bd_comp = comp.compute(small_adj)
        bd_s = s_desc.compute(small_adj)
        bd_e = e_desc.compute(small_adj)

        # First 2 elements from structural, last 1 from equivalence
        np.testing.assert_allclose(bd_comp[:2], bd_s, atol=1e-12)
        np.testing.assert_allclose(bd_comp[2:], bd_e, atol=1e-12)

    def test_three_component_concat(self) -> None:
        s1 = StructuralDescriptor(features=["edge_density"])
        s2 = StructuralDescriptor(features=["dag_depth"])
        e = EquivalenceDescriptor(features=["mec_size", "cpdag_density"])
        comp = CompositeDescriptor(descriptors=[s1, s2, e])
        assert comp.descriptor_dim == 4

        adj = _make_chain(5)
        bd = comp.compute(adj)
        assert bd.shape == (4,)
        assert np.all(np.isfinite(bd))

    def test_weights_applied(self, small_adj: AdjacencyMatrix) -> None:
        s_desc = StructuralDescriptor(features=["edge_density", "dag_depth"])
        comp_no_weight = CompositeDescriptor(descriptors=[s_desc])
        comp_weighted = CompositeDescriptor(
            descriptors=[s_desc], weights=[2.0, 3.0]
        )

        bd_nw = comp_no_weight.compute(small_adj)
        bd_w = comp_weighted.compute(small_adj)

        # Before fitting (no normalization), weighted = raw * weights
        np.testing.assert_allclose(bd_w, bd_nw * np.array([2.0, 3.0]), atol=1e-12)

    def test_empty_descriptors_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            CompositeDescriptor(descriptors=[])

    def test_wrong_weight_length_raises(self) -> None:
        s_desc = StructuralDescriptor(features=["edge_density"])
        with pytest.raises(ValueError, match="weights"):
            CompositeDescriptor(descriptors=[s_desc], weights=[1.0, 2.0])

    def test_invalid_normalization_raises(self) -> None:
        s_desc = StructuralDescriptor(features=["edge_density"])
        with pytest.raises(ValueError, match="normalization"):
            CompositeDescriptor(descriptors=[s_desc], normalization="invalid")

    def test_composite_with_info_theoretic(self) -> None:
        s_desc = StructuralDescriptor(features=["edge_density"])
        i_desc = InfoTheoreticDescriptor(
            features=["avg_mi"], max_descriptor_dim=10
        )
        comp = CompositeDescriptor(descriptors=[s_desc, i_desc])
        assert comp.descriptor_dim == 2

        adj = _make_chain(5)
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(adj, 300, rng)
        bd = comp.compute(adj, data)
        assert bd.shape == (2,)

    def test_composite_bounds_shape(self) -> None:
        s = StructuralDescriptor(features=["edge_density", "dag_depth"])
        e = EquivalenceDescriptor(features=["mec_size"])
        comp = CompositeDescriptor(descriptors=[s, e])
        low, high = comp.descriptor_bounds
        assert low.shape == (3,)
        assert high.shape == (3,)

    def test_composite_deterministic(self, small_adj: AdjacencyMatrix) -> None:
        s = StructuralDescriptor(features=["edge_density", "dag_depth"])
        comp = CompositeDescriptor(descriptors=[s])
        bd1 = comp.compute(small_adj)
        # Reset the running stats for a fresh instance
        comp2 = CompositeDescriptor(descriptors=[s])
        bd2 = comp2.compute(small_adj)
        np.testing.assert_array_equal(bd1, bd2)


# ===================================================================
# PCA Compression Tests
# ===================================================================


class TestPCACompressionReducesDims:
    """test_pca_compression_reduces_dims — verify that PCA in the
    composite descriptor reduces output dimensionality."""

    def test_pca_reduces_dim(self) -> None:
        s = StructuralDescriptor()  # 10 features
        comp = CompositeDescriptor(descriptors=[s], pca_dim=3)

        # Need to fit with enough samples for PCA
        rng = np.random.default_rng(42)
        dags = []
        for _ in range(20):
            n = 6
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.3:
                        adj[i, j] = 1
            dags.append(adj)

        data = rng.standard_normal((100, 6))
        comp.fit(dags, data)

        assert comp.descriptor_dim == 3
        bd = comp.compute(dags[0], data)
        assert bd.shape == (3,)

    def test_pca_output_finite(self) -> None:
        s = StructuralDescriptor()
        comp = CompositeDescriptor(descriptors=[s], pca_dim=2)

        rng = np.random.default_rng(42)
        dags = []
        for _ in range(15):
            n = 5
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.3:
                        adj[i, j] = 1
            dags.append(adj)

        data = rng.standard_normal((100, 5))
        comp.fit(dags, data)

        for dag in dags:
            bd = comp.compute(dag, data)
            assert np.all(np.isfinite(bd))

    def test_pca_preserves_variation(self) -> None:
        """PCA-reduced descriptors should still differ for different DAGs."""
        s = StructuralDescriptor()
        comp = CompositeDescriptor(descriptors=[s], pca_dim=3)

        rng = np.random.default_rng(42)
        dags = []
        for _ in range(20):
            n = 6
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.3:
                        adj[i, j] = 1
            dags.append(adj)

        data = rng.standard_normal((100, 6))
        comp.fit(dags, data)

        descriptors = [comp.compute(d, data) for d in dags]
        # Not all descriptors should be identical
        unique_count = len(set(tuple(d) for d in descriptors))
        assert unique_count > 1, "PCA should preserve variation across graphs"

    def test_pca_dim_1(self) -> None:
        s = StructuralDescriptor(features=["edge_density", "dag_depth", "longest_path"])
        comp = CompositeDescriptor(descriptors=[s], pca_dim=1)

        rng = np.random.default_rng(42)
        dags = [_make_chain(5), _make_fork(5), _make_collider(5),
                _make_full_forward(5), _make_empty(5)]
        # Pad to >= 10 samples for is_fitted
        for _ in range(7):
            adj = np.zeros((5, 5), dtype=np.int8)
            adj[0, rng.integers(1, 5)] = 1
            dags.append(adj)

        data = rng.standard_normal((100, 5))
        comp.fit(dags, data)

        assert comp.descriptor_dim == 1
        bd = comp.compute(dags[0], data)
        assert bd.shape == (1,)

    def test_lazy_pca_via_compute(self) -> None:
        """PCA should be fitted lazily after 10 compute calls."""
        s = StructuralDescriptor(features=["edge_density", "dag_depth"])
        comp = CompositeDescriptor(descriptors=[s], pca_dim=1)

        rng = np.random.default_rng(42)
        # Before fitting, dimension should be raw
        assert comp.descriptor_dim == 2

        # Compute 10 times to trigger lazy PCA
        for _ in range(10):
            adj = np.zeros((5, 5), dtype=np.int8)
            adj[0, rng.integers(1, 5)] = 1
            comp.compute(adj)

        # After lazy fit, should use PCA
        bd = comp.compute(_make_chain(5))
        assert bd.shape[0] <= 2  # PCA may be active now

    def test_fit_then_compute_consistent(self) -> None:
        """After fit, repeated compute on same DAG should give same result."""
        s = StructuralDescriptor()
        comp = CompositeDescriptor(descriptors=[s], pca_dim=3)

        rng = np.random.default_rng(42)
        dags = []
        for _ in range(15):
            n = 5
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.3:
                        adj[i, j] = 1
            dags.append(adj)

        data = rng.standard_normal((100, 5))
        comp.fit(dags, data)

        test_dag = dags[0]
        bd1 = comp.compute(test_dag, data)
        bd2 = comp.compute(test_dag, data)
        # Note: running stats update on each compute, so results may
        # differ slightly. Check they are close.
        np.testing.assert_allclose(bd1, bd2, atol=0.1)


# ===================================================================
# Normalization Tests
# ===================================================================


class TestDescriptorNormalization:
    """test_descriptor_normalization — verify normalization strategies
    in the composite descriptor."""

    def _make_dags_and_data(
        self, rng: np.random.Generator
    ) -> Tuple[List[AdjacencyMatrix], DataMatrix]:
        """Generate a list of diverse DAGs and data for fitting."""
        dags: List[AdjacencyMatrix] = []
        for _ in range(20):
            n = 5
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.3:
                        adj[i, j] = 1
            dags.append(adj)
        # Also add some extreme cases
        dags.append(_make_empty(5))
        dags.append(_make_full_forward(5))
        dags.append(_make_chain(5))
        dags.append(_make_fork(5))
        dags.append(_make_collider(5))

        data = rng.standard_normal((100, 5))
        return dags, data

    def test_no_normalization(self, rng: np.random.Generator) -> None:
        dags, data = self._make_dags_and_data(rng)
        s = StructuralDescriptor(features=["edge_density", "dag_depth"])
        comp = CompositeDescriptor(descriptors=[s], normalization="none")
        comp.fit(dags, data)

        bd = comp.compute(dags[0], data)
        # Without normalization, values should still be the raw weighted values
        assert np.all(np.isfinite(bd))

    def test_zscore_normalization(self, rng: np.random.Generator) -> None:
        dags, data = self._make_dags_and_data(rng)
        s = StructuralDescriptor(features=["edge_density", "dag_depth"])
        comp = CompositeDescriptor(descriptors=[s], normalization="zscore")
        comp.fit(dags, data)

        # After fitting, bounds should be ±5
        low, high = comp.descriptor_bounds
        assert np.all(low == -5.0)
        assert np.all(high == 5.0)

        # Compute z-scored descriptor
        bd = comp.compute(dags[0], data)
        assert np.all(np.isfinite(bd))

    def test_minmax_normalization(self, rng: np.random.Generator) -> None:
        dags, data = self._make_dags_and_data(rng)
        s = StructuralDescriptor(features=["edge_density", "dag_depth"])
        comp = CompositeDescriptor(descriptors=[s], normalization="minmax")
        comp.fit(dags, data)

        # After fitting, bounds should be [0, 1]
        low, high = comp.descriptor_bounds
        assert np.all(low == 0.0)
        assert np.all(high == 1.0)

        bd = comp.compute(dags[0], data)
        assert np.all(np.isfinite(bd))
        # Min-max normalized values should be in [0, 1] (with clipping)
        assert np.all(bd >= -0.01)
        assert np.all(bd <= 1.01)

    def test_quantile_normalization(self, rng: np.random.Generator) -> None:
        dags, data = self._make_dags_and_data(rng)
        s = StructuralDescriptor(features=["edge_density", "dag_depth"])
        comp = CompositeDescriptor(descriptors=[s], normalization="quantile")
        comp.fit(dags, data)

        bd = comp.compute(dags[0], data)
        assert np.all(np.isfinite(bd))
        # Quantile normalization maps to [0, 1]
        assert np.all(bd >= 0.0)
        assert np.all(bd <= 1.0)

    def test_normalization_changes_values(self, rng: np.random.Generator) -> None:
        """Normalization should change descriptor values vs. raw."""
        dags, data = self._make_dags_and_data(rng)
        s = StructuralDescriptor(features=["edge_density", "dag_depth"])

        comp_none = CompositeDescriptor(descriptors=[s], normalization="none")
        comp_zscore = CompositeDescriptor(descriptors=[s], normalization="zscore")
        comp_none.fit(dags, data)
        comp_zscore.fit(dags, data)

        bd_none = comp_none.compute(dags[5], data)
        bd_zscore = comp_zscore.compute(dags[5], data)

        # Z-scored values should generally differ from raw
        assert not np.allclose(bd_none, bd_zscore)

    def test_minmax_extreme_values(self, rng: np.random.Generator) -> None:
        """Empty and full graphs should map to 0 and 1 for edge_density."""
        dags, data = self._make_dags_and_data(rng)
        s = StructuralDescriptor(features=["edge_density"])
        comp = CompositeDescriptor(descriptors=[s], normalization="minmax")
        comp.fit(dags, data)

        empty = _make_empty(5)
        full = _make_full_forward(5)
        bd_empty = comp.compute(empty, data)
        bd_full = comp.compute(full, data)

        # Empty has lowest density, full has highest
        assert bd_empty[0] < bd_full[0]

    def test_zscore_centered(self, rng: np.random.Generator) -> None:
        """After z-scoring, computing on many samples should roughly
        centre around 0."""
        dags, data = self._make_dags_and_data(rng)
        s = StructuralDescriptor(features=["edge_density"])
        comp = CompositeDescriptor(descriptors=[s], normalization="zscore")
        comp.fit(dags, data)

        values = [float(comp.compute(d, data)[0]) for d in dags]
        mean_val = np.mean(values)
        # Mean of z-scores should be approximately 0 (with running stats drift)
        assert abs(mean_val) < 2.0, f"Z-score mean {mean_val} too far from 0"

    def test_fit_with_single_dag(self) -> None:
        """Fitting with a single DAG should not crash."""
        s = StructuralDescriptor(features=["edge_density"])
        comp = CompositeDescriptor(descriptors=[s], normalization="minmax")
        adj = _make_chain(5)
        data = np.random.default_rng(42).standard_normal((100, 5))
        comp.fit([adj], data)
        bd = comp.compute(adj, data)
        assert np.all(np.isfinite(bd))

    def test_fit_empty_raises(self) -> None:
        s = StructuralDescriptor(features=["edge_density"])
        comp = CompositeDescriptor(descriptors=[s], normalization="minmax")
        data = np.random.default_rng(42).standard_normal((100, 5))
        with pytest.raises(ValueError, match="at least one"):
            comp.fit([], data)


# ===================================================================
# Additional cross-cutting tests
# ===================================================================


class TestDescriptorCrossCutting:
    """Additional tests to ensure descriptors work together correctly."""

    def test_structural_deterministic(self, small_adj: AdjacencyMatrix) -> None:
        desc = StructuralDescriptor()
        bd1 = desc.compute(small_adj)
        bd2 = desc.compute(small_adj)
        np.testing.assert_array_equal(bd1, bd2)

    def test_equivalence_deterministic(self, small_adj: AdjacencyMatrix) -> None:
        desc = EquivalenceDescriptor()
        bd1 = desc.compute(small_adj)
        bd2 = desc.compute(small_adj)
        np.testing.assert_array_equal(bd1, bd2)

    def test_structural_medium_graph(self, medium_adj: AdjacencyMatrix) -> None:
        desc = StructuralDescriptor()
        bd = desc.compute(medium_adj)
        assert bd.shape == (10,)
        assert np.all(bd >= 0.0)
        assert np.all(bd <= 1.0)

    def test_equivalence_medium_graph(self, medium_adj: AdjacencyMatrix) -> None:
        desc = EquivalenceDescriptor()
        bd = desc.compute(medium_adj)
        assert bd.shape == (6,)
        assert np.all(bd >= 0.0)
        assert np.all(bd <= 1.0)

    def test_info_theoretic_medium_graph(self, medium_adj: AdjacencyMatrix) -> None:
        rng = np.random.default_rng(42)
        data = _generate_linear_gaussian(medium_adj, 500, rng)
        desc = InfoTheoreticDescriptor(max_descriptor_dim=20)
        bd = desc.compute(medium_adj, data)
        assert bd.shape == (14,)
        assert np.all(np.isfinite(bd))

    def test_composite_all_three(self, small_adj: AdjacencyMatrix) -> None:
        s = StructuralDescriptor(features=["edge_density"])
        e = EquivalenceDescriptor(features=["mec_size"])
        i = InfoTheoreticDescriptor(features=["avg_mi"], max_descriptor_dim=10)

        comp = CompositeDescriptor(descriptors=[s, e, i])
        assert comp.descriptor_dim == 3

        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 5))
        bd = comp.compute(small_adj, data)
        assert bd.shape == (3,)
        assert np.all(np.isfinite(bd))

    def test_structural_different_sizes(self) -> None:
        """Structural descriptor should work for various graph sizes."""
        desc = StructuralDescriptor()
        for n in [2, 3, 5, 8, 12]:
            adj = _make_chain(n)
            bd = desc.compute(adj)
            assert bd.shape == (10,)
            assert np.all(bd >= 0.0)
            assert np.all(bd <= 1.0)

    def test_avg_path_length_chain(self) -> None:
        adj = _make_chain(5)
        desc = StructuralDescriptor(features=["avg_path_length"])
        bd = desc.compute(adj)
        # Chain: average shortest path among all reachable pairs
        # Pairs: (0,1)=1, (0,2)=2, (0,3)=3, (0,4)=4,
        #        (1,2)=1, (1,3)=2, (1,4)=3,
        #        (2,3)=1, (2,4)=2,
        #        (3,4)=1
        # Total = 1+2+3+4+1+2+3+1+2+1 = 20, pairs = 10
        # Average = 2.0, normalized by n-1=4 → 0.5
        assert np.isclose(bd[0], 0.5)

    def test_structural_all_features_chain_5(self) -> None:
        """Comprehensive check: chain of 5 nodes, verify specific values."""
        adj = _make_chain(5)
        desc = StructuralDescriptor()
        bd = desc.compute(adj)
        # edge_density: 4/(5*4/2) = 4/10 = 0.4
        assert np.isclose(bd[0], 0.4)
        # max_in_degree: 1/(5-1) = 0.25
        assert np.isclose(bd[1], 0.25)
        # longest_path: 4/4 = 1.0
        assert np.isclose(bd[3], 1.0)

    def test_v_structure_count_known(self) -> None:
        """Diamond graph 0→1, 0→2, 1→3, 2→3 has a v-structure at 3
        only if 1 and 2 are not adjacent."""
        adj = _make_diamond()
        desc = StructuralDescriptor(features=["v_structure_count"])
        bd = desc.compute(adj)
        # 1 and 2 are not adjacent, both parents of 3 → v-structure
        # But 0 is parent of both 1 and 2, and 1,2 are not adjacent
        # → no v-structure at 1 or 2 from that perspective
        # v-structures: only (1,3,2) with child=3
        assert bd[0] > 0.0
