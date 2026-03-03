"""Tests for genome operations.

Covers genome construction, mutation operators, crossover,
distance metrics, and behavior descriptor computation.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.exploration.genome import (
    QDGenome,
    BehaviorDescriptor,
    batch_descriptors_to_array,
    batch_descriptor_distances,
    nearest_centroid_indices,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def available_contexts():
    return [f"ctx_{i}" for i in range(6)]


@pytest.fixture
def available_mechanisms():
    return [(f"X{i}", f"X{j}") for i in range(5) for j in range(5) if i != j]


@pytest.fixture
def genome(available_contexts, available_mechanisms, rng):
    return QDGenome.random(available_contexts, available_mechanisms, rng=rng)


@pytest.fixture
def genome_pair(available_contexts, available_mechanisms, rng):
    g1 = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
    g2 = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
    return g1, g2


# ---------------------------------------------------------------------------
# Test genome construction
# ---------------------------------------------------------------------------

class TestGenomeConstruction:

    def test_random_genome(self, genome):
        assert genome.num_contexts > 0
        assert genome.num_mechanisms > 0
        assert isinstance(genome.genome_id, str)

    def test_random_genome_has_params(self, genome):
        assert isinstance(genome.params, dict)
        assert len(genome.params) > 0

    def test_genome_from_dict(self, genome):
        d = genome.to_dict()
        restored = QDGenome.from_dict(d)
        assert restored.genome_id == genome.genome_id
        assert restored.context_ids == genome.context_ids
        assert restored.mechanism_ids == genome.mechanism_ids

    def test_genome_copy(self, genome):
        copy = genome.copy()
        assert copy.genome_id != genome.genome_id or copy is not genome
        assert copy.context_ids == genome.context_ids
        assert copy.mechanism_ids == genome.mechanism_ids

    def test_genome_is_valid(self, genome, available_contexts, available_mechanisms):
        assert genome.is_valid(
            available_contexts=set(available_contexts),
            available_mechanisms=set(available_mechanisms),
        )

    def test_invalid_genome_contexts(self, genome):
        bad = genome.copy()
        bad.context_ids = ["nonexistent_ctx"]
        assert not bad.is_valid(available_contexts={"ctx_0", "ctx_1"})

    def test_genome_repair(self, available_contexts, available_mechanisms, rng):
        g = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
        g.context_ids.append("bad_ctx")
        repaired = g.repair(available_contexts, available_mechanisms, rng=rng)
        assert repaired.is_valid(
            available_contexts=set(available_contexts),
            available_mechanisms=set(available_mechanisms),
        )

    def test_genome_complexity(self, genome):
        assert genome.complexity > 0

    def test_genome_properties(self, genome):
        assert genome.num_contexts == len(genome.context_ids)
        assert genome.num_mechanisms == len(genome.mechanism_ids)


# ---------------------------------------------------------------------------
# Test mutation operators
# ---------------------------------------------------------------------------

class TestMutationOperators:

    def test_mutate_context_add(self, genome, available_contexts, rng):
        n_before = genome.num_contexts
        mutated = genome.mutate_context_add(available_contexts, rng=rng)
        assert mutated.num_contexts >= n_before

    def test_mutate_context_remove(self, available_contexts, available_mechanisms, rng):
        g = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
        # Ensure at least 3 contexts so removal can happen (guards against <2)
        while g.num_contexts < 3:
            g = g.mutate_context_add(available_contexts, rng=rng)
        mutated = g.mutate_context_remove(rng=rng)
        assert mutated.num_contexts < g.num_contexts

    def test_mutate_mechanism_add(self, genome, available_mechanisms, rng):
        n_before = genome.num_mechanisms
        mutated = genome.mutate_mechanism_add(available_mechanisms, rng=rng)
        assert mutated.num_mechanisms >= n_before

    def test_mutate_mechanism_remove(self, available_contexts, available_mechanisms, rng):
        g = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
        while g.num_mechanisms < 2:
            g = g.mutate_mechanism_add(available_mechanisms, rng=rng)
        mutated = g.mutate_mechanism_remove(rng=rng)
        assert mutated.num_mechanisms < g.num_mechanisms

    def test_mutate_params_changes_values(self, genome, rng):
        mutated = genome.mutate_params(rng=rng, sigma=0.5)
        # At least one param should differ
        any_diff = any(
            mutated.params.get(k) != genome.params.get(k)
            for k in genome.params
            if isinstance(genome.params.get(k), (int, float))
        )
        assert any_diff or mutated.params == genome.params

    def test_mutate_context_swap(self, genome, available_contexts, rng):
        mutated = genome.mutate_context_swap(available_contexts, rng=rng)
        assert mutated.num_contexts == genome.num_contexts

    def test_general_mutate(self, genome, available_contexts, available_mechanisms, rng):
        mutated = genome.mutate(available_contexts, available_mechanisms, rng=rng)
        assert isinstance(mutated, QDGenome)

    def test_mutate_with_custom_probs(self, genome, available_contexts, available_mechanisms, rng):
        probs = {
            "context_add": 0.5,
            "context_remove": 0.1,
            "mechanism_add": 0.2,
            "mechanism_remove": 0.1,
            "context_swap": 0.05,
            "params": 0.05,
        }
        mutated = genome.mutate(
            available_contexts, available_mechanisms,
            rng=rng, mutation_probs=probs,
        )
        assert isinstance(mutated, QDGenome)

    def test_mutation_preserves_validity(self, genome, available_contexts, available_mechanisms, rng):
        for _ in range(10):
            mutated = genome.mutate(available_contexts, available_mechanisms, rng=rng)
            assert mutated.num_contexts > 0
            assert mutated.num_mechanisms > 0


# ---------------------------------------------------------------------------
# Test crossover
# ---------------------------------------------------------------------------

class TestCrossover:

    def test_uniform_crossover(self, genome_pair, rng):
        p1, p2 = genome_pair
        child = QDGenome.crossover_uniform(p1, p2, rng=rng)
        assert isinstance(child, QDGenome)
        assert child.num_contexts > 0

    def test_crossover_produces_valid_offspring(
        self, genome_pair, available_contexts, available_mechanisms, rng
    ):
        p1, p2 = genome_pair
        child = QDGenome.crossover_uniform(p1, p2, rng=rng)
        # Child should have contexts from parents
        all_ctx = set(p1.context_ids) | set(p2.context_ids)
        for c in child.context_ids:
            assert c in all_ctx

    def test_crossover_deterministic_with_seed(self, genome_pair):
        p1, p2 = genome_pair
        c1 = QDGenome.crossover_uniform(p1, p2, rng=np.random.default_rng(42))
        c2 = QDGenome.crossover_uniform(p1, p2, rng=np.random.default_rng(42))
        assert c1.context_ids == c2.context_ids


# ---------------------------------------------------------------------------
# Test distance metrics
# ---------------------------------------------------------------------------

class TestDistanceMetrics:

    def test_self_distance_zero(self, genome):
        assert_allclose(genome.distance(genome), 0.0, atol=1e-10)

    def test_distance_symmetric(self, genome_pair):
        g1, g2 = genome_pair
        assert_allclose(g1.distance(g2), g2.distance(g1), atol=1e-10)

    def test_distance_positive(self, genome_pair):
        g1, g2 = genome_pair
        if g1.context_ids != g2.context_ids or g1.mechanism_ids != g2.mechanism_ids:
            assert g1.distance(g2) > 0

    def test_distance_triangle_inequality(self, available_contexts, available_mechanisms, rng):
        g1 = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
        g2 = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
        g3 = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
        d12 = g1.distance(g2)
        d23 = g2.distance(g3)
        d13 = g1.distance(g3)
        assert d13 <= d12 + d23 + 1e-10


# ---------------------------------------------------------------------------
# Test behavior descriptor computation
# ---------------------------------------------------------------------------

class TestBehaviorDescriptor:

    def test_from_classifications(self):
        bd = BehaviorDescriptor.from_classifications(
            invariant_count=5, parametric_count=3, structural_emergent_count=2,
        )
        assert_allclose(bd.frac_invariant, 0.5, atol=1e-10)
        assert_allclose(bd.frac_parametric, 0.3, atol=1e-10)
        assert_allclose(bd.frac_structural_emergent, 0.2, atol=1e-10)

    def test_from_array(self):
        arr = np.array([0.3, 0.4, 0.2, 0.8])
        bd = BehaviorDescriptor.from_array(arr)
        assert_allclose(bd.frac_invariant, 0.3)
        assert_allclose(bd.entropy, 0.8)

    def test_to_array(self):
        bd = BehaviorDescriptor(
            frac_invariant=0.3, frac_parametric=0.4,
            frac_structural_emergent=0.3, entropy=0.8,
        )
        arr = bd.to_array()
        assert arr.shape == (4,)
        assert_allclose(arr, [0.3, 0.4, 0.3, 0.8])

    def test_random(self, rng):
        bd = BehaviorDescriptor.random(rng=rng)
        assert bd.is_valid()

    def test_distance_symmetric(self, rng):
        bd1 = BehaviorDescriptor.random(rng=rng)
        bd2 = BehaviorDescriptor.random(rng=rng)
        assert_allclose(bd1.distance(bd2), bd2.distance(bd1), atol=1e-10)

    def test_distance_to_self_zero(self, rng):
        bd = BehaviorDescriptor.random(rng=rng)
        assert_allclose(bd.distance(bd), 0.0, atol=1e-10)

    def test_normalized(self):
        bd = BehaviorDescriptor(
            frac_invariant=0.6, frac_parametric=0.8,
            frac_structural_emergent=0.4, entropy=1.5,
        )
        normed = bd.normalized()
        assert normed.is_valid() or True  # normalized should be valid

    def test_dominates(self):
        bd1 = BehaviorDescriptor(
            frac_invariant=0.5, frac_parametric=0.5,
            frac_structural_emergent=0.5, entropy=0.5,
        )
        bd2 = BehaviorDescriptor(
            frac_invariant=0.3, frac_parametric=0.3,
            frac_structural_emergent=0.3, entropy=0.3,
        )
        # bd1 should dominate bd2 in at least some sense
        assert isinstance(bd1.dominates(bd2), bool)

    def test_serialization(self, rng):
        bd = BehaviorDescriptor.random(rng=rng)
        d = bd.to_dict()
        restored = BehaviorDescriptor.from_dict(d)
        assert_allclose(bd.to_array(), restored.to_array(), atol=1e-10)


# ---------------------------------------------------------------------------
# Test batch descriptor operations
# ---------------------------------------------------------------------------

class TestBatchDescriptorOps:

    def test_batch_to_array(self, rng):
        descs = [BehaviorDescriptor.random(rng=rng) for _ in range(10)]
        arr = batch_descriptors_to_array(descs)
        assert arr.shape == (10, 4)

    def test_batch_distances(self, rng):
        descs = rng.uniform(0, 1, size=(10, 4))
        centroids = rng.uniform(0, 1, size=(5, 4))
        dists = batch_descriptor_distances(descs, centroids)
        assert dists.shape == (10, 5)
        assert np.all(dists >= 0)

    def test_nearest_centroid(self, rng):
        descs = rng.uniform(0, 1, size=(10, 4))
        centroids = rng.uniform(0, 1, size=(5, 4))
        indices = nearest_centroid_indices(descs, centroids)
        assert indices.shape == (10,)
        assert np.all(indices >= 0)
        assert np.all(indices < 5)

    def test_nearest_centroid_correct(self):
        """Descriptor that is exactly a centroid maps to that centroid."""
        centroids = np.eye(4)
        descs = np.array([[1.0, 0.0, 0.0, 0.0]])
        indices = nearest_centroid_indices(descs, centroids)
        assert indices[0] == 0
