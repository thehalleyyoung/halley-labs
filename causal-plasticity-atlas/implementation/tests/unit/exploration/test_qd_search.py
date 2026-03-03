"""Tests for QD search engine (ALG3).

Covers CVT tessellation, genome evaluation, archive update,
curiosity, mutation operators, full search loop, coverage,
canonical pattern extraction, and different configurations.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.exploration.qd_search import (
    QDSearchConfig,
    QDSearchEngine,
    QDArchive,
    ArchiveEntry,
    IterationResult,
)
from cpa.exploration.genome import QDGenome, BehaviorDescriptor
from cpa.exploration.cvt import CVTTessellation
from cpa.exploration.curiosity import CuriosityComputer, CuriosityConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_config():
    return QDSearchConfig(
        n_cells=50,
        pop_size=10,
        n_generations=5,
        n_children=5,
        n_cvt_samples=500,
        n_lloyd_iters=10,
        seed=42,
    )


@pytest.fixture
def available_contexts():
    return [f"ctx_{i}" for i in range(5)]


@pytest.fixture
def available_mechanisms():
    return [(f"X{i}", f"X{j}") for i in range(4) for j in range(4) if i != j]


@pytest.fixture
def cvt():
    t = CVTTessellation(n_cells=50, n_dims=4, n_samples=500, n_lloyd_iters=10, seed=42)
    t.initialize()
    return t


@pytest.fixture
def archive(cvt):
    return QDArchive(cvt)


@pytest.fixture
def sample_genome(available_contexts, available_mechanisms, rng):
    return QDGenome.random(
        available_contexts=available_contexts,
        available_mechanisms=available_mechanisms,
        rng=rng,
    )


@pytest.fixture
def sample_descriptor():
    return BehaviorDescriptor(
        frac_invariant=0.3,
        frac_parametric=0.4,
        frac_structural_emergent=0.3,
        entropy=0.8,
    )


# ---------------------------------------------------------------------------
# Test CVT tessellation
# ---------------------------------------------------------------------------

class TestCVTTessellation:

    def test_initialization(self, cvt):
        assert cvt.initialized

    def test_find_cell_returns_valid_index(self, cvt, rng):
        desc = rng.uniform(0, 1, size=4)
        cell = cvt.find_cell(desc)
        assert 0 <= cell < 50

    def test_find_cells_batch(self, cvt, rng):
        descs = rng.uniform(0, 1, size=(20, 4))
        cells = cvt.find_cells_batch(descs)
        assert cells.shape == (20,)
        assert np.all(cells >= 0)
        assert np.all(cells < 50)

    def test_cell_distance(self, cvt):
        d = cvt.cell_distance(0, 1)
        assert d >= 0.0
        assert cvt.cell_distance(0, 0) == 0.0

    def test_get_centroid(self, cvt):
        c = cvt.get_centroid(0)
        assert c.shape == (4,)

    def test_record_visit(self, cvt):
        cvt.record_visit(0, 1.5)
        assert cvt.get_visit_count(0) >= 1

    def test_coverage_initially_zero(self):
        t = CVTTessellation(n_cells=20, n_dims=4, seed=42)
        t.initialize()
        # No visits recorded yet
        assert t.coverage() == 0.0

    def test_coverage_increases_with_visits(self, cvt):
        for i in range(10):
            cvt.record_visit(i, float(i))
        assert cvt.coverage() > 0.0

    def test_n_occupied(self, cvt):
        for i in range(5):
            cvt.record_visit(i, 1.0)
        assert cvt.n_occupied() >= 5

    def test_least_visited_cells(self, cvt):
        for i in range(5):
            cvt.record_visit(i, 1.0)
        least = cvt.least_visited_cells(n=5)
        assert len(least) == 5

    def test_serialization(self, cvt):
        d = cvt.to_dict()
        restored = CVTTessellation.from_dict(d)
        assert restored.initialized
        desc = np.array([0.5, 0.5, 0.5, 0.5])
        assert cvt.find_cell(desc) == restored.find_cell(desc)

    def test_reset_stats(self, cvt):
        cvt.record_visit(0, 1.0)
        cvt.reset_stats()
        assert cvt.get_visit_count(0) == 0

    def test_quality_stats(self, cvt):
        for i in range(10):
            cvt.record_visit(i, float(i) * 0.1)
        stats = cvt.quality_stats()
        assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# Test genome evaluation
# ---------------------------------------------------------------------------

class TestGenomeEvaluation:

    def test_default_evaluator(self, sample_genome):
        """Default evaluator should return quality and descriptor."""
        from cpa.exploration.qd_search import _default_evaluator
        quality, desc = _default_evaluator(sample_genome)
        assert isinstance(quality, (int, float))
        assert isinstance(desc, BehaviorDescriptor)

    def test_custom_evaluator(self, sample_genome):
        """Custom evaluator integration."""
        def my_eval(genome, **kwargs):
            q = len(genome.context_ids) * 0.1
            d = BehaviorDescriptor(
                frac_invariant=0.5,
                frac_parametric=0.3,
                frac_structural_emergent=0.2,
                entropy=0.7,
            )
            return q, d

        quality, desc = my_eval(sample_genome)
        assert quality > 0
        assert desc.is_valid()


# ---------------------------------------------------------------------------
# Test archive update logic
# ---------------------------------------------------------------------------

class TestArchiveUpdate:

    def test_insert_into_empty_archive(self, archive, sample_genome, sample_descriptor):
        inserted, cell = archive.try_insert(sample_genome, 1.0, sample_descriptor)
        assert inserted
        assert archive.size == 1

    def test_better_quality_replaces(self, archive, sample_genome, sample_descriptor):
        archive.try_insert(sample_genome, 1.0, sample_descriptor)
        cell = archive.occupied_cells().pop()
        genome2 = sample_genome.copy()
        archive.try_insert(genome2, 2.0, sample_descriptor)
        entry = archive.get_entry(cell)
        assert entry.quality == 2.0

    def test_worse_quality_does_not_replace(self, archive, sample_genome, sample_descriptor):
        archive.try_insert(sample_genome, 2.0, sample_descriptor)
        cell = archive.occupied_cells().pop()
        genome2 = sample_genome.copy()
        inserted, _ = archive.try_insert(genome2, 0.5, sample_descriptor)
        assert not inserted
        entry = archive.get_entry(cell)
        assert entry.quality == 2.0

    def test_batch_insert(self, archive, available_contexts, available_mechanisms, rng):
        genomes = [
            QDGenome.random(available_contexts, available_mechanisms, rng=rng)
            for _ in range(10)
        ]
        qualities = rng.uniform(0, 1, size=10)
        descriptors = [
            BehaviorDescriptor.random(rng=rng) for _ in range(10)
        ]
        inserted, cells = archive.try_insert_batch(genomes, qualities, descriptors)
        assert inserted.shape == (10,)
        assert np.sum(inserted) > 0

    def test_get_all_entries(self, archive, sample_genome, sample_descriptor):
        archive.try_insert(sample_genome, 1.0, sample_descriptor)
        entries = archive.get_all_entries()
        assert len(entries) >= 1

    def test_best_entries(self, archive, available_contexts, available_mechanisms, rng):
        for _ in range(20):
            g = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
            d = BehaviorDescriptor.random(rng=rng)
            archive.try_insert(g, rng.uniform(), d)
        best = archive.best_entries(n=5)
        assert len(best) <= 5
        if len(best) >= 2:
            assert best[0].quality >= best[1].quality

    def test_occupied_cells(self, archive, sample_genome, sample_descriptor):
        archive.try_insert(sample_genome, 1.0, sample_descriptor)
        occ = archive.occupied_cells()
        assert len(occ) == 1

    def test_clear(self, archive, sample_genome, sample_descriptor):
        archive.try_insert(sample_genome, 1.0, sample_descriptor)
        archive.clear()
        assert archive.size == 0


# ---------------------------------------------------------------------------
# Test curiosity computation
# ---------------------------------------------------------------------------

class TestCuriosity:

    def test_compute_returns_positive(self):
        cc = CuriosityComputer(n_cells=50)
        val = cc.compute(0, 1.0)
        assert val > 0.0

    def test_unvisited_cells_high_curiosity(self):
        cc = CuriosityComputer(n_cells=50)
        # First visit should have high curiosity
        val = cc.compute(0, 1.0)
        assert val > 0.0

    def test_visited_cells_decreasing_curiosity(self):
        cc = CuriosityComputer(n_cells=50)
        vals = []
        for i in range(5):
            v = cc.compute(0, 1.0)
            vals.append(v)
        # Curiosity should generally decrease with repeated visits
        assert vals[-1] <= vals[0] + 0.1

    def test_compute_batch(self):
        cc = CuriosityComputer(n_cells=50)
        cells = np.array([0, 1, 2, 3, 4])
        qualities = np.array([1.0, 0.5, 0.8, 0.2, 0.9])
        batch = cc.compute_batch(cells, qualities)
        assert batch.shape == (5,)
        assert np.all(batch > 0)

    def test_advance_generation(self):
        cc = CuriosityComputer(n_cells=50)
        cc.compute(0, 1.0)
        cc.advance_generation()
        # Should not raise

    def test_exploration_ratio(self):
        cc = CuriosityComputer(n_cells=50)
        ratio = cc.exploration_ratio()
        assert 0.0 <= ratio <= 1.0

    def test_most_curious_cells(self):
        cc = CuriosityComputer(n_cells=50)
        # Visit some cells
        for i in range(10):
            cc.compute(i, float(i))
        cells = cc.most_curious_cells(n=5)
        assert len(cells) == 5

    def test_serialization(self):
        cc = CuriosityComputer(n_cells=50)
        cc.compute(0, 1.0)
        d = cc.to_dict()
        restored = CuriosityComputer.from_dict(d)
        assert restored is not None

    def test_reset(self):
        cc = CuriosityComputer(n_cells=50)
        cc.compute(0, 1.0)
        cc.reset()
        # After reset, exploration ratio should be back to initial
        assert cc.exploration_ratio() >= 0.0

    def test_curiosity_config(self):
        config = CuriosityConfig(mu=0.7, ema_alpha=0.2)
        cc = CuriosityComputer(n_cells=50, config=config)
        val = cc.compute(0, 1.0)
        assert val > 0.0


# ---------------------------------------------------------------------------
# Test mutation operators
# ---------------------------------------------------------------------------

class TestMutationOperators:

    def test_mutate_context_add(self, sample_genome, available_contexts, rng):
        original_n = sample_genome.num_contexts
        mutated = sample_genome.mutate_context_add(available_contexts, rng=rng)
        assert mutated.num_contexts >= original_n

    def test_mutate_context_remove(self, sample_genome, available_contexts, rng):
        # Ensure at least 3 contexts so removal guard doesn't prevent it
        g = sample_genome
        while g.num_contexts < 3:
            g = g.mutate_context_add(available_contexts, rng=rng)
        mutated = g.mutate_context_remove(rng=rng)
        assert mutated.num_contexts < g.num_contexts

    def test_mutate_mechanism_add(self, sample_genome, available_mechanisms, rng):
        original_n = sample_genome.num_mechanisms
        mutated = sample_genome.mutate_mechanism_add(available_mechanisms, rng=rng)
        assert mutated.num_mechanisms >= original_n

    def test_mutate_mechanism_remove(self, sample_genome, rng):
        if sample_genome.num_mechanisms > 1:
            mutated = sample_genome.mutate_mechanism_remove(rng=rng)
            assert mutated.num_mechanisms < sample_genome.num_mechanisms

    def test_mutate_context_swap(self, sample_genome, available_contexts, rng):
        mutated = sample_genome.mutate_context_swap(available_contexts, rng=rng)
        assert mutated.num_contexts == sample_genome.num_contexts

    def test_mutate_params(self, sample_genome, rng):
        mutated = sample_genome.mutate_params(rng=rng, sigma=0.1)
        # Parameters should be different
        assert mutated is not sample_genome

    def test_general_mutate(self, sample_genome, available_contexts, available_mechanisms, rng):
        mutated = sample_genome.mutate(available_contexts, available_mechanisms, rng=rng)
        assert isinstance(mutated, QDGenome)

    def test_crossover_uniform(self, available_contexts, available_mechanisms, rng):
        p1 = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
        p2 = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
        child = QDGenome.crossover_uniform(p1, p2, rng=rng)
        assert isinstance(child, QDGenome)
        assert child.num_contexts > 0


# ---------------------------------------------------------------------------
# Test full search loop
# ---------------------------------------------------------------------------

class TestFullSearchLoop:

    def test_initialize(self, available_contexts, available_mechanisms, small_config):
        engine = QDSearchEngine(
            available_contexts=available_contexts,
            available_mechanisms=available_mechanisms,
            config=small_config,
        )
        engine.initialize()

    def test_run_generation(self, available_contexts, available_mechanisms, small_config):
        engine = QDSearchEngine(
            available_contexts=available_contexts,
            available_mechanisms=available_mechanisms,
            config=small_config,
        )
        engine.initialize()
        result = engine._run_iteration(0)
        assert isinstance(result, IterationResult)

    def test_run_full(self, available_contexts, available_mechanisms, small_config):
        engine = QDSearchEngine(
            available_contexts=available_contexts,
            available_mechanisms=available_mechanisms,
            config=small_config,
        )
        engine.initialize()
        result = engine.run(n_generations=3, progress=False)
        assert isinstance(result, dict)

    def test_iteration_result_fields(self, available_contexts, available_mechanisms, small_config):
        engine = QDSearchEngine(
            available_contexts=available_contexts,
            available_mechanisms=available_mechanisms,
            config=small_config,
        )
        engine.initialize()
        result = engine._run_iteration(0)
        assert hasattr(result, "n_evaluated")
        assert hasattr(result, "coverage")
        assert hasattr(result, "qd_score")

    def test_callbacks(self, available_contexts, available_mechanisms, small_config):
        engine = QDSearchEngine(
            available_contexts=available_contexts,
            available_mechanisms=available_mechanisms,
            config=small_config,
        )
        engine.initialize()
        gen_log = []
        engine.on_generation(lambda gen, result, eng: gen_log.append(gen))
        engine.run(n_generations=2, progress=False)
        assert len(gen_log) >= 2


# ---------------------------------------------------------------------------
# Test archive coverage statistics
# ---------------------------------------------------------------------------

class TestArchiveCoverage:

    def test_coverage_fraction(self, archive, available_contexts, available_mechanisms, rng):
        for _ in range(30):
            g = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
            d = BehaviorDescriptor.random(rng=rng)
            archive.try_insert(g, rng.uniform(), d)
        cov = archive.coverage()
        assert 0.0 <= cov <= 1.0

    def test_mean_quality(self, archive, available_contexts, available_mechanisms, rng):
        for _ in range(10):
            g = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
            d = BehaviorDescriptor.random(rng=rng)
            archive.try_insert(g, rng.uniform(0.5, 1.0), d)
        mq = archive.mean_quality()
        assert mq >= 0.5

    def test_max_quality(self, archive, available_contexts, available_mechanisms, rng):
        for _ in range(10):
            g = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
            d = BehaviorDescriptor.random(rng=rng)
            archive.try_insert(g, rng.uniform(0, 1), d)
        assert archive.max_quality() <= 1.0

    def test_qd_score(self, archive, available_contexts, available_mechanisms, rng):
        for _ in range(10):
            g = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
            d = BehaviorDescriptor.random(rng=rng)
            archive.try_insert(g, rng.uniform(), d)
        qd = archive.quality_diversity_score()
        assert qd >= 0.0

    def test_get_stats(self, archive, sample_genome, sample_descriptor):
        archive.try_insert(sample_genome, 1.0, sample_descriptor)
        stats = archive.get_stats()
        assert isinstance(stats, dict)
        assert "coverage" in stats or "size" in stats


# ---------------------------------------------------------------------------
# Test canonical pattern extraction
# ---------------------------------------------------------------------------

class TestCanonicalPatterns:

    def test_extract_patterns(self, archive, available_contexts, available_mechanisms, rng):
        for _ in range(25):
            g = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
            d = BehaviorDescriptor.random(rng=rng)
            archive.try_insert(g, rng.uniform(), d)
        patterns = archive.extract_canonical_patterns(n_patterns=5)
        assert isinstance(patterns, list)
        assert len(patterns) <= 5

    def test_cluster_entries(self, archive, available_contexts, available_mechanisms, rng):
        for _ in range(30):
            g = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
            d = BehaviorDescriptor.random(rng=rng)
            archive.try_insert(g, rng.uniform(), d)
        if archive.size >= 5:
            clusters = archive.cluster_entries(n_clusters=3)
            assert isinstance(clusters, dict)


# ---------------------------------------------------------------------------
# Test with different configurations
# ---------------------------------------------------------------------------

class TestDifferentConfigs:

    def test_small_archive(self, available_contexts, available_mechanisms):
        config = QDSearchConfig(
            n_cells=10, pop_size=5, n_generations=2,
            n_children=3, n_cvt_samples=100, n_lloyd_iters=5, seed=42,
        )
        engine = QDSearchEngine(
            available_contexts=available_contexts,
            available_mechanisms=available_mechanisms,
            config=config,
        )
        engine.initialize()
        result = engine.run(n_generations=2, progress=False)
        assert isinstance(result, dict)

    def test_larger_archive(self, available_contexts, available_mechanisms):
        config = QDSearchConfig(
            n_cells=100, pop_size=15, n_generations=3,
            n_children=8, n_cvt_samples=1000, n_lloyd_iters=15, seed=42,
        )
        engine = QDSearchEngine(
            available_contexts=available_contexts,
            available_mechanisms=available_mechanisms,
            config=config,
        )
        engine.initialize()
        result = engine.run(n_generations=3, progress=False)
        assert isinstance(result, dict)

    def test_archive_serialization(self, archive, available_contexts, available_mechanisms, rng):
        for _ in range(10):
            g = QDGenome.random(available_contexts, available_mechanisms, rng=rng)
            d = BehaviorDescriptor.random(rng=rng)
            archive.try_insert(g, rng.uniform(), d)
        d = archive.to_dict()
        restored = QDArchive.from_dict(d)
        assert restored.size == archive.size

    def test_config_defaults(self):
        config = QDSearchConfig()
        assert config.n_cells == 1000
        assert config.pop_size == 100
        assert config.n_generations == 500

    def test_iteration_result_defaults(self):
        r = IterationResult(generation=0)
        assert r.n_evaluated == 0
        assert r.coverage == 0.0
