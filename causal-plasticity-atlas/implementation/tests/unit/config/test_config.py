"""Unit tests for cpa.config – ExperimentConfig, HyperparameterSpace, GridSearch, ComponentRegistry."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cpa.config.experiment import (
    ExperimentConfig,
    ExperimentRunner,
    ResultsTracker,
)
from cpa.config.hyperparameters import (
    HyperparameterSpace,
    GridSearch,
    RandomSearch,
    ParameterRange,
    HyperparameterConfig,
)
from cpa.config.registry import (
    ComponentRegistry,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_config():
    return ExperimentConfig(
        name="test_experiment",
        description="A test experiment",
        seed=42,
        n_contexts=3,
        n_nodes=5,
        n_samples=100,
        methods=["pc", "ges"],
        metrics=["shd", "f1"],
        output_dir="/tmp/cpa_test",
    )


@pytest.fixture
def config_dict():
    return {
        "name": "from_dict_exp",
        "description": "Test from dict",
        "seed": 123,
        "n_contexts": 4,
        "n_nodes": 10,
        "n_samples": 500,
        "methods": ["pc"],
        "metrics": ["shd"],
        "output_dir": "/tmp/cpa_dict",
    }


@pytest.fixture
def hyper_space():
    space = HyperparameterSpace()
    space.add_continuous("alpha", 0.01, 0.1, log_scale=False)
    space.add_continuous("learning_rate", 1e-4, 1e-1, log_scale=True)
    space.add_integer("max_depth", 1, 5)
    return space


@pytest.fixture
def registry():
    return ComponentRegistry()


# ===================================================================
# Tests – ExperimentConfig serialization
# ===================================================================


class TestExperimentConfigSerialization:
    """Test ExperimentConfig serialization round-trip."""

    def test_to_dict(self, simple_config):
        d = simple_config.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test_experiment"

    def test_from_dict_roundtrip(self, simple_config):
        d = simple_config.to_dict()
        restored = ExperimentConfig.from_dict(d)
        assert restored.name == simple_config.name
        assert restored.seed == simple_config.seed
        assert restored.n_contexts == simple_config.n_contexts

    def test_from_dict(self, config_dict):
        config = ExperimentConfig.from_dict(config_dict)
        assert config.name == "from_dict_exp"
        assert config.n_nodes == 10

    def test_roundtrip_preserves_all_fields(self, simple_config):
        d = simple_config.to_dict()
        restored = ExperimentConfig.from_dict(d)
        assert restored.n_nodes == simple_config.n_nodes
        assert restored.n_samples == simple_config.n_samples
        assert restored.methods == simple_config.methods
        assert restored.metrics == simple_config.metrics

    def test_validate(self, simple_config):
        errors = simple_config.validate()
        assert isinstance(errors, list)

    def test_validate_valid_config(self, simple_config):
        errors = simple_config.validate()
        assert len(errors) == 0

    def test_hash(self, simple_config):
        h = simple_config.hash()
        assert isinstance(h, str)
        assert len(h) > 0

    def test_with_overrides(self, simple_config):
        new_config = simple_config.with_overrides(n_nodes=20, seed=99)
        assert new_config.n_nodes == 20
        assert new_config.seed == 99
        # Original unchanged
        assert simple_config.n_nodes == 5


# ===================================================================
# Tests – ExperimentConfig validation
# ===================================================================


class TestExperimentConfigValidation:
    """Test config validation catches invalid configs."""

    def test_invalid_negative_samples(self):
        config = ExperimentConfig(
            name="bad", description="bad", seed=42,
            n_contexts=3, n_nodes=5, n_samples=-1,
            methods=["pc"], metrics=["shd"],
        )
        errors = config.validate()
        assert len(errors) > 0

    def test_invalid_zero_nodes(self):
        config = ExperimentConfig(
            name="bad", description="bad", seed=42,
            n_contexts=3, n_nodes=0, n_samples=100,
            methods=["pc"], metrics=["shd"],
        )
        errors = config.validate()
        assert len(errors) > 0


# ===================================================================
# Tests – ResultsTracker
# ===================================================================


class TestResultsTracker:
    """Test ResultsTracker logging."""

    def test_log_metric(self):
        tracker = ResultsTracker()
        tracker.log_metric("loss", 0.5, step=1)
        tracker.log_metric("loss", 0.3, step=2)
        metrics = tracker.get_metric("loss")
        assert len(metrics) == 2

    def test_log_artifact(self):
        tracker = ResultsTracker()
        tracker.log_artifact("model", {"weights": [1, 2, 3]})
        artifact = tracker.get_artifact("model")
        assert artifact == {"weights": [1, 2, 3]}

    def test_summary(self):
        tracker = ResultsTracker()
        tracker.log_metric("acc", 0.9)
        summary = tracker.summary()
        assert isinstance(summary, dict)


# ===================================================================
# Tests – HyperparameterSpace sampling
# ===================================================================


class TestHyperparameterSpace:
    """Test HyperparameterSpace sampling covers space."""

    def test_sample_returns_dict(self, hyper_space):
        rng = np.random.default_rng(42)
        sample = hyper_space.sample(rng)
        assert isinstance(sample, dict)

    def test_sample_has_all_params(self, hyper_space):
        rng = np.random.default_rng(42)
        sample = hyper_space.sample(rng)
        assert "alpha" in sample
        assert "learning_rate" in sample
        assert "max_depth" in sample

    def test_sample_within_bounds(self, hyper_space):
        rng = np.random.default_rng(42)
        for _ in range(50):
            sample = hyper_space.sample(rng)
            assert 0.01 <= sample["alpha"] <= 0.1
            assert 1e-4 <= sample["learning_rate"] <= 1e-1
            assert 1 <= sample["max_depth"] <= 5

    def test_grid_returns_list(self, hyper_space):
        grid = hyper_space.grid(n_per_dim=3)
        assert isinstance(grid, list)
        assert len(grid) > 0

    def test_grid_covers_space(self, hyper_space):
        grid = hyper_space.grid(n_per_dim=3)
        for params in grid:
            assert 0.01 <= params["alpha"] <= 0.1

    def test_dimensions(self, hyper_space):
        assert hyper_space.dimensions() == 3

    def test_add_discrete(self):
        space = HyperparameterSpace()
        space.add_discrete("batch_size", [16, 32, 64, 128])
        rng = np.random.default_rng(42)
        sample = space.sample(rng)
        assert sample["batch_size"] in [16, 32, 64, 128]

    def test_add_categorical(self):
        space = HyperparameterSpace()
        space.add_categorical("optimizer", ["adam", "sgd", "rmsprop"])
        rng = np.random.default_rng(42)
        sample = space.sample(rng)
        assert sample["optimizer"] in ["adam", "sgd", "rmsprop"]

    def test_many_samples_cover_range(self, hyper_space):
        rng = np.random.default_rng(42)
        alphas = [hyper_space.sample(rng)["alpha"] for _ in range(100)]
        assert min(alphas) < 0.03
        assert max(alphas) > 0.08


# ===================================================================
# Tests – GridSearch
# ===================================================================


class TestGridSearch:
    """Test GridSearch finds optimum for convex function."""

    def test_grid_search_convex(self):
        space = HyperparameterSpace()
        space.add_continuous("x", -5.0, 5.0)
        # Objective: minimize (x-1)^2 → optimum at x=1
        gs = GridSearch(space, objective_fn=lambda p: -(p["x"] - 1.0) ** 2)
        results = gs.search(n_per_dim=21)
        best = gs.best_params()
        assert_allclose(best["x"], 1.0, atol=0.5)

    def test_grid_search_returns_list(self):
        space = HyperparameterSpace()
        space.add_continuous("x", 0.0, 1.0)
        gs = GridSearch(space, objective_fn=lambda p: -p["x"] ** 2)
        results = gs.search(n_per_dim=5)
        assert isinstance(results, list)

    def test_best_params_2d(self):
        space = HyperparameterSpace()
        space.add_continuous("x", -2.0, 2.0)
        space.add_continuous("y", -2.0, 2.0)

        def obj(p):
            return -((p["x"] - 0.5) ** 2 + (p["y"] + 0.5) ** 2)

        gs = GridSearch(space, objective_fn=obj)
        gs.search(n_per_dim=11)
        best = gs.best_params()
        assert_allclose(best["x"], 0.5, atol=0.5)
        assert_allclose(best["y"], -0.5, atol=0.5)


# ===================================================================
# Tests – RandomSearch
# ===================================================================


class TestRandomSearch:
    """Test RandomSearch."""

    def test_search_returns_results(self):
        space = HyperparameterSpace()
        space.add_continuous("x", -5.0, 5.0)
        rs = RandomSearch(space, objective_fn=lambda p: -p["x"] ** 2,
                          n_trials=20)
        results = rs.search(seed=42)
        assert isinstance(results, list)
        assert len(results) == 20

    def test_top_k(self):
        space = HyperparameterSpace()
        space.add_continuous("x", -5.0, 5.0)
        rs = RandomSearch(space, objective_fn=lambda p: -p["x"] ** 2,
                          n_trials=50)
        rs.search(seed=42)
        top = rs.top_k(5)
        assert len(top) == 5


# ===================================================================
# Tests – ComponentRegistry
# ===================================================================


class TestComponentRegistry:
    """Test ComponentRegistry register/get/list."""

    def test_register_and_get(self, registry):
        registry.register("my_scorer", lambda: 42, category="scores")
        component = registry.get("my_scorer", category="scores")
        assert component() == 42

    def test_get_missing_raises(self, registry):
        with pytest.raises(Exception):
            registry.get("nonexistent", category="scores")

    def test_list_components(self, registry):
        registry.register("scorer_a", lambda: 1, category="scores")
        registry.register("scorer_b", lambda: 2, category="scores")
        components = registry.list_components(category="scores")
        assert "scorer_a" in components
        assert "scorer_b" in components

    def test_has(self, registry):
        registry.register("my_test", lambda: 0, category="tests")
        assert registry.has("my_test", category="tests")
        assert not registry.has("missing", category="tests")

    def test_unregister(self, registry):
        registry.register("temp", lambda: 0, category="temp")
        registry.unregister("temp", category="temp")
        assert not registry.has("temp", category="temp")

    def test_categories(self, registry):
        registry.register("a", lambda: 0, category="cat1")
        registry.register("b", lambda: 0, category="cat2")
        cats = registry.categories()
        assert "cat1" in cats
        assert "cat2" in cats

    def test_list_all_components(self, registry):
        registry.register("x", lambda: 0, category="cat1")
        registry.register("y", lambda: 0, category="cat2")
        all_comps = registry.list_components()
        assert isinstance(all_comps, list)

    def test_register_class(self, registry):
        class MyBaseline:
            name = "my_baseline"

        registry.register("my_baseline", MyBaseline, category="baselines")
        cls = registry.get("my_baseline", category="baselines")
        assert cls.name == "my_baseline"

    def test_overwrite_register(self, registry):
        registry.register("item", lambda: 1, category="test")
        registry.register("item", lambda: 2, category="test")
        component = registry.get("item", category="test")
        assert component() == 2
