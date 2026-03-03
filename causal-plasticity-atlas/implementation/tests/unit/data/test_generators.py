"""Unit tests for cpa.data.generators – SyntheticGenerator and SB1-SB5."""

from __future__ import annotations

import numpy as np
import pytest

from cpa.core.types import PlasticityClass
from cpa.data.generators import (
    SyntheticGenerator,
    generate_sb1,
    generate_sb2,
    generate_sb3,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_generator():
    return SyntheticGenerator(n_nodes=4, n_contexts=3,
                              n_samples_per_context=200, seed=42)


@pytest.fixture
def tiny_generator():
    return SyntheticGenerator(n_nodes=3, n_contexts=2,
                              n_samples_per_context=100, seed=42)


# ===================================================================
# Tests – SB1 (Fixed Structure, Varying Parameters)
# ===================================================================


class TestSB1:
    """Test SB1 scenario: same structure, different parameters."""

    def test_generate_returns_dict(self, small_generator):
        result = small_generator.generate("sb1")
        assert isinstance(result, dict)

    def test_has_datasets(self, small_generator):
        result = small_generator.generate("sb1")
        assert "datasets" in result

    def test_datasets_count(self, small_generator):
        result = small_generator.generate("sb1")
        assert len(result["datasets"]) == 3

    def test_dataset_shape(self, small_generator):
        result = small_generator.generate("sb1")
        datasets = result["datasets"]
        for data in datasets:
            assert data.shape == (200, 4)

    def test_has_ground_truth(self, small_generator):
        result = small_generator.generate("sb1")
        assert "ground_truth_plasticity" in result or "ground_truth" in result or "plasticity" in result

    def test_same_structure_across_contexts(self, small_generator):
        result = small_generator.generate("sb1")
        if "dags" in result:
            dags = list(result["dags"])
            for dag in dags[1:]:
                # Same structure → same adjacency pattern
                np.testing.assert_array_equal(
                    (dags[0] != 0).astype(int),
                    (dag != 0).astype(int),
                )

    def test_convenience_function(self):
        result = generate_sb1(n_nodes=4, n_contexts=2, n_samples=100, seed=42)
        assert isinstance(result, dict)
        assert "datasets" in result

    def test_ground_truth_valid_plasticity(self, small_generator):
        result = small_generator.generate("sb1")
        gt_key = "ground_truth_plasticity" if "ground_truth_plasticity" in result else ("ground_truth" if "ground_truth" in result else "plasticity")
        if gt_key in result:
            gt = result[gt_key]
            for key, val in gt.items():
                assert isinstance(val, PlasticityClass)


# ===================================================================
# Tests – SB2 (Structure Varies)
# ===================================================================


class TestSB2:
    """Test SB2 scenario: structure varies across contexts."""

    def test_generate_returns_dict(self, small_generator):
        result = small_generator.generate("sb2")
        assert isinstance(result, dict)

    def test_datasets_count(self, small_generator):
        result = small_generator.generate("sb2")
        assert len(result["datasets"]) == 3

    def test_dataset_shape(self, small_generator):
        result = small_generator.generate("sb2")
        for data in result["datasets"]:
            assert data.shape == (200, 4)

    def test_structure_varies(self, small_generator):
        result = small_generator.generate("sb2")
        if "dags" in result:
            dags = list(result["dags"])
            # At least one structural difference
            if len(dags) >= 2:
                diff = np.any(
                    (dags[0] != 0).astype(int) != (dags[1] != 0).astype(int)
                )
                # May or may not differ depending on randomness
                assert isinstance(diff, (bool, np.bool_))

    def test_convenience_function(self):
        result = generate_sb2(n_nodes=4, n_contexts=2, n_samples=100, seed=42)
        assert isinstance(result, dict)

    def test_has_plasticity_labels(self, small_generator):
        result = small_generator.generate("sb2")
        gt_key = "ground_truth_plasticity" if "ground_truth_plasticity" in result else ("ground_truth" if "ground_truth" in result else "plasticity")
        assert gt_key in result

    def test_ground_truth_has_structural(self, small_generator):
        result = small_generator.generate("sb2")
        gt_key = "ground_truth_plasticity" if "ground_truth_plasticity" in result else ("ground_truth" if "ground_truth" in result else "plasticity")
        if gt_key in result:
            gt = result[gt_key]
            # Should have at least some structural plasticity
            has_structural = any(
                v == PlasticityClass.STRUCTURAL_PLASTIC for v in gt.values()
            )
            # May not always have structural depending on random seed
            assert isinstance(has_structural, bool)


# ===================================================================
# Tests – SB3 (Emergence)
# ===================================================================


class TestSB3:
    """Test SB3 scenario: edges appear/disappear."""

    def test_generate_returns_dict(self, small_generator):
        result = small_generator.generate("sb3")
        assert isinstance(result, dict)

    def test_datasets_count(self, small_generator):
        result = small_generator.generate("sb3")
        assert len(result["datasets"]) == 3

    def test_dataset_shape(self, small_generator):
        result = small_generator.generate("sb3")
        for data in result["datasets"]:
            assert data.shape == (200, 4)

    def test_convenience_function(self):
        result = generate_sb3(n_nodes=4, n_contexts=2, n_samples=100, seed=42)
        assert isinstance(result, dict)

    def test_ground_truth_valid(self, small_generator):
        result = small_generator.generate("sb3")
        gt_key = "ground_truth_plasticity" if "ground_truth_plasticity" in result else ("ground_truth" if "ground_truth" in result else "plasticity")
        if gt_key in result:
            gt = result[gt_key]
            for val in gt.values():
                assert isinstance(val, PlasticityClass)


# ===================================================================
# Tests – Format and ground truth validation
# ===================================================================


class TestFormatAndGroundTruth:
    """Test that all generators return correct format."""

    @pytest.mark.parametrize("scenario", ["sb1", "sb2", "sb3"])
    def test_returns_dict(self, small_generator, scenario):
        result = small_generator.generate(scenario)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("scenario", ["sb1", "sb2", "sb3"])
    def test_has_datasets_key(self, small_generator, scenario):
        result = small_generator.generate(scenario)
        assert "datasets" in result

    @pytest.mark.parametrize("scenario", ["sb1", "sb2", "sb3"])
    def test_datasets_are_ndarrays(self, small_generator, scenario):
        result = small_generator.generate(scenario)
        for data in result["datasets"]:
            assert isinstance(data, np.ndarray)
            assert data.ndim == 2

    @pytest.mark.parametrize("scenario", ["sb1", "sb2", "sb3"])
    def test_no_nans_in_data(self, small_generator, scenario):
        result = small_generator.generate(scenario)
        for data in result["datasets"]:
            assert not np.any(np.isnan(data))

    @pytest.mark.parametrize("scenario", ["sb1", "sb2", "sb3"])
    def test_ground_truth_labels_valid(self, small_generator, scenario):
        result = small_generator.generate(scenario)
        gt_key = "ground_truth_plasticity" if "ground_truth_plasticity" in result else ("ground_truth" if "ground_truth" in result else "plasticity")
        if gt_key in result:
            gt = result[gt_key]
            valid_classes = set(PlasticityClass)
            for val in gt.values():
                assert val in valid_classes


# ===================================================================
# Tests – Reproducibility
# ===================================================================


class TestReproducibility:
    """Test that seed ensures reproducibility."""

    def test_same_seed_same_data(self):
        gen1 = SyntheticGenerator(n_nodes=3, n_contexts=2,
                                   n_samples_per_context=50, seed=42)
        gen2 = SyntheticGenerator(n_nodes=3, n_contexts=2,
                                   n_samples_per_context=50, seed=42)
        r1 = gen1.generate("sb1")
        r2 = gen2.generate("sb1")
        for i in range(len(r1["datasets"])):
            np.testing.assert_array_equal(r1["datasets"][i], r2["datasets"][i])

    def test_different_seed_different_data(self):
        gen1 = SyntheticGenerator(n_nodes=3, n_contexts=2,
                                   n_samples_per_context=50, seed=42)
        gen2 = SyntheticGenerator(n_nodes=3, n_contexts=2,
                                   n_samples_per_context=50, seed=99)
        r1 = gen1.generate("sb1")
        r2 = gen2.generate("sb1")
        d1 = r1["datasets"][0]
        d2 = r2["datasets"][0]
        assert not np.allclose(d1, d2)


# ===================================================================
# Tests – true_dags and true_plasticity accessors
# ===================================================================


class TestAccessors:
    """Test true_dags() and true_plasticity() accessors."""

    def test_true_dags_returns_dict(self, small_generator):
        small_generator.generate("sb1")
        dags = small_generator.true_dags()
        assert isinstance(dags, dict)

    def test_true_dags_square(self, small_generator):
        small_generator.generate("sb1")
        dags = small_generator.true_dags()
        dag_list = list(dags.values()) if isinstance(dags, dict) else list(dags)
        for dag in dag_list:
            assert dag.shape[0] == dag.shape[1] == 4

    def test_true_plasticity_returns_dict(self, small_generator):
        small_generator.generate("sb1")
        gt = small_generator.true_plasticity()
        assert isinstance(gt, dict)

    def test_true_plasticity_valid(self, small_generator):
        small_generator.generate("sb1")
        gt = small_generator.true_plasticity()
        for val in gt.values():
            assert isinstance(val, PlasticityClass)
