"""Tests for causal_qd.data module.

Covers LinearGaussianSCM, DataGenerator, DataLoader, and DataPreprocessor.
"""

from __future__ import annotations

import os
import tempfile
from typing import Tuple

import numpy as np
import numpy.testing as npt
import pytest

from causal_qd.core.dag import DAG
from causal_qd.data.scm import LinearGaussianSCM
from causal_qd.data.generator import DataGenerator, generate_random_scm, generate_from_known_structure
from causal_qd.data.loader import DataLoader, DataValidationResult, VariableType
from causal_qd.data.preprocessor import DataPreprocessor
from causal_qd.types import AdjacencyMatrix, DataMatrix


# ===================================================================
# Helpers
# ===================================================================

def _chain_dag(n: int) -> DAG:
    """Create a chain DAG: 0→1→…→(n-1)."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
    return DAG(adj)


def _fork_dag() -> DAG:
    """0→1, 0→2, 0→3."""
    adj = np.zeros((4, 4), dtype=np.int8)
    adj[0, 1] = 1
    adj[0, 2] = 1
    adj[0, 3] = 1
    return DAG(adj)


def _make_scm(dag: DAG, rng: np.random.Generator) -> LinearGaussianSCM:
    """Build a LinearGaussianSCM with known weights."""
    n = dag.n_nodes
    weights = np.zeros((n, n), dtype=np.float64)
    for i, j in dag.edges:
        weights[i, j] = rng.uniform(0.5, 1.0)
    noise_std = np.full(n, 0.5)
    return LinearGaussianSCM(dag=dag, weights=weights, noise_std=noise_std)


# ===================================================================
# LinearGaussianSCM
# ===================================================================

class TestLinearGaussianSCM:
    """Tests for causal_qd.data.scm.LinearGaussianSCM."""

    def test_init_stores_attributes(self):
        dag = _chain_dag(4)
        weights = np.zeros((4, 4), dtype=np.float64)
        weights[0, 1] = 0.8
        weights[1, 2] = 0.7
        weights[2, 3] = 0.6
        noise_std = np.array([1.0, 0.5, 0.5, 0.5])

        scm = LinearGaussianSCM(dag=dag, weights=weights, noise_std=noise_std)

        assert scm.n_nodes == 4
        npt.assert_array_equal(scm.weights, weights)
        npt.assert_array_equal(scm.noise_std, noise_std)
        assert scm.dag.n_nodes == 4

    def test_sample_correct_shape(self):
        dag = _chain_dag(5)
        scm = LinearGaussianSCM.from_dag(dag, rng=np.random.default_rng(0))
        data = scm.sample(200, rng=np.random.default_rng(1))
        assert data.shape == (200, 5)

    def test_sample_deterministic_with_seed(self):
        dag = _chain_dag(4)
        scm = LinearGaussianSCM.from_dag(dag, rng=np.random.default_rng(0))
        d1 = scm.sample(100, rng=np.random.default_rng(42))
        d2 = scm.sample(100, rng=np.random.default_rng(42))
        npt.assert_array_equal(d1, d2)

    def test_linear_gaussian_scm_generates_correct_covariance(self):
        """Empirical covariance should approximate the theoretical covariance
        for a linear Gaussian SCM when using enough samples."""
        dag = _chain_dag(3)
        weights = np.zeros((3, 3), dtype=np.float64)
        weights[0, 1] = 0.8
        weights[1, 2] = 0.7
        noise_std = np.array([1.0, 0.5, 0.5])
        scm = LinearGaussianSCM(dag=dag, weights=weights, noise_std=noise_std)

        n_samples = 50_000
        data = scm.sample(n_samples, rng=np.random.default_rng(123))
        emp_cov = np.cov(data, rowvar=False)

        # Theoretical: B = (I - W)^{-1}, Cov = B.T @ diag(sigma^2) @ B
        W = weights.astype(np.float64)
        B = np.linalg.inv(np.eye(3) - W)
        D = np.diag(noise_std ** 2)
        theo_cov = B.T @ D @ B

        npt.assert_allclose(emp_cov, theo_cov, atol=0.15)

    def test_intervene_changes_distribution(self):
        dag = _chain_dag(3)
        scm = LinearGaussianSCM.from_dag(dag, rng=np.random.default_rng(0))
        scm_do = scm.intervene(node=1, value=5.0)

        data = scm_do.sample(500, rng=np.random.default_rng(1))
        # All samples for node 1 should equal intervention value
        npt.assert_allclose(data[:, 1], 5.0, atol=1e-10)

    def test_from_dag_classmethod(self):
        dag = _chain_dag(6)
        scm = LinearGaussianSCM.from_dag(dag, rng=np.random.default_rng(0))
        assert scm.n_nodes == 6
        assert scm.weights.shape == (6, 6)
        assert scm.noise_std.shape == (6,)
        # Root node should have zero incoming weight
        npt.assert_equal(scm.weights[:, 0].sum(), 0.0)

    def test_from_dag_weight_range(self):
        dag = _chain_dag(4)
        scm = LinearGaussianSCM.from_dag(
            dag, weight_range=(0.5, 0.6), rng=np.random.default_rng(0)
        )
        edge_weights = scm.weights[scm.weights != 0]
        assert np.all(np.abs(edge_weights) >= 0.5)
        assert np.all(np.abs(edge_weights) <= 0.6)

    def test_n_nodes_property(self):
        for n in [3, 5, 10]:
            dag = _chain_dag(n)
            scm = LinearGaussianSCM.from_dag(dag, rng=np.random.default_rng(0))
            assert scm.n_nodes == n

    def test_sample_root_node_variance(self):
        """Root node variance should match noise_std^2."""
        dag = _chain_dag(3)
        noise_std = np.array([2.0, 0.5, 0.5])
        weights = np.zeros((3, 3), dtype=np.float64)
        weights[0, 1] = 0.5
        weights[1, 2] = 0.5
        scm = LinearGaussianSCM(dag=dag, weights=weights, noise_std=noise_std)

        data = scm.sample(20_000, rng=np.random.default_rng(0))
        root_var = np.var(data[:, 0])
        npt.assert_allclose(root_var, 4.0, atol=0.2)

    def test_sample_different_rng_gives_different_data(self):
        dag = _chain_dag(4)
        scm = LinearGaussianSCM.from_dag(dag, rng=np.random.default_rng(0))
        d1 = scm.sample(100, rng=np.random.default_rng(1))
        d2 = scm.sample(100, rng=np.random.default_rng(2))
        assert not np.allclose(d1, d2)


# ===================================================================
# DataGenerator
# ===================================================================

class TestDataGenerator:
    """Tests for causal_qd.data.generator.DataGenerator."""

    def test_synthetic_data_correct_dimensions(self):
        dag = _chain_dag(5)
        gen = DataGenerator()
        data = gen.generate(dag, n_samples=100, rng=np.random.default_rng(0))
        assert data.shape == (100, 5)

    def test_generate_with_weights(self):
        dag = _chain_dag(3)
        weights = np.zeros((3, 3), dtype=np.float64)
        weights[0, 1] = 0.9
        weights[1, 2] = 0.8
        noise_std = np.array([1.0, 0.5, 0.5])
        gen = DataGenerator()
        data = gen.generate_with_weights(
            dag, weights, noise_std, n_samples=200, rng=np.random.default_rng(0)
        )
        assert data.shape == (200, 3)
        assert np.isfinite(data).all()

    def test_generate_deterministic(self):
        dag = _chain_dag(4)
        gen = DataGenerator()
        d1 = gen.generate(dag, n_samples=100, rng=np.random.default_rng(42))
        d2 = gen.generate(dag, n_samples=100, rng=np.random.default_rng(42))
        npt.assert_array_equal(d1, d2)

    def test_generate_interventional(self):
        dag = _chain_dag(4)
        gen = DataGenerator()
        data = gen.generate_interventional(
            dag, target_node=1, intervention_value=10.0,
            n_samples=200, rng=np.random.default_rng(0),
        )
        assert data.shape == (200, 4)
        npt.assert_allclose(data[:, 1], 10.0, atol=1e-10)

    def test_generate_soft_intervention(self):
        dag = _chain_dag(4)
        gen = DataGenerator()
        data_obs = gen.generate(dag, n_samples=5000, rng=np.random.default_rng(0))
        data_int = gen.generate_soft_intervention(
            dag, target_node=1, shift=5.0,
            n_samples=5000, rng=np.random.default_rng(0),
        )
        assert data_int.shape == (5000, 4)
        # Intervened node mean should be shifted
        assert data_int[:, 1].mean() > data_obs[:, 1].mean() + 2.0

    def test_generate_random_scm(self):
        data, dag, weights = generate_random_scm(
            n_nodes=5, n_samples=200, edge_prob=0.3, rng=np.random.default_rng(0)
        )
        assert data.shape == (200, 5)
        assert dag.n_nodes == 5
        assert weights.shape == (5, 5)

    def test_generate_from_known_structure(self):
        dag = _chain_dag(4)
        data = generate_from_known_structure(
            dag, n_samples=300, rng=np.random.default_rng(0)
        )
        assert data.shape == (300, 4)

    def test_fork_structure_correlation(self):
        """Children of a fork root should be correlated through the parent."""
        dag = _fork_dag()
        gen = DataGenerator(weight_range=(0.8, 0.9))
        data = gen.generate(dag, n_samples=5000, rng=np.random.default_rng(0))
        corr = np.corrcoef(data[:, 1], data[:, 2])[0, 1]
        assert abs(corr) > 0.2, f"Expected non-trivial correlation, got {corr}"


# ===================================================================
# DataLoader
# ===================================================================

class TestDataLoader:
    """Tests for causal_qd.data.loader.DataLoader."""

    def test_data_loader_csv(self, tmp_path):
        """Load a CSV and check dimensions / contents."""
        rng = np.random.default_rng(0)
        original = rng.standard_normal((50, 4))
        csv_path = str(tmp_path / "data.csv")
        np.savetxt(csv_path, original, delimiter=",")
        loaded = DataLoader.load_csv(csv_path)
        assert loaded.shape == (50, 4)
        npt.assert_allclose(loaded, original, atol=1e-6)

    def test_data_loader_csv_with_header(self, tmp_path):
        csv_path = str(tmp_path / "header.csv")
        with open(csv_path, "w") as f:
            f.write("a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
        loaded = DataLoader.load_csv(csv_path, has_header=True)
        assert loaded.shape == (2, 3)
        npt.assert_allclose(loaded[0], [1.0, 2.0, 3.0])

    def test_data_loader_numpy(self, tmp_path):
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((30, 5))
        npy_path = str(tmp_path / "data.npy")
        np.save(npy_path, arr)
        loaded = DataLoader.load_numpy(npy_path)
        npt.assert_array_equal(loaded, arr)

    def test_validate_good_data(self):
        data = np.random.default_rng(0).standard_normal((100, 5))
        result = DataLoader.validate(data)
        assert isinstance(result, DataValidationResult)
        assert result.is_valid
        assert result.n_samples == 100
        assert result.n_variables == 5
        assert not result.has_missing
        assert result.missing_count == 0

    def test_validate_data_with_nans(self):
        data = np.random.default_rng(0).standard_normal((100, 5))
        data[10, 2] = np.nan
        data[20, 3] = np.nan
        result = DataLoader.validate(data)
        assert result.has_missing
        assert result.missing_count >= 2

    def test_validate_constant_columns(self):
        data = np.random.default_rng(0).standard_normal((100, 4))
        data[:, 2] = 5.0  # constant column
        result = DataLoader.validate(data)
        assert result.has_constant_columns
        assert 2 in result.constant_columns

    def test_infer_variable_types(self):
        rng = np.random.default_rng(0)
        data = np.zeros((100, 3))
        data[:, 0] = rng.standard_normal(100)      # continuous
        data[:, 1] = rng.choice([0.0, 1.0], 100)   # binary
        data[:, 2] = 5.0                             # constant
        types = DataLoader.infer_variable_types(data)
        assert types[0] == VariableType.CONTINUOUS
        assert types[1] == VariableType.BINARY
        assert types[2] == VariableType.CONSTANT

    def test_data_loader_parquet(self, tmp_path):
        """Load a Parquet file and check dimensions."""
        pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        import pandas as pd
        rng = np.random.default_rng(0)
        original = rng.standard_normal((50, 4))
        df = pd.DataFrame(original, columns=["a", "b", "c", "d"])
        parquet_path = str(tmp_path / "data.parquet")
        df.to_parquet(parquet_path, index=False)
        loaded = DataLoader.load_parquet(parquet_path)
        assert loaded.shape == (50, 4)
        npt.assert_allclose(loaded, original, atol=1e-10)


# ===================================================================
# DataPreprocessor
# ===================================================================

class TestDataPreprocessor:
    """Tests for causal_qd.data.preprocessor.DataPreprocessor."""

    def test_preprocessor_standardization(self):
        rng = np.random.default_rng(0)
        data = rng.normal(loc=5.0, scale=3.0, size=(500, 4))
        standardized = DataPreprocessor.standardize(data)

        assert standardized.shape == data.shape
        npt.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-10)
        npt.assert_allclose(standardized.std(axis=0), 1.0, atol=0.01)

    def test_preprocessor_discretization(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((200, 3))
        discretized = DataPreprocessor.discretize(data, n_bins=5, method="equal_frequency")

        assert discretized.shape == data.shape
        unique_vals = np.unique(discretized)
        assert len(unique_vals) <= 5
        assert discretized.dtype in (np.float64, np.int64, np.int32, float)

    def test_discretization_equal_width(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 10, size=(300, 2))
        discretized = DataPreprocessor.discretize(data, n_bins=4, method="equal_width")
        assert discretized.shape == data.shape
        unique_vals = np.unique(discretized)
        assert len(unique_vals) <= 4

    def test_min_max_scale(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(-10, 10, size=(100, 3))
        scaled = DataPreprocessor.min_max_scale(data)
        assert scaled.shape == data.shape
        npt.assert_allclose(scaled.min(axis=0), 0.0, atol=1e-10)
        npt.assert_allclose(scaled.max(axis=0), 1.0, atol=1e-10)

    def test_min_max_scale_custom_range(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(-10, 10, size=(100, 3))
        scaled = DataPreprocessor.min_max_scale(data, feature_range=(-1.0, 1.0))
        npt.assert_allclose(scaled.min(axis=0), -1.0, atol=1e-10)
        npt.assert_allclose(scaled.max(axis=0), 1.0, atol=1e-10)

    def test_robust_scale(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((200, 3))
        scaled = DataPreprocessor.robust_scale(data)
        assert scaled.shape == data.shape
        # Median should be approximately 0
        npt.assert_allclose(np.median(scaled, axis=0), 0.0, atol=0.05)

    def test_impute_missing_mean(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 4))
        col_means = data.mean(axis=0)
        data[5, 0] = np.nan
        data[10, 2] = np.nan
        imputed = DataPreprocessor.impute_missing(data, strategy="mean")
        assert not np.isnan(imputed).any()
        npt.assert_allclose(imputed[5, 0], col_means[0], atol=0.15)

    def test_impute_missing_median(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 3))
        data[0, 0] = np.nan
        imputed = DataPreprocessor.impute_missing(data, strategy="median")
        assert not np.isnan(imputed).any()

    def test_standardize_preserves_shape(self):
        shapes = [(50, 3), (100, 10), (1000, 2)]
        rng = np.random.default_rng(0)
        for shape in shapes:
            data = rng.standard_normal(shape)
            standardized = DataPreprocessor.standardize(data)
            assert standardized.shape == shape

    def test_remove_outliers(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((200, 3))
        data[0, 0] = 100.0  # extreme outlier
        cleaned = DataPreprocessor.remove_outliers(data, threshold=3.0, method="zscore")
        assert cleaned.shape[1] == 3
        assert cleaned.shape[0] < 200  # at least the outlier row removed

    def test_clip_outliers(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((200, 3))
        data[0, 0] = 100.0
        clipped = DataPreprocessor.clip_outliers(data, threshold=3.0)
        assert clipped.shape == data.shape
        assert clipped[0, 0] < 100.0  # outlier was clipped

    def test_remove_constant_columns(self):
        data = np.zeros((100, 5))
        data[:, 0] = np.arange(100)
        data[:, 1] = 7.0  # constant
        data[:, 2] = np.arange(100) * 2
        data[:, 3] = 7.0  # constant
        data[:, 4] = np.arange(100) * 3
        filtered, removed = DataPreprocessor.remove_constant_columns(data)
        assert filtered.shape == (100, 3)
        assert 1 in removed
        assert 3 in removed

    def test_detrend(self):
        rng = np.random.default_rng(0)
        t = np.arange(200, dtype=np.float64)
        data = np.column_stack([t * 2 + rng.standard_normal(200), rng.standard_normal(200)])
        detrended = DataPreprocessor.detrend(data)
        assert detrended.shape == data.shape
        # Linear trend should be removed — mean of detrended ≈ 0
        assert abs(np.polyfit(np.arange(200), detrended[:, 0], 1)[0]) < 0.5
