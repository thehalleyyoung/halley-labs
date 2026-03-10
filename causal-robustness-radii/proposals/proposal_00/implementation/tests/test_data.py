"""Tests for causalcert.data – loading, parsing, preprocessing, synthetic."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causalcert.data.loader import load_csv, load_parquet, load_auto
from causalcert.data.dag_io import (
    load_dag,
    save_dag,
    parse_dot,
    format_dot,
    parse_json,
    format_json,
)
from causalcert.data.preprocessing import (
    standardize,
    encode_categorical,
    impute_missing,
    preprocess,
    select_numeric_columns,
)
from causalcert.data.synthetic import (
    generate_linear_gaussian,
    generate_nonlinear,
    generate_linear_gaussian_with_treatment,
    random_dag,
    compute_true_ate_linear,
)
from causalcert.data.types_inference import infer_variable_types, is_binary
from causalcert.data.validation import (
    validate_data,
    validate_dag_data_compatibility,
    detect_outliers,
)
from causalcert.types import AdjacencyMatrix, VariableType

# ── helpers ───────────────────────────────────────────────────────────────

def _adj(n: int, edges: list[tuple[int, int]]) -> AdjacencyMatrix:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


# ═══════════════════════════════════════════════════════════════════════════
# CSV / Parquet loading
# ═══════════════════════════════════════════════════════════════════════════


class TestCSVLoading:
    def test_load_csv_basic(self, sample_csv: Path) -> None:
        df = load_csv(str(sample_csv))
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 500
        assert df.shape[1] == 4

    def test_load_csv_columns(self, sample_csv: Path) -> None:
        df = load_csv(str(sample_csv), columns=["X0", "X1"])
        assert list(df.columns) == ["X0", "X1"]

    def test_load_csv_nrows(self, sample_csv: Path) -> None:
        df = load_csv(str(sample_csv), nrows=10)
        assert df.shape[0] == 10


class TestParquetLoading:
    def test_load_parquet_basic(self, sample_parquet: Path) -> None:
        df = load_parquet(str(sample_parquet))
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 500

    def test_load_parquet_columns(self, sample_parquet: Path) -> None:
        df = load_parquet(str(sample_parquet), columns=["X0", "X2"])
        assert list(df.columns) == ["X0", "X2"]


class TestAutoLoading:
    def test_auto_csv(self, sample_csv: Path) -> None:
        df = load_auto(str(sample_csv))
        assert df.shape[0] == 500

    def test_auto_parquet(self, sample_parquet: Path) -> None:
        df = load_auto(str(sample_parquet))
        assert df.shape[0] == 500


# ═══════════════════════════════════════════════════════════════════════════
# DOT / JSON DAG parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestDOTParsing:
    def test_format_dot(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        dot = format_dot(adj, node_names=["A", "B", "C"])
        assert "A" in dot
        assert "->" in dot

    def test_parse_dot(self, sample_dot: str) -> None:
        adj, names = parse_dot(sample_dot)
        assert adj.shape[0] >= 3
        assert len(names) >= 3

    def test_round_trip_dot(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        names = ["X", "Y", "Z"]
        dot = format_dot(adj, node_names=names)
        adj2, names2 = parse_dot(dot)
        assert adj2.shape == adj.shape
        assert adj2.sum() == adj.sum()


class TestJSONParsing:
    def test_format_json(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        j = format_json(adj, node_names=["A", "B", "C"])
        parsed = json.loads(j)
        assert "edges" in parsed or "n_nodes" in parsed

    def test_parse_json(self) -> None:
        j = format_json(_adj(3, [(0, 1), (1, 2)]), node_names=["A", "B", "C"])
        adj, names, meta = parse_json(j)
        assert adj.shape[0] == 3
        assert adj[0, 1] == 1
        assert names == ["A", "B", "C"]

    def test_round_trip_json(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        names = ["A", "B", "C", "D"]
        j = format_json(adj, node_names=names)
        adj2, names2, _ = parse_json(j)
        np.testing.assert_array_equal(adj, adj2)


# ═══════════════════════════════════════════════════════════════════════════
# DAG save / load
# ═══════════════════════════════════════════════════════════════════════════


class TestSaveLoadDAG:
    def test_save_load_json(self, tmp_dir: Path) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        path = tmp_dir / "test.json"
        save_dag(adj, str(path), node_names=["A", "B", "C"])
        adj2, names2 = load_dag(str(path))
        np.testing.assert_array_equal(adj, adj2)

    def test_save_load_dot(self, tmp_dir: Path) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        path = tmp_dir / "test.dot"
        save_dag(adj, str(path), node_names=["A", "B", "C"])
        adj2, names2 = load_dag(str(path))
        assert adj2.sum() == adj.sum()


# ═══════════════════════════════════════════════════════════════════════════
# Type inference
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeInference:
    def test_continuous_column(self) -> None:
        df = pd.DataFrame({"x": np.random.default_rng(42).standard_normal(100)})
        types = infer_variable_types(df)
        assert types["x"] == VariableType.CONTINUOUS

    def test_binary_column(self) -> None:
        df = pd.DataFrame({"x": [0.0, 1.0] * 50})
        types = infer_variable_types(df)
        assert types["x"] == VariableType.BINARY

    def test_ordinal_column(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5] * 20})
        types = infer_variable_types(df)
        assert types["x"] in (VariableType.ORDINAL, VariableType.NOMINAL)

    def test_is_binary(self) -> None:
        assert is_binary(pd.Series([0, 1, 0, 1, 0]))
        assert not is_binary(pd.Series([0, 1, 2]))

    def test_overrides(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3] * 30, "y": np.random.default_rng(42).standard_normal(90)})
        types = infer_variable_types(df, overrides={"x": VariableType.CONTINUOUS})
        assert types["x"] == VariableType.CONTINUOUS


# ═══════════════════════════════════════════════════════════════════════════
# Data validation
# ═══════════════════════════════════════════════════════════════════════════


class TestDataValidation:
    def test_validate_ok(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        validate_data(df)  # should not raise

    def test_validate_empty_raises(self) -> None:
        df = pd.DataFrame()
        with pytest.raises(Exception):
            validate_data(df, min_rows=1)

    def test_validate_missing_raises(self) -> None:
        df = pd.DataFrame({"x": [1.0, np.nan]})
        with pytest.raises(Exception):
            validate_data(df, allow_missing=False)

    def test_validate_missing_allowed(self) -> None:
        df = pd.DataFrame({"x": [1.0, np.nan]})
        validate_data(df, allow_missing=True)

    def test_dag_data_compatibility(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        df = pd.DataFrame({"X0": [1.0], "X1": [2.0], "X2": [3.0]})
        validate_dag_data_compatibility(adj, df, node_names=["X0", "X1", "X2"])

    def test_detect_outliers(self) -> None:
        rng = np.random.default_rng(42)
        data = np.concatenate([rng.standard_normal(98), [100, -100]])
        df = pd.DataFrame({"x": data})
        result = detect_outliers(df)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════


class TestPreprocessing:
    def test_standardize_zscore(self) -> None:
        df = pd.DataFrame({"x": np.random.default_rng(42).standard_normal(100)})
        result = standardize(df, method="zscore")
        assert abs(result["x"].mean()) < 0.1
        assert abs(result["x"].std() - 1.0) < 0.1

    def test_encode_categorical(self) -> None:
        df = pd.DataFrame({"cat": ["A", "B", "C", "A", "B"]})
        result = encode_categorical(df, columns=["cat"])
        assert result.shape[1] >= 2

    def test_impute_mean(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
        result = impute_missing(df, method="mean")
        assert not result.isnull().any().any()

    def test_preprocess_pipeline(self) -> None:
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.standard_normal(100),
            "y": rng.standard_normal(100),
        })
        result = preprocess(df)
        assert not result.isnull().any().any()

    def test_select_numeric(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0], "cat": ["A", "B"]})
        cols = select_numeric_columns(df)
        assert "x" in cols
        assert "cat" not in cols


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data generation
# ═══════════════════════════════════════════════════════════════════════════


class TestSyntheticData:
    def test_generate_linear_gaussian(self) -> None:
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        df, weights = generate_linear_gaussian(adj, n=500, seed=42)
        assert df.shape == (500, 4)
        assert weights.shape == (4, 4)

    def test_generate_with_treatment(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        df, weights, true_ate = generate_linear_gaussian_with_treatment(
            adj, treatment=1, outcome=2, n=500, true_ate=2.0, seed=42
        )
        assert df.shape[0] == 500
        assert isinstance(true_ate, float)

    def test_generate_nonlinear(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        df = generate_nonlinear(adj, n=200, seed=42)
        assert df.shape == (200, 3)

    def test_random_dag_is_dag(self) -> None:
        from causalcert.dag.validation import is_dag
        adj = random_dag(10, edge_prob=0.3, seed=42)
        assert is_dag(adj)

    def test_compute_true_ate(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        weights = np.zeros((3, 3))
        weights[0, 1] = 0.5
        weights[0, 2] = 0.3
        weights[1, 2] = 0.7
        ate = compute_true_ate_linear(adj, weights, treatment=1, outcome=2)
        assert isinstance(ate, float)

    def test_deterministic_seed(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        df1, _ = generate_linear_gaussian(adj, n=100, seed=42)
        df2, _ = generate_linear_gaussian(adj, n=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases in data loading
# ═══════════════════════════════════════════════════════════════════════════


class TestDataEdgeCases:
    def test_load_nonexistent_file(self) -> None:
        with pytest.raises(Exception):
            load_csv("/nonexistent/path/to/file.csv")

    def test_load_empty_csv(self, tmp_dir: Path) -> None:
        p = tmp_dir / "empty.csv"
        p.write_text("x,y\n")
        df = load_csv(str(p))
        assert df.shape[0] == 0

    def test_single_row_csv(self, tmp_dir: Path) -> None:
        p = tmp_dir / "single.csv"
        p.write_text("x,y\n1.0,2.0\n")
        df = load_csv(str(p))
        assert df.shape[0] == 1

    def test_impute_median(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 10.0]})
        result = impute_missing(df, method="median")
        assert not result.isnull().any().any()

    def test_standardize_minmax(self) -> None:
        df = pd.DataFrame({"x": [0.0, 5.0, 10.0]})
        result = standardize(df, method="minmax")
        assert result["x"].min() >= -0.1
        assert result["x"].max() <= 1.1

    def test_type_summary(self) -> None:
        from causalcert.data.types_inference import type_summary
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "b": [0, 1, 0],
        })
        types = infer_variable_types(df)
        summary = type_summary(df, types)
        assert isinstance(summary, dict)
        assert "x" in summary
