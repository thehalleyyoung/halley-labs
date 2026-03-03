"""Tests for the CausalQD CLI commands."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from causal_qd.cli.main import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV data file."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 5))
    path = tmp_path / "data.csv"
    np.savetxt(str(path), data, delimiter=",", header="v0,v1,v2,v3,v4", comments="")
    return str(path)


@pytest.fixture
def sample_results_dir(tmp_path):
    """Create a sample results directory with results.json."""
    results = {
        "qd_score": 42.5,
        "coverage": 0.3,
        "n_elites": 8,
        "best_adjacency": [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        "qd_score_history": [10, 20, 30, 42.5],
        "coverage_history": [0.1, 0.2, 0.25, 0.3],
        "best_quality_history": [-50.0, -40.0, -35.0, -30.0],
    }
    (tmp_path / "results.json").write_text(json.dumps(results))
    return str(tmp_path)


class TestCLIHelp:
    def test_main_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "CausalQD" in result.output

    def test_run_help(self, runner):
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0

    def test_evaluate_help(self, runner):
        result = runner.invoke(main, ["evaluate", "--help"])
        assert result.exit_code == 0

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestRunCommand:
    def test_run_missing_data(self, runner, tmp_path):
        result = runner.invoke(main, ["run", "--data", str(tmp_path / "nonexistent.csv")])
        assert result.exit_code != 0

    def test_run_small_experiment(self, runner, sample_csv, tmp_path):
        output_dir = str(tmp_path / "output")
        result = runner.invoke(main, [
            "run",
            "--data", sample_csv,
            "--n-generations", "2",
            "--batch-size", "4",
            "--output", output_dir,
            "--seed", "42",
        ])
        assert result.exit_code == 0, f"Output: {result.output}"
        assert Path(output_dir).exists()


class TestEvaluateCommand:
    def test_evaluate_text_output(self, runner, sample_results_dir):
        result = runner.invoke(main, ["evaluate", "--results", sample_results_dir])
        assert result.exit_code == 0
        assert "qd_score" in result.output

    def test_evaluate_json_output(self, runner, sample_results_dir):
        result = runner.invoke(main, ["evaluate", "--results", sample_results_dir, "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "qd_score" in parsed

    def test_evaluate_missing_results(self, runner, tmp_path):
        # tmp_path exists but has no results.json inside
        result = runner.invoke(main, ["evaluate", "--results", str(tmp_path)])
        assert result.exit_code != 0


class TestBenchmarkCommand:
    def test_benchmark_small(self, runner, tmp_path):
        output_dir = str(tmp_path / "bench")
        result = runner.invoke(main, [
            "benchmark",
            "--n-nodes", "3",
            "--n-samples", "50",
            "--n-generations", "2",
            "--batch-size", "4",
            "--output", output_dir,
            "--no-baselines",
        ])
        assert result.exit_code == 0, f"Output: {result.output}"


class TestCertificateCommand:
    def test_certificate_missing_data(self, runner, sample_results_dir):
        result = runner.invoke(main, ["certificate", "--results", sample_results_dir, "--data", "/nonexistent.csv"])
        assert result.exit_code != 0


class TestEvaluateCSVOutput:
    def test_evaluate_csv_output(self, runner, sample_results_dir):
        result = runner.invoke(main, ["evaluate", "--results", sample_results_dir, "--format", "csv"])
        assert result.exit_code == 0
        assert "qd_score" in result.output


class TestFromPandasCommand:
    def test_from_pandas_csv(self, runner):
        result = runner.invoke(main, ["from-pandas"])
        assert result.exit_code == 0
        assert "pandas" in result.output
        assert "to_csv" in result.output

    def test_from_pandas_parquet(self, runner):
        result = runner.invoke(main, ["from-pandas", "--format", "parquet"])
        assert result.exit_code == 0
        assert "to_parquet" in result.output

    def test_from_pandas_npz(self, runner):
        result = runner.invoke(main, ["from-pandas", "--format", "npz"])
        assert result.exit_code == 0
        assert "savez" in result.output

    def test_from_pandas_save_snippet(self, runner, tmp_path):
        out = str(tmp_path / "snippet.py")
        result = runner.invoke(main, ["from-pandas", "--output", out])
        assert result.exit_code == 0
        assert Path(out).exists()
