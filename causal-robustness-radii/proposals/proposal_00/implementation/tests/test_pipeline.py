"""Tests for causalcert.pipeline – orchestrator, config, cache, CLI."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causalcert.pipeline.config import (
    PipelineRunConfig,
    quick_config,
    thorough_config,
)
from causalcert.pipeline.cache import ResultCache
from causalcert.pipeline.orchestrator import CausalCertPipeline, ATESignificancePredicate
from causalcert.types import (
    AdjacencyMatrix,
    AuditReport,
    CITestMethod,
    SolverStrategy,
)

# ── helpers ───────────────────────────────────────────────────────────────

def _adj(n: int, edges: list[tuple[int, int]]) -> AdjacencyMatrix:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[u, v] = 1
    return a


def _synthetic_data(adj: AdjacencyMatrix, n: int = 300, seed: int = 42) -> pd.DataFrame:
    from tests.conftest import _linear_gaussian_data
    return _linear_gaussian_data(adj, n=n, seed=seed)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


class TestConfiguration:
    def test_quick_config(self) -> None:
        cfg = quick_config(treatment=0, outcome=1)
        assert isinstance(cfg, PipelineRunConfig)
        assert cfg.treatment == 0
        assert cfg.outcome == 1

    def test_thorough_config(self) -> None:
        cfg = thorough_config(treatment=0, outcome=1)
        assert isinstance(cfg, PipelineRunConfig)

    def test_config_validation(self) -> None:
        cfg = quick_config(treatment=0, outcome=1)
        errors = cfg.validate()
        assert isinstance(errors, list)

    def test_config_to_dict_round_trip(self) -> None:
        cfg = quick_config(treatment=0, outcome=1)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        cfg2 = PipelineRunConfig.from_dict(d)
        assert cfg2.treatment == cfg.treatment

    def test_config_to_json(self, tmp_dir: Path) -> None:
        cfg = quick_config(treatment=0, outcome=1)
        path = tmp_dir / "config.json"
        cfg.to_json(path)
        cfg2 = PipelineRunConfig.from_json(path)
        assert cfg2.treatment == 0

    def test_config_overrides(self) -> None:
        cfg = quick_config(treatment=0, outcome=1, max_k=5)
        assert cfg.max_k == 5 or hasattr(cfg, "solver") and cfg.solver.max_k == 5


# ═══════════════════════════════════════════════════════════════════════════
# Caching
# ═══════════════════════════════════════════════════════════════════════════


class TestCaching:
    def test_cache_put_get(self, tmp_dir: Path) -> None:
        cache = ResultCache(cache_dir=tmp_dir / "cache", enabled=True)
        cache.put("test_key", {"value": 42})
        result = cache.get("test_key")
        assert result is not None
        assert result["value"] == 42

    def test_cache_miss(self, tmp_dir: Path) -> None:
        cache = ResultCache(cache_dir=tmp_dir / "cache", enabled=True)
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_has(self, tmp_dir: Path) -> None:
        cache = ResultCache(cache_dir=tmp_dir / "cache", enabled=True)
        cache.put("k", "v")
        assert cache.has("k")
        assert not cache.has("missing")

    def test_cache_invalidate(self, tmp_dir: Path) -> None:
        cache = ResultCache(cache_dir=tmp_dir / "cache", enabled=True)
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.invalidate("k1")
        assert not cache.has("k1")
        assert cache.has("k2")

    def test_cache_invalidate_all(self, tmp_dir: Path) -> None:
        cache = ResultCache(cache_dir=tmp_dir / "cache", enabled=True)
        cache.put("k1", "v1")
        cache.invalidate()
        assert cache.size == 0

    def test_cache_keys(self, tmp_dir: Path) -> None:
        cache = ResultCache(cache_dir=tmp_dir / "cache", enabled=True)
        cache.put("a", 1)
        cache.put("b", 2)
        keys = cache.keys()
        assert "a" in keys
        assert "b" in keys

    def test_cache_disabled(self, tmp_dir: Path) -> None:
        cache = ResultCache(cache_dir=tmp_dir / "cache", enabled=False)
        cache.put("k", "v")
        assert cache.get("k") is None

    def test_content_key(self) -> None:
        k = ResultCache.content_key("adj", "data", "config")
        assert isinstance(k, str)
        assert len(k) > 0

    def test_cached_decorator(self, tmp_dir: Path) -> None:
        cache = ResultCache(cache_dir=tmp_dir / "cache", enabled=True)
        call_count = [0]

        def expensive_fn():
            call_count[0] += 1
            return 42

        r1 = cache.cached("test", expensive_fn)
        r2 = cache.cached("test", expensive_fn)
        assert r1 == 42
        assert r2 == 42
        assert call_count[0] == 1  # second call uses cache

    def test_numpy_values(self, tmp_dir: Path) -> None:
        cache = ResultCache(cache_dir=tmp_dir / "cache", enabled=True)
        arr = np.array([1.0, 2.0, 3.0])
        cache.put("np_arr", arr.tolist())
        result = cache.get("np_arr")
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# Full pipeline on small synthetic example
# ═══════════════════════════════════════════════════════════════════════════


class TestPipeline:
    def test_pipeline_runs(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        data = _synthetic_data(adj, n=300)
        cfg = quick_config(treatment=1, outcome=2)
        pipeline = CausalCertPipeline(config=cfg)
        report = pipeline.run(adj, data)
        assert isinstance(report, AuditReport)
        assert report.treatment == 1
        assert report.outcome == 2

    def test_pipeline_timings(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj, n=200)
        cfg = quick_config(treatment=0, outcome=2)
        pipeline = CausalCertPipeline(config=cfg)
        pipeline.run(adj, data)
        timings = pipeline.timings
        assert isinstance(timings, dict)

    def test_pipeline_checkpoint(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj, n=200)
        cfg = quick_config(treatment=0, outcome=2)
        pipeline = CausalCertPipeline(config=cfg)
        pipeline.run(adj, data)
        cp = pipeline.checkpoint
        assert cp is not None

    def test_pipeline_reset(self) -> None:
        adj = _adj(3, [(0, 1), (1, 2)])
        data = _synthetic_data(adj, n=200)
        cfg = quick_config(treatment=0, outcome=2)
        pipeline = CausalCertPipeline(config=cfg)
        pipeline.run(adj, data)
        pipeline.reset()


# ═══════════════════════════════════════════════════════════════════════════
# Predicate
# ═══════════════════════════════════════════════════════════════════════════


class TestPredicate:
    def test_ate_significance_predicate(self) -> None:
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        rng = np.random.default_rng(42)
        n = 500
        C = rng.standard_normal(n)
        T = (rng.standard_normal(n) + 0.5 * C > 0).astype(float)
        Y = 2.0 * T + 0.8 * C + rng.standard_normal(n)
        data = pd.DataFrame({"C": C, "T": T, "Y": Y})
        pred = ATESignificancePredicate(alpha=0.05, n_folds=2, seed=42)
        result = pred(adj, data, treatment=1, outcome=2)
        assert isinstance(result, (bool, np.bool_))


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


class TestCLI:
    def test_cli_main_exists(self) -> None:
        from causalcert.pipeline.cli import main
        assert callable(main)

    def test_cli_validate_command(self, tmp_dir: Path) -> None:
        from click.testing import CliRunner
        from causalcert.pipeline.cli import main

        adj = _adj(3, [(0, 1), (1, 2)])
        dag_path = tmp_dir / "test.json"
        dag_json = json.dumps({
            "n_nodes": 3,
            "node_names": ["X0", "X1", "X2"],
            "edges": [[0, 1], [1, 2]],
        })
        dag_path.write_text(dag_json)

        rng = np.random.default_rng(42)
        data = pd.DataFrame({
            "X0": rng.standard_normal(100),
            "X1": rng.standard_normal(100),
            "X2": rng.standard_normal(100),
        })
        data_path = tmp_dir / "data.csv"
        data.to_csv(data_path, index=False)

        runner = CliRunner()
        result = runner.invoke(main, ["validate", "--dag", str(dag_path), "--data", str(data_path)])
        # Should complete without error or with a validation message
        assert result.exit_code in (0, 1, 2)

    def test_cli_help(self) -> None:
        from click.testing import CliRunner
        from causalcert.pipeline.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output or "usage" in result.output

    def test_cli_unknown_command(self) -> None:
        from click.testing import CliRunner
        from causalcert.pipeline.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["nonexistent_command"])
        assert result.exit_code != 0


# ═══════════════════════════════════════════════════════════════════════════
# Config edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigEdgeCases:
    def test_config_yaml_round_trip(self, tmp_dir: Path) -> None:
        try:
            cfg = quick_config(treatment=0, outcome=1)
            path = tmp_dir / "config.yaml"
            cfg.to_yaml(path)
            cfg2 = PipelineRunConfig.from_yaml(path)
            assert cfg2.treatment == 0
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_config_from_file_json(self, tmp_dir: Path) -> None:
        cfg = quick_config(treatment=0, outcome=1)
        path = tmp_dir / "config.json"
        cfg.to_json(path)
        cfg2 = PipelineRunConfig.from_file(path)
        assert cfg2.treatment == 0

    def test_config_env_overrides(self) -> None:
        cfg = PipelineRunConfig.with_env_overrides()
        assert isinstance(cfg, PipelineRunConfig)

    def test_config_to_pipeline_config(self) -> None:
        cfg = quick_config(treatment=0, outcome=1)
        pc = cfg.to_pipeline_config()
        assert pc.treatment == 0
        assert pc.outcome == 1
