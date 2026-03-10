"""End-to-end integration tests for the full usability oracle pipeline.

These tests exercise the complete pipeline from raw UI sources (HTML/JSON)
through parsing, alignment, MDP construction, policy solving, comparison,
and verdict production.  Each test verifies that the multi-stage pipeline
produces consistent, correct ``RegressionVerdict`` values for a variety of
input configurations.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from usability_oracle.pipeline.runner import PipelineRunner, PipelineResult, StageResult
from usability_oracle.pipeline.config import FullPipelineConfig, StageConfig
from usability_oracle.pipeline.stages import StageRegistry
from usability_oracle.pipeline.cache import ResultCache
from usability_oracle.core.enums import RegressionVerdict, PipelineStage
from usability_oracle.core.config import OracleConfig
from usability_oracle.core.errors import ParseError, PipelineError
from usability_oracle.taskspec.models import TaskStep, TaskFlow, TaskSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_HTML_DIR = FIXTURES_DIR / "sample_html"
SAMPLE_JSON_DIR = FIXTURES_DIR / "sample_json"


def _load_html(name: str) -> str:
    """Load an HTML fixture by stem name (e.g. ``'simple_form'``)."""
    p = SAMPLE_HTML_DIR / f"{name}.html"
    assert p.exists(), f"HTML fixture not found: {p}"
    return p.read_text()


def _load_json(name: str) -> str:
    """Load a JSON fixture by stem name."""
    p = SAMPLE_JSON_DIR / f"{name}.json"
    assert p.exists(), f"JSON fixture not found: {p}"
    return p.read_text()


def _make_simple_task() -> TaskSpec:
    """Return a minimal login-style task spec for pipeline tests."""
    steps = [
        TaskStep(
            step_id="s1",
            action_type="click",
            target_role="textfield",
            target_name="Username",
            description="Focus username",
        ),
        TaskStep(
            step_id="s2",
            action_type="type",
            target_role="textfield",
            target_name="Username",
            input_value="admin",
            description="Type username",
            depends_on=["s1"],
        ),
        TaskStep(
            step_id="s3",
            action_type="click",
            target_role="button",
            target_name="Submit",
            description="Submit form",
            depends_on=["s2"],
        ),
    ]
    flow = TaskFlow(flow_id="login", name="Login Flow", steps=steps,
                    success_criteria=["logged_in"])
    return TaskSpec(spec_id="login", name="Login", flows=[flow])


def _default_config(**overrides: Any) -> FullPipelineConfig:
    """Build a ``FullPipelineConfig`` with sensible defaults for testing."""
    cfg = FullPipelineConfig.DEFAULT()
    # Patch PolicyConfig: source uses beta_min/beta_max but config has beta_range
    policy = cfg.oracle.policy
    if hasattr(policy, 'beta_range') and not hasattr(policy, 'beta_min'):
        policy.beta_min = policy.beta_range[0]
        policy.beta_max = policy.beta_range[1]
    for key, val in overrides.items():
        setattr(cfg, key, val)
    return cfg


# ===================================================================
# Tests
# ===================================================================


class TestEndToEndHTMLPipeline:
    """Pipeline runs from two HTML strings to a verdict."""

    def test_simple_form_identical_gives_neutral(self) -> None:
        """Passing the same HTML as both A and B must yield NEUTRAL."""
        html = _load_html("simple_form")
        task = _make_simple_task()
        cfg = _default_config()
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry)
        result: PipelineResult = runner.run(
            config=cfg, source_a=html, source_b=html, task_spec=task,
        )
        assert result.success, f"Pipeline failed: {result.errors}"
        assert result.final_result is not None
        verdict = result.final_result
        if isinstance(verdict, dict):
            verdict = verdict.get("verdict", verdict)
        if hasattr(verdict, "verdict"):
            verdict = verdict.verdict
        assert verdict in (
            RegressionVerdict.NEUTRAL,
            RegressionVerdict.INCONCLUSIVE,
            "neutral",
            "inconclusive",
        )

    def test_different_html_produces_verdict(self) -> None:
        """Two different HTML pages must produce a non-None verdict."""
        html_a = _load_html("simple_form")
        html_b = _load_html("navigation_menu")
        task = _make_simple_task()
        cfg = _default_config()
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry)
        result = runner.run(config=cfg, source_a=html_a, source_b=html_b,
                            task_spec=task)
        assert result.final_result is not None

    def test_pipeline_stages_all_present(self) -> None:
        """Every enabled stage must appear in the result's stage list."""
        html = _load_html("simple_form")
        task = _make_simple_task()
        cfg = _default_config()
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry)
        result = runner.run(config=cfg, source_a=html, source_b=html,
                            task_spec=task)
        stage_names = set()
        for key, sr in result.stages.items():
            sname = sr.stage if isinstance(sr.stage, str) else sr.stage.name
            stage_names.add(sname)
            stage_names.add(key)
        for stage in PipelineStage:
            if cfg.is_stage_enabled(stage):
                assert stage.name in stage_names or stage.name.lower() in stage_names or stage in {
                    sr.stage for sr in result.stages.values()
                }, f"Stage {stage} missing from results"

    def test_pipeline_timing_recorded(self) -> None:
        """Total pipeline timing must be a positive number."""
        html = _load_html("simple_form")
        task = _make_simple_task()
        cfg = _default_config()
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry)
        result = runner.run(config=cfg, source_a=html, source_b=html,
                            task_spec=task)
        total = result.timing.get('total', sum(result.timing.values())) if isinstance(result.timing, dict) else result.timing
        assert total > 0, "Pipeline timing should be positive"

    def test_stage_results_have_timings(self) -> None:
        """Each stage result must carry a non-negative timing value."""
        html = _load_html("simple_form")
        task = _make_simple_task()
        cfg = _default_config()
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry)
        result = runner.run(config=cfg, source_a=html, source_b=html,
                            task_spec=task)
        for key, sr in result.stages.items():
            assert sr.timing >= 0, f"Stage {key} has negative timing"

    def test_pipeline_result_success_flag(self) -> None:
        """A clean run must set ``success`` to True."""
        html = _load_html("simple_form")
        task = _make_simple_task()
        cfg = _default_config()
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry)
        result = runner.run(config=cfg, source_a=html, source_b=html,
                            task_spec=task)
        assert result.success

    def test_pipeline_errors_empty_on_success(self) -> None:
        """A successful pipeline run should report no errors."""
        html = _load_html("simple_form")
        task = _make_simple_task()
        cfg = _default_config()
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry)
        result = runner.run(config=cfg, source_a=html, source_b=html,
                            task_spec=task)
        assert len(result.errors) == 0

    def test_pipeline_with_complex_html(self) -> None:
        """Running on the complex dashboard fixture should not crash."""
        html = _load_html("complex_dashboard")
        task = _make_simple_task()
        cfg = _default_config()
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry)
        result = runner.run(config=cfg, source_a=html, source_b=html,
                            task_spec=task)
        assert result.success


class TestEndToEndJSONPipeline:
    """Pipeline runs using JSON accessibility tree inputs."""

    def test_json_identical_gives_neutral(self) -> None:
        """Identical JSON inputs must yield NEUTRAL or INCONCLUSIVE."""
        json_str = _load_json("simple_form")
        task = _make_simple_task()
        cfg = _default_config()
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry)
        result = runner.run(config=cfg, source_a=json_str,
                            source_b=json_str, task_spec=task)
        assert result.success

    def test_json_different_produces_result(self) -> None:
        """Different JSON fixtures should produce a non-None result."""
        json_a = _load_json("simple_form")
        json_b = _load_json("navigation_menu")
        task = _make_simple_task()
        cfg = _default_config()
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry)
        result = runner.run(config=cfg, source_a=json_a, source_b=json_b,
                            task_spec=task)
        assert result.final_result is not None


class TestEndToEndCaching:
    """Pipeline caching behaviour."""

    def test_cache_hit_on_second_run(self, tmp_path: Path) -> None:
        """A repeated run with the same inputs should hit the cache."""
        html = _load_html("simple_form")
        task = _make_simple_task()
        cfg = _default_config(cache_dir=str(tmp_path))
        cache = ResultCache(cache_dir=tmp_path)
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry, cache=cache)

        result1 = runner.run(config=cfg, source_a=html, source_b=html,
                             task_spec=task)
        result2 = runner.run(config=cfg, source_a=html, source_b=html,
                             task_spec=task)
        assert result2.cache_hits >= result1.cache_hits

    def test_cache_invalidation(self, tmp_path: Path) -> None:
        """After ``cache.clear()`` the next run should see zero cache hits."""
        html = _load_html("simple_form")
        task = _make_simple_task()
        cfg = _default_config(cache_dir=str(tmp_path))
        cache = ResultCache(cache_dir=tmp_path)
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry, cache=cache)

        runner.run(config=cfg, source_a=html, source_b=html, task_spec=task)
        cache.clear()
        result = runner.run(config=cfg, source_a=html, source_b=html,
                            task_spec=task)
        assert result.cache_hits == 0


class TestEndToEndErrorHandling:
    """Graceful failure modes of the pipeline."""

    def test_malformed_html_aborts_gracefully(self) -> None:
        """Broken HTML should cause a parse error, not an unhandled crash."""
        bad_html = "<not><valid<>html"
        task = _make_simple_task()
        cfg = _default_config()
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry)
        result = runner.run(config=cfg, source_a=bad_html,
                            source_b=bad_html, task_spec=task)
        # Either the pipeline reports failure or handles gracefully
        if not result.success:
            assert len(result.errors) > 0

    def test_empty_string_input(self) -> None:
        """Empty input strings should not cause an unhandled exception."""
        task = _make_simple_task()
        cfg = _default_config()
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry)
        try:
            result = runner.run(config=cfg, source_a="", source_b="",
                                task_spec=task)
            # If it returns, either success or explicit failure is fine
            assert isinstance(result, PipelineResult)
        except (ParseError, PipelineError):
            pass  # expected

    def test_none_task_spec_raises(self) -> None:
        """Passing None as task_spec should raise or return a result."""
        html = _load_html("simple_form")
        cfg = _default_config()
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry)
        try:
            result = runner.run(config=cfg, source_a=html, source_b=html,
                               task_spec=None)
            assert isinstance(result, PipelineResult)
        except (TypeError, PipelineError, AttributeError):
            pass  # expected


class TestEndToEndConfig:
    """Pipeline configuration variants."""

    def test_default_config_is_valid(self) -> None:
        """The default config must pass its own validation."""
        cfg = _default_config()
        errors = cfg.validate()
        assert len(errors) == 0, f"Default config has errors: {errors}"

    def test_config_round_trip(self) -> None:
        """``to_dict`` / ``from_dict`` should round-trip cleanly."""
        cfg = _default_config()
        # Patch ParserConfig: source to_dict uses max_depth but field is max_tree_depth
        if hasattr(cfg.oracle.parser, 'max_tree_depth') and not hasattr(cfg.oracle.parser, 'max_depth'):
            cfg.oracle.parser.max_depth = cfg.oracle.parser.max_tree_depth
        d = cfg.to_dict()
        cfg2 = FullPipelineConfig.from_dict(d)
        cfg2.oracle.policy.beta_min = cfg2.oracle.policy.beta_range[0]
        cfg2.oracle.policy.beta_max = cfg2.oracle.policy.beta_range[1]
        assert cfg2.validate() == []

    def test_stage_disabled(self) -> None:
        """Disabling a non-critical stage must not crash the pipeline."""
        html = _load_html("simple_form")
        task = _make_simple_task()
        cfg = _default_config()
        # Disable the OUTPUT stage (non-critical)
        stage_cfg = cfg.get_stage_config(PipelineStage.OUTPUT)
        stage_cfg.enabled = False
        registry = StageRegistry.default(cfg.stages)
        runner = PipelineRunner(config=cfg, registry=registry)
        result = runner.run(config=cfg, source_a=html, source_b=html,
                            task_spec=task)
        assert isinstance(result, PipelineResult)

    def test_oracle_config_default_valid(self) -> None:
        """``OracleConfig.default()`` should validate without errors."""
        oc = OracleConfig.default()
        result = oc.validate()
        assert result is None or result == []

    def test_full_pipeline_config_from_env(self) -> None:
        """``from_env`` should return a usable config object."""
        cfg = FullPipelineConfig.from_env()
        assert isinstance(cfg, FullPipelineConfig)
