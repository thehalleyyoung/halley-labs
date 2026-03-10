"""
usability_oracle.pipeline.runner — End-to-end pipeline orchestration.

The :class:`PipelineRunner` drives all analysis stages in sequence,
manages caching, handles errors, and produces a :class:`PipelineResult`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from usability_oracle.core.enums import PipelineStage
from usability_oracle.core.errors import PipelineError, StageError
from usability_oracle.pipeline.cache import ResultCache
from usability_oracle.pipeline.config import FullPipelineConfig, StageConfig
from usability_oracle.pipeline.stages import StageRegistry, BaseStageExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Output of a single pipeline stage.

    Attributes
    ----------
    stage : PipelineStage
    output : Any
        Stage-specific output data.
    timing : float
        Wall-clock execution time (seconds).
    errors : list[str]
        Error messages (empty on success).
    cached : bool
        Whether the result was served from cache.
    """

    stage: PipelineStage
    output: Any = None
    timing: float = 0.0
    errors: list[str] = field(default_factory=list)
    cached: bool = False

    @property
    def success(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "timing": self.timing,
            "success": self.success,
            "errors": self.errors,
            "cached": self.cached,
        }


@dataclass
class PipelineResult:
    """Aggregated result of a full pipeline run.

    Attributes
    ----------
    stages : dict[PipelineStage, StageResult]
        Per-stage results.
    final_result : Any
        The final output (typically comparison or repair result).
    timing : dict[str, float]
        Per-stage timing + total.
    cache_hits : int
        Number of stages served from cache.
    success : bool
        True if all required stages succeeded.
    errors : list[str]
        Aggregated errors across all stages.
    """

    stages: dict[str, StageResult] = field(default_factory=dict)
    final_result: Any = None
    timing: dict[str, float] = field(default_factory=dict)
    cache_hits: int = 0
    success: bool = False
    errors: list[str] = field(default_factory=list)

    def get_stage(self, stage: PipelineStage) -> StageResult | None:
        return self.stages.get(stage.value)

    @property
    def total_time(self) -> float:
        return self.timing.get("total", 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "total_time": self.total_time,
            "cache_hits": self.cache_hits,
            "errors": self.errors,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "timing": self.timing,
        }


# ---------------------------------------------------------------------------
# PipelineRunner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """Orchestrate the end-to-end usability analysis pipeline.

    Parameters
    ----------
    config : FullPipelineConfig
        Pipeline configuration.
    registry : StageRegistry | None
        Custom stage registry; default created if None.
    cache : ResultCache | None
        Custom cache; default in-memory cache created if None.
    """

    def __init__(
        self,
        config: FullPipelineConfig | None = None,
        registry: StageRegistry | None = None,
        cache: ResultCache | None = None,
    ) -> None:
        self.config = config or FullPipelineConfig.DEFAULT()
        self.registry = registry or StageRegistry.default(self.config.stages)
        self.cache = cache or ResultCache(
            cache_dir=self.config.cache_dir,
            default_ttl=self.config.cache_ttl,
        )
        self._cache_hits = 0

    # ── Main entry point --------------------------------------------------

    def run(
        self,
        config: FullPipelineConfig | None = None,
        source_a: Any = None,
        source_b: Any = None,
        task_spec: Any = None,
    ) -> PipelineResult:
        """Run the full analysis pipeline.

        For a **diff** workflow (comparing two UIs), provide both
        *source_a* and *source_b*.  For a **single analysis**, provide
        only *source_a*.

        Parameters
        ----------
        config : FullPipelineConfig, optional
            Override the runner's config for this run.
        source_a : str | Path
            Primary UI source (HTML, JSON, or file path).
        source_b : str | Path, optional
            Second UI source for comparison mode.
        task_spec : TaskSpec-like, optional
            Task specification for cost analysis.

        Returns
        -------
        PipelineResult
        """
        cfg = config or self.config
        t0 = time.monotonic()
        self._cache_hits = 0

        result = PipelineResult()
        is_diff = source_b is not None

        try:
            # Stage 1: Parse
            tree_a = self._run_parse_stage(source_a, cfg, result)
            tree_b = None
            if is_diff:
                tree_b = self._run_parse_stage(source_b, cfg, result, suffix="_b")

            # Stage 2: Align (only in diff mode)
            alignment = None
            if is_diff and tree_a and tree_b:
                alignment = self._run_stage(
                    PipelineStage.ALIGN, cfg, result,
                    tree_a=tree_a, tree_b=tree_b,
                )

            # Stage 3: Cost annotation
            cost_a = None
            if tree_a:
                cost_a = self._run_stage(
                    PipelineStage.COST, cfg, result,
                    tree=tree_a, task_spec=task_spec,
                    cognitive_config=cfg.oracle.cognitive,
                )

            cost_b = None
            if tree_b:
                cost_b = self._run_stage(
                    PipelineStage.COST, cfg, result,
                    tree=tree_b, task_spec=task_spec,
                    cognitive_config=cfg.oracle.cognitive,
                    cache_suffix="_b",
                )

            # Stage 4: MDP construction
            mdp_a = self._run_stage(
                PipelineStage.MDP_BUILD, cfg, result,
                tree=tree_a, task_spec=task_spec,
            ) if tree_a else None

            mdp_b = None
            if tree_b:
                mdp_b = self._run_stage(
                    PipelineStage.MDP_BUILD, cfg, result,
                    tree=tree_b, task_spec=task_spec,
                    cache_suffix="_b",
                )

            # Stage 5: Bisimulation
            bisim_a = None
            if mdp_a and cfg.is_stage_enabled(PipelineStage.BISIMULATE):
                bisim_a = self._run_stage(
                    PipelineStage.BISIMULATE, cfg, result, mdp=mdp_a,
                )

            # Stage 6: Policy computation
            policy_a = None
            if mdp_a:
                policy_a = self._run_stage(
                    PipelineStage.POLICY, cfg, result,
                    mdp=mdp_a, beta=cfg.oracle.policy.beta_min,
                )

            policy_b = None
            if mdp_b:
                policy_b = self._run_stage(
                    PipelineStage.POLICY, cfg, result,
                    mdp=mdp_b, beta=cfg.oracle.policy.beta_min,
                    cache_suffix="_b",
                )

            # Stage 7: Comparison (diff mode)
            comparison = None
            if is_diff and policy_a and policy_b:
                comparison = self._run_stage(
                    PipelineStage.COMPARE, cfg, result,
                    policy_a=policy_a, policy_b=policy_b,
                )

            # Stage 8: Bottleneck detection
            bottlenecks = None
            if mdp_a:
                bottlenecks = self._run_stage(
                    PipelineStage.BOTTLENECK, cfg, result,
                    mdp=mdp_a, policy=policy_a,
                )

            # Stage 9: Repair synthesis
            repair_result = None
            if mdp_a and bottlenecks and cfg.is_stage_enabled(PipelineStage.REPAIR):
                repair_result = self._run_stage(
                    PipelineStage.REPAIR, cfg, result,
                    mdp=mdp_a, bottlenecks=bottlenecks,
                )

            # Determine final result
            if comparison is not None:
                result.final_result = comparison
            elif repair_result is not None:
                result.final_result = repair_result
            elif bottlenecks is not None:
                result.final_result = bottlenecks
            elif cost_a is not None:
                result.final_result = cost_a

            result.success = all(
                sr.success for sr in result.stages.values()
            )

        except PipelineError as exc:
            result.errors.append(str(exc))
            result.success = False
        except Exception as exc:
            result.errors.append(f"Unexpected error: {exc}")
            result.success = False
            logger.exception("Pipeline failed with unexpected error")

        total_time = time.monotonic() - t0
        result.timing["total"] = total_time
        result.cache_hits = self._cache_hits

        logger.info(
            "Pipeline completed in %.3fs (success=%s, cache_hits=%d)",
            total_time, result.success, result.cache_hits,
        )

        return result

    # ── Stage execution ---------------------------------------------------

    def _run_parse_stage(
        self,
        source: Any,
        cfg: FullPipelineConfig,
        result: PipelineResult,
        suffix: str = "",
    ) -> Any:
        """Run the parse stage for a single source."""
        return self._run_stage(
            PipelineStage.PARSE, cfg, result,
            source=source,
            parser_config=cfg.oracle.parser,
            cache_suffix=suffix,
        )

    def _run_stage(
        self,
        stage: PipelineStage,
        cfg: FullPipelineConfig,
        result: PipelineResult,
        cache_suffix: str = "",
        **kwargs: Any,
    ) -> Any:
        """Execute a single stage with caching and error handling.

        Returns the stage output, or None on failure.
        """
        stage_cfg = cfg.get_stage_config(stage)
        stage_key = f"{stage.value}{cache_suffix}"

        if not stage_cfg.enabled:
            logger.debug("Stage %s disabled, skipping", stage_key)
            return None

        # Check cache
        cache_key = self.cache.compute_key(stage_key, kwargs)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            sr = StageResult(
                stage=stage, output=cached, timing=0.0, cached=True,
            )
            result.stages[stage_key] = sr
            result.timing[stage_key] = 0.0
            logger.debug("Cache hit for %s", stage_key)
            return cached

        # Execute
        if not self.registry.has(stage):
            logger.warning("No executor for stage %s", stage.value)
            return None

        executor = self.registry.get(stage)
        t0 = time.monotonic()

        try:
            # Remove our internal cache_suffix kwarg before passing to executor
            exec_kwargs = {k: v for k, v in kwargs.items() if k != "cache_suffix"}
            output = executor.execute(**exec_kwargs)
            elapsed = time.monotonic() - t0

            sr = StageResult(stage=stage, output=output, timing=elapsed)
            result.stages[stage_key] = sr
            result.timing[stage_key] = elapsed

            # Cache result
            self.cache.set(cache_key, output, ttl=cfg.cache_ttl)

            return output

        except (StageError, Exception) as exc:
            elapsed = time.monotonic() - t0
            sr = StageResult(
                stage=stage, timing=elapsed, errors=[str(exc)],
            )
            result.stages[stage_key] = sr
            result.timing[stage_key] = elapsed
            result.errors.append(f"{stage_key}: {exc}")

            if stage_cfg.fail_fast:
                raise PipelineError(
                    f"Stage {stage_key} failed: {exc}"
                ) from exc

            logger.warning("Stage %s failed (non-fatal): %s", stage_key, exc)
            return None

    # ── Individual stage methods (convenience) ----------------------------

    def _parse_stage(self, source: Any, config: Any) -> Any:
        """Parse source into AccessibilityTree."""
        executor = self.registry.get(PipelineStage.PARSE)
        return executor.execute(source=source, parser_config=config)

    def _align_stage(self, tree_a: Any, tree_b: Any, config: Any) -> Any:
        executor = self.registry.get(PipelineStage.ALIGN)
        return executor.execute(tree_a=tree_a, tree_b=tree_b)

    def _cost_stage(self, tree: Any, task_spec: Any, config: Any) -> Any:
        executor = self.registry.get(PipelineStage.COST)
        return executor.execute(tree=tree, task_spec=task_spec)

    def _mdp_stage(self, tree: Any, task_spec: Any, config: Any) -> Any:
        executor = self.registry.get(PipelineStage.MDP_BUILD)
        return executor.execute(tree=tree, task_spec=task_spec)

    def _bisimulate_stage(self, mdp: Any, config: Any) -> Any:
        executor = self.registry.get(PipelineStage.BISIMULATE)
        return executor.execute(mdp=mdp)

    def _policy_stage(self, mdp: Any, config: Any) -> Any:
        executor = self.registry.get(PipelineStage.POLICY)
        return executor.execute(mdp=mdp)

    def _compare_stage(
        self, policy_a: Any, policy_b: Any, config: Any
    ) -> Any:
        executor = self.registry.get(PipelineStage.COMPARE)
        return executor.execute(policy_a=policy_a, policy_b=policy_b)

    def _bottleneck_stage(self, mdp: Any, policy: Any, config: Any) -> Any:
        executor = self.registry.get(PipelineStage.BOTTLENECK)
        return executor.execute(mdp=mdp, policy=policy)

    def _repair_stage(
        self, mdp: Any, bottlenecks: Any, config: Any
    ) -> Any:
        executor = self.registry.get(PipelineStage.REPAIR)
        return executor.execute(mdp=mdp, bottlenecks=bottlenecks)

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def run_with_progress(
        self,
        source_a: Any,
        source_b: Any,
        task_spec: Any,
        callback: Callable[[str, float, str], None] | None = None,
    ) -> dict[str, Any]:
        """Run the full pipeline with progress callbacks.

        The callback receives (stage_name, progress_fraction, message).
        """
        stages = [
            ("parse_a", 0.1, lambda: self._parse_stage(source_a, {})),
            ("parse_b", 0.2, lambda: self._parse_stage(source_b, {})),
            ("cost_a", 0.3, lambda: self._cost_stage(None, task_spec, {})),
            ("cost_b", 0.4, lambda: self._cost_stage(None, task_spec, {})),
            ("mdp_a", 0.5, lambda: self._mdp_stage(None, task_spec, {})),
            ("mdp_b", 0.6, lambda: self._mdp_stage(None, task_spec, {})),
            ("compare", 0.8, lambda: self._compare_stage(None, None, {})),
            ("bottleneck", 0.9, lambda: self._bottleneck_stage(None, None, {})),
        ]
        results: dict[str, Any] = {}
        for stage_name, progress, executor in stages:
            if callback:
                callback(stage_name, progress, f"Running {stage_name}...")
            try:
                results[stage_name] = executor()
            except Exception as exc:
                results[stage_name] = {"error": str(exc)}
                if callback:
                    callback(stage_name, progress, f"Error in {stage_name}: {exc}")

        if callback:
            callback("done", 1.0, "Pipeline complete")
        return results

    # ------------------------------------------------------------------
    # Dry-run mode
    # ------------------------------------------------------------------

    def dry_run(
        self,
        source_a: Any = None,
        source_b: Any = None,
        task_spec: Any = None,
    ) -> list[dict[str, Any]]:
        """Preview the pipeline stages that would be executed without running them.

        Returns a list of stage descriptions with estimated costs.
        """
        plan: list[dict[str, Any]] = []

        all_stages = [
            PipelineStage.PARSE,
            PipelineStage.ALIGN,
            PipelineStage.COST,
            PipelineStage.MDP_BUILD,
            PipelineStage.BISIMULATE,
            PipelineStage.POLICY,
            PipelineStage.COMPARE,
            PipelineStage.BOTTLENECK,
            PipelineStage.REPAIR,
        ]

        for stage in all_stages:
            registered = stage in self.registry._executors if hasattr(self.registry, '_executors') else True
            plan.append({
                "stage": stage.value,
                "registered": registered,
                "description": _STAGE_DESCRIPTIONS.get(stage, ""),
                "order": len(plan),
            })
        return plan

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def run_with_retries(
        self,
        stage: PipelineStage,
        kwargs: dict[str, Any],
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> Any:
        """Execute a single stage with retry logic.

        Uses exponential backoff between retries.
        """
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                executor = self.registry.get(stage)
                return executor.execute(**kwargs)
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    delay = backoff_factor * (2 ** attempt)
                    import time as _time
                    _time.sleep(delay)

        raise RuntimeError(
            f"Stage {stage.value} failed after {max_retries + 1} attempts: {last_err}"
        ) from last_err

    # ------------------------------------------------------------------
    # Stage profiling
    # ------------------------------------------------------------------

    def profiled_run(
        self,
        source_a: Any,
        source_b: Any,
        task_spec: Any,
    ) -> dict[str, Any]:
        """Run the pipeline and collect detailed timing for each stage."""
        import time as _time

        profile: dict[str, Any] = {}
        overall_start = _time.time()

        stage_defs = [
            ("parse_a", lambda: self._parse_stage(source_a, {})),
            ("parse_b", lambda: self._parse_stage(source_b, {})),
            ("cost", lambda: self._cost_stage(None, task_spec, {})),
            ("mdp", lambda: self._mdp_stage(None, task_spec, {})),
            ("compare", lambda: self._compare_stage(None, None, {})),
            ("bottleneck", lambda: self._bottleneck_stage(None, None, {})),
        ]

        for name, executor in stage_defs:
            t0 = _time.time()
            try:
                result = executor()
                elapsed = _time.time() - t0
                profile[name] = {
                    "elapsed_s": elapsed,
                    "success": True,
                }
            except Exception as exc:
                elapsed = _time.time() - t0
                profile[name] = {
                    "elapsed_s": elapsed,
                    "success": False,
                    "error": str(exc),
                }

        profile["total_elapsed_s"] = _time.time() - overall_start
        return profile


# ---------------------------------------------------------------------------
# Stage descriptions for dry-run
# ---------------------------------------------------------------------------

_STAGE_DESCRIPTIONS: dict[PipelineStage, str] = {
    PipelineStage.PARSE: "Parse accessibility data into tree representation",
    PipelineStage.ALIGN: "Align nodes between before/after trees",
    PipelineStage.COST: "Compute cognitive/motor costs for interactions",
    PipelineStage.MDP_BUILD: "Build MDP from accessibility tree and task spec",
    PipelineStage.BISIMULATE: "Compute bisimulation quotient of MDP",
    PipelineStage.POLICY: "Solve bounded-rational policy via softmax value iteration",
    PipelineStage.COMPARE: "Compare policies/values between before and after",
    PipelineStage.BOTTLENECK: "Identify usability bottlenecks from MDP analysis",
    PipelineStage.REPAIR: "Suggest repairs for identified bottlenecks",
}
