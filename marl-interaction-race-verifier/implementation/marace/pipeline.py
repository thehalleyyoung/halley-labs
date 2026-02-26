"""MARACE Pipeline Orchestrator.

Main pipeline for the Multi-Agent Race Condition Verifier (MARACE) system.
Coordinates loading policies, configuring environments, recording traces,
building happens-before graphs, decomposing interaction groups, running
abstract interpretation, adversarial search, importance sampling, and
generating race catalogs, reports, and proof certificates.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline stage definitions
# ---------------------------------------------------------------------------


class PipelineStage(enum.Enum):
    """Ordered stages of the MARACE analysis pipeline."""

    LOAD_POLICIES = "load_policies"
    CONFIGURE_ENV = "configure_env"
    PARSE_SPEC = "parse_spec"
    RECORD_TRACES = "record_traces"
    BUILD_HB_GRAPH = "build_hb_graph"
    DECOMPOSE_GROUPS = "decompose_groups"
    ABSTRACT_INTERPRET = "abstract_interpret"
    ADVERSARIAL_SEARCH = "adversarial_search"
    IMPORTANCE_SAMPLING = "importance_sampling"
    GENERATE_CATALOG = "generate_catalog"
    GENERATE_REPORTS = "generate_reports"
    GENERATE_CERTIFICATES = "generate_certificates"
    DONE = "done"


_STAGE_ORDER: List[PipelineStage] = [
    PipelineStage.LOAD_POLICIES,
    PipelineStage.CONFIGURE_ENV,
    PipelineStage.PARSE_SPEC,
    PipelineStage.RECORD_TRACES,
    PipelineStage.BUILD_HB_GRAPH,
    PipelineStage.DECOMPOSE_GROUPS,
    PipelineStage.ABSTRACT_INTERPRET,
    PipelineStage.ADVERSARIAL_SEARCH,
    PipelineStage.IMPORTANCE_SAMPLING,
    PipelineStage.GENERATE_CATALOG,
    PipelineStage.GENERATE_REPORTS,
    PipelineStage.GENERATE_CERTIFICATES,
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Complete configuration for a MARACE pipeline run.

    Attributes:
        env_config: Dictionary describing the multi-agent environment.
        policy_paths: File paths to serialised agent policies.
        spec_path: Path to the safety / liveness specification file.
        trace_paths: Optional pre-recorded trace files to use instead of
            recording new traces.
        output_dir: Directory for pipeline outputs.
        checkpoint_dir: If set, enables checkpoint / resume support.
        num_trace_samples: Number of execution traces to sample.
        max_schedule_depth: Maximum depth of schedule exploration.
        abstraction_domain: Abstract domain name (e.g. ``"interval"``,
            ``"octagon"``).
        adversarial_budget: Maximum number of adversarial replay attempts.
        importance_samples: Number of importance-sampling draws for
            probability estimation.
        parallel: Whether to use parallel execution for independent stages.
        max_workers: Thread-pool size; ``None`` lets the runtime decide.
        timeout_s: Global wall-clock timeout in seconds.
        report_formats: List of output report formats (``"text"``,
            ``"json"``, ``"html"``).
        generate_certificates: Whether to emit formal proof certificates.
        verbose: Enable verbose / debug logging.
    """

    env_config: dict
    policy_paths: List[str]
    spec_path: str = ""
    trace_paths: Optional[List[str]] = None

    output_dir: str = "output"
    checkpoint_dir: Optional[str] = None

    num_trace_samples: int = 1000
    max_schedule_depth: int = 20
    abstraction_domain: str = "interval"

    adversarial_budget: int = 100
    importance_samples: int = 10000

    parallel: bool = True
    max_workers: Optional[int] = None
    timeout_s: float = 3600.0

    report_formats: List[str] = field(default_factory=lambda: ["text"])
    generate_certificates: bool = True
    verbose: bool = False

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Construct a ``PipelineConfig`` from a plain dictionary."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Requires the optional ``pyyaml`` dependency.
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML config loading. "
                "Install it with: pip install pyyaml"
            ) from exc
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    """Mutable state that tracks pipeline progress and intermediate results.

    Attributes:
        current_stage: The stage currently being executed.
        completed_stages: Stages that have finished successfully.
        start_time: Wall-clock start time (``time.monotonic``).
        stage_times: Per-stage wall-clock durations in seconds.
        policies: Loaded agent policy objects.
        environment: Configured environment instance.
        specification: Parsed safety/liveness specification.
        traces: Recorded or loaded execution traces.
        hb_graph: Happens-before graph built from traces.
        interaction_groups: Decomposed interaction groups.
        fixpoint_results: Results from abstract interpretation keyed by
            group identifier.
        adversarial_replays: Schedules found by adversarial search that
            witness race conditions.
        probability_bounds: Per-race probability bounds from importance
            sampling.
        race_catalog: Compiled catalog of detected races.
        reports: Generated report objects.
        certificates: Generated proof certificates.
        errors: Accumulated non-fatal error messages.
    """

    current_stage: PipelineStage = PipelineStage.LOAD_POLICIES
    completed_stages: List[PipelineStage] = field(default_factory=list)

    start_time: Optional[float] = None
    stage_times: Dict[str, float] = field(default_factory=dict)

    policies: List[Any] = field(default_factory=list)
    environment: Any = None
    specification: Any = None
    traces: List[Any] = field(default_factory=list)
    hb_graph: Any = None
    interaction_groups: List[Any] = field(default_factory=list)
    fixpoint_results: Dict[str, Any] = field(default_factory=dict)
    adversarial_replays: List[Any] = field(default_factory=list)
    probability_bounds: Dict[str, float] = field(default_factory=dict)
    race_catalog: Any = None
    reports: List[Any] = field(default_factory=list)
    certificates: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def mark_stage_complete(self, stage: PipelineStage) -> None:
        """Record *stage* as successfully completed."""
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)

    def elapsed(self) -> float:
        """Return seconds elapsed since the pipeline started.

        Returns ``0.0`` if the pipeline has not started yet.
        """
        if self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time


# ---------------------------------------------------------------------------
# Checkpoint support
# ---------------------------------------------------------------------------


class PipelineCheckpoint:
    """Persist and restore :class:`PipelineState` for crash recovery.

    Checkpoints are written as pickle files into *checkpoint_dir*.  Only
    the most recent checkpoint is kept.
    """

    _FILENAME = "marace_checkpoint.pkl"

    def __init__(self, checkpoint_dir: str) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def _path(self) -> Path:
        return self._dir / self._FILENAME

    def save(self, state: PipelineState, stage: PipelineStage) -> None:
        """Persist *state* tagged with the last completed *stage*."""
        payload = {
            "stage": stage.value,
            "state": self._serialize_state(state),
        }
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self._path)
        logger.debug("Checkpoint saved at stage %s", stage.value)

    def load(self) -> Tuple[Optional[PipelineState], Optional[PipelineStage]]:
        """Load the most recent checkpoint.

        Returns:
            A ``(state, stage)`` tuple, or ``(None, None)`` if no
            checkpoint exists.
        """
        if not self._path.exists():
            return None, None
        try:
            with open(self._path, "rb") as fh:
                payload = pickle.load(fh)  # noqa: S301
            state = self._deserialize_state(payload["state"])
            stage = PipelineStage(payload["stage"])
            logger.info("Resumed from checkpoint at stage %s", stage.value)
            return state, stage
        except Exception:
            logger.warning("Corrupt checkpoint file; starting fresh", exc_info=True)
            return None, None

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _serialize_state(state: PipelineState) -> dict:
        """Convert *state* into a pickle-friendly dictionary."""
        data: Dict[str, Any] = {}
        data["current_stage"] = state.current_stage.value
        data["completed_stages"] = [s.value for s in state.completed_stages]
        data["start_time"] = state.start_time
        data["stage_times"] = dict(state.stage_times)
        data["policies"] = state.policies
        data["environment"] = state.environment
        data["specification"] = state.specification
        data["traces"] = list(state.traces)
        data["hb_graph"] = state.hb_graph
        data["interaction_groups"] = list(state.interaction_groups)
        data["fixpoint_results"] = dict(state.fixpoint_results)
        data["adversarial_replays"] = list(state.adversarial_replays)
        data["probability_bounds"] = dict(state.probability_bounds)
        data["race_catalog"] = state.race_catalog
        data["reports"] = list(state.reports)
        data["certificates"] = list(state.certificates)
        data["errors"] = list(state.errors)
        return data

    @staticmethod
    def _deserialize_state(data: dict) -> PipelineState:
        """Reconstruct a :class:`PipelineState` from serialised *data*."""
        state = PipelineState()
        state.current_stage = PipelineStage(data["current_stage"])
        state.completed_stages = [PipelineStage(v) for v in data.get("completed_stages", [])]
        state.start_time = data.get("start_time")
        state.stage_times = data.get("stage_times", {})
        state.policies = data.get("policies", [])
        state.environment = data.get("environment")
        state.specification = data.get("specification")
        state.traces = data.get("traces", [])
        state.hb_graph = data.get("hb_graph")
        state.interaction_groups = data.get("interaction_groups", [])
        state.fixpoint_results = data.get("fixpoint_results", {})
        state.adversarial_replays = data.get("adversarial_replays", [])
        state.probability_bounds = data.get("probability_bounds", {})
        state.race_catalog = data.get("race_catalog")
        state.reports = data.get("reports", [])
        state.certificates = data.get("certificates", [])
        state.errors = data.get("errors", [])
        return state

    def cleanup(self) -> None:
        """Remove checkpoint files."""
        if self._path.exists():
            self._path.unlink()
            logger.debug("Checkpoint file removed")


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------


class MARACEPipeline:
    """Orchestrate a full MARACE analysis run.

    Typical usage::

        config = PipelineConfig.from_yaml("config.yaml")
        pipeline = MARACEPipeline(config)
        state = pipeline.run()
        print(pipeline.summary())

    The pipeline executes stages in order, recording timing information,
    handling errors, and optionally checkpointing after each stage.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._state = PipelineState()
        self._checkpoint: Optional[PipelineCheckpoint] = None
        if config.checkpoint_dir is not None:
            self._checkpoint = PipelineCheckpoint(config.checkpoint_dir)

        self._setup_logging()

        # Map each stage enum to its handler.
        self._stage_handlers = {
            PipelineStage.LOAD_POLICIES: self._run_stage_load_policies,
            PipelineStage.CONFIGURE_ENV: self._run_stage_configure_env,
            PipelineStage.PARSE_SPEC: self._run_stage_parse_spec,
            PipelineStage.RECORD_TRACES: self._run_stage_record_traces,
            PipelineStage.BUILD_HB_GRAPH: self._run_stage_build_hb_graph,
            PipelineStage.DECOMPOSE_GROUPS: self._run_stage_decompose_groups,
            PipelineStage.ABSTRACT_INTERPRET: self._run_stage_abstract_interpret,
            PipelineStage.ADVERSARIAL_SEARCH: self._run_stage_adversarial_search,
            PipelineStage.IMPORTANCE_SAMPLING: self._run_stage_importance_sampling,
            PipelineStage.GENERATE_CATALOG: self._run_stage_generate_catalog,
            PipelineStage.GENERATE_REPORTS: self._run_stage_generate_reports,
            PipelineStage.GENERATE_CERTIFICATES: self._run_stage_generate_certificates,
        }

    # -- public interface -----------------------------------------------------

    def run(self) -> PipelineState:
        """Execute the full pipeline and return the final state.

        Stages are run in order.  If a checkpoint exists and the stage has
        already been completed the stage is skipped.  Each stage is timed
        and checkpointed on success.

        Returns:
            The :class:`PipelineState` with all results populated.

        Raises:
            TimeoutError: If the global *timeout_s* is exceeded.
            RuntimeError: If a critical stage fails irrecoverably.
        """
        self._state.start_time = time.monotonic()

        # Attempt to resume from checkpoint.
        if self._checkpoint is not None:
            saved_state, saved_stage = self._checkpoint.load()
            if saved_state is not None and saved_stage is not None:
                self._state = saved_state
                self._state.start_time = time.monotonic()
                self._log(
                    f"Resumed from checkpoint after stage {saved_stage.value}",
                    level="info",
                )

        Path(self._config.output_dir).mkdir(parents=True, exist_ok=True)

        for stage in _STAGE_ORDER:
            if self._should_skip_stage(stage):
                self._log(f"Skipping already-completed stage: {stage.value}")
                continue

            # Global timeout guard.
            if self._state.elapsed() > self._config.timeout_s:
                msg = (
                    f"Global timeout ({self._config.timeout_s}s) exceeded "
                    f"at stage {stage.value}"
                )
                self._log(msg, level="error")
                self._state.errors.append(msg)
                raise TimeoutError(msg)

            self._state.current_stage = stage
            self._log(f"Starting stage: {stage.value}")
            stage_start = time.monotonic()

            handler = self._stage_handlers[stage]
            try:
                result = handler()
                self._apply_stage_result(stage, result)
            except Exception as exc:
                err_msg = f"Stage {stage.value} failed: {exc}"
                self._log(err_msg, level="error")
                self._state.errors.append(err_msg)
                if self._is_critical_stage(stage):
                    raise RuntimeError(err_msg) from exc
                # Non-critical: record and continue.

            elapsed = time.monotonic() - stage_start
            self._state.stage_times[stage.value] = elapsed
            self._state.mark_stage_complete(stage)
            self._log(f"Completed stage {stage.value} in {elapsed:.2f}s")

            if self._checkpoint is not None:
                self._checkpoint.save(self._state, stage)

        self._state.current_stage = PipelineStage.DONE
        self._log("Pipeline finished")
        return self._state

    def summary(self) -> str:
        """Return a human-readable summary of the pipeline results."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("MARACE Pipeline Summary")
        lines.append("=" * 60)
        lines.append(f"Status           : {self._state.current_stage.value}")
        lines.append(f"Elapsed          : {self._state.elapsed():.2f}s")
        lines.append(f"Stages completed : {len(self._state.completed_stages)}")
        lines.append(f"Policies loaded  : {len(self._state.policies)}")
        lines.append(f"Traces           : {len(self._state.traces)}")
        lines.append(
            f"Interaction groups: {len(self._state.interaction_groups)}"
        )
        lines.append(
            f"Fixpoint results : {len(self._state.fixpoint_results)}"
        )
        lines.append(
            f"Adversarial hits : {len(self._state.adversarial_replays)}"
        )
        lines.append(
            f"Probability bounds: {len(self._state.probability_bounds)}"
        )
        lines.append(
            f"Race catalog     : "
            f"{'present' if self._state.race_catalog is not None else 'n/a'}"
        )
        lines.append(f"Reports          : {len(self._state.reports)}")
        lines.append(f"Certificates     : {len(self._state.certificates)}")
        if self._state.errors:
            lines.append(f"Errors           : {len(self._state.errors)}")
            for err in self._state.errors:
                lines.append(f"  - {err}")
        lines.append("")
        lines.append("Stage timings:")
        for stage_name, duration in self._state.stage_times.items():
            lines.append(f"  {stage_name:<25s} {duration:>8.2f}s")
        lines.append("=" * 60)
        return "\n".join(lines)

    # -- stage implementations ------------------------------------------------

    def _run_stage_load_policies(self) -> List[Any]:
        """Load agent policies from the paths specified in config.

        Attempts to import from the ``policy`` subpackage.  Falls back to
        a stub that reads each path as a JSON file.

        Returns:
            A list of policy objects (or dictionaries as stubs).
        """
        policies: List[Any] = []
        try:
            from . import policy as policy_mod  # type: ignore[attr-defined]
            loader = getattr(policy_mod, "load_policy", None)
        except (ImportError, AttributeError):
            loader = None

        for path in self._config.policy_paths:
            try:
                if loader is not None:
                    pol = loader(path)
                else:
                    pol = self._stub_load_policy(path)
                policies.append(pol)
                self._log(f"Loaded policy from {path}")
            except Exception as exc:
                err = f"Failed to load policy {path}: {exc}"
                self._log(err, level="warning")
                self._state.errors.append(err)

        if not policies:
            raise RuntimeError("No policies were loaded; cannot continue")
        return policies

    def _run_stage_configure_env(self) -> Any:
        """Configure the multi-agent environment.

        Attempts to use the ``env`` subpackage; falls back to returning the
        raw *env_config* dictionary.
        """
        try:
            from . import env as env_mod  # type: ignore[attr-defined]
            builder = getattr(env_mod, "build_environment", None)
        except (ImportError, AttributeError):
            builder = None

        if builder is not None:
            environment = builder(self._config.env_config)
        else:
            environment = self._stub_build_environment(self._config.env_config)
        self._log("Environment configured")
        return environment

    def _run_stage_parse_spec(self) -> Any:
        """Parse the safety / liveness specification.

        Returns ``None`` when no spec path is configured.
        """
        if not self._config.spec_path:
            self._log("No specification path provided; skipping parse")
            return None

        try:
            from . import spec as spec_mod  # type: ignore[attr-defined]
            parser = getattr(spec_mod, "parse_spec", None)
        except (ImportError, AttributeError):
            parser = None

        if parser is not None:
            specification = parser(self._config.spec_path)
        else:
            specification = self._stub_parse_spec(self._config.spec_path)
        self._log(f"Specification parsed from {self._config.spec_path}")
        return specification

    def _run_stage_record_traces(self) -> List[Any]:
        """Record execution traces or load them from disk.

        If ``trace_paths`` is set in the config the traces are loaded
        directly.  Otherwise the environment is sampled for
        ``num_trace_samples`` episodes.
        """
        if self._config.trace_paths:
            return self._load_traces(self._config.trace_paths)

        try:
            from . import trace as trace_mod  # type: ignore[attr-defined]
            recorder = getattr(trace_mod, "record_traces", None)
        except (ImportError, AttributeError):
            recorder = None

        if recorder is not None:
            traces = recorder(
                self._state.environment,
                self._state.policies,
                self._config.num_trace_samples,
            )
        else:
            traces = self._stub_record_traces(
                self._state.environment,
                self._state.policies,
                self._config.num_trace_samples,
            )
        self._log(f"Recorded {len(traces)} traces")
        return traces

    def _run_stage_build_hb_graph(self) -> Any:
        """Build a happens-before graph from the recorded traces."""
        try:
            from . import hb as hb_mod  # type: ignore[attr-defined]
            builder = getattr(hb_mod, "build_hb_graph", None)
        except (ImportError, AttributeError):
            builder = None

        if builder is not None:
            graph = builder(self._state.traces)
        else:
            graph = self._stub_build_hb_graph(self._state.traces)
        self._log("Happens-before graph constructed")
        return graph

    def _run_stage_decompose_groups(self) -> List[Any]:
        """Decompose the happens-before graph into interaction groups."""
        try:
            from . import decomposition as decomp_mod  # type: ignore[attr-defined]
            decomposer = getattr(decomp_mod, "decompose", None)
        except (ImportError, AttributeError):
            decomposer = None

        if decomposer is not None:
            groups = decomposer(self._state.hb_graph)
        else:
            groups = self._stub_decompose_groups(self._state.hb_graph)
        self._log(f"Decomposed into {len(groups)} interaction groups")
        return groups

    def _run_stage_abstract_interpret(self) -> Dict[str, Any]:
        """Run abstract interpretation over each interaction group.

        Uses a thread pool when ``config.parallel`` is enabled.
        """
        try:
            from . import abstract as abs_mod  # type: ignore[attr-defined]
            interpreter = getattr(abs_mod, "interpret_group", None)
        except (ImportError, AttributeError):
            interpreter = None

        if interpreter is None:
            interpreter = self._stub_interpret_group

        groups = self._state.interaction_groups
        results: Dict[str, Any] = {}

        if self._config.parallel and len(groups) > 1:
            results = self._parallel_interpret(interpreter, groups)
        else:
            for idx, group in enumerate(groups):
                gid = self._group_id(group, idx)
                try:
                    results[gid] = interpreter(
                        group,
                        domain=self._config.abstraction_domain,
                        spec=self._state.specification,
                    )
                except Exception as exc:
                    err = f"Abstract interpretation failed for group {gid}: {exc}"
                    self._log(err, level="warning")
                    self._state.errors.append(err)

        self._log(f"Abstract interpretation complete for {len(results)} groups")
        return results

    def _run_stage_adversarial_search(self) -> List[Any]:
        """Search for adversarial schedules that witness races."""
        try:
            from . import search as search_mod  # type: ignore[attr-defined]
            searcher = getattr(search_mod, "adversarial_search", None)
        except (ImportError, AttributeError):
            searcher = None

        if searcher is not None:
            replays = searcher(
                self._state.environment,
                self._state.policies,
                self._state.fixpoint_results,
                budget=self._config.adversarial_budget,
                max_depth=self._config.max_schedule_depth,
            )
        else:
            replays = self._stub_adversarial_search(
                self._state.fixpoint_results,
                self._config.adversarial_budget,
            )
        self._log(f"Adversarial search found {len(replays)} witnesses")
        return replays

    def _run_stage_importance_sampling(self) -> Dict[str, float]:
        """Estimate race-condition probabilities via importance sampling."""
        try:
            from . import sampling as samp_mod  # type: ignore[attr-defined]
            sampler = getattr(samp_mod, "importance_sample", None)
        except (ImportError, AttributeError):
            sampler = None

        if sampler is not None:
            bounds = sampler(
                self._state.environment,
                self._state.policies,
                self._state.adversarial_replays,
                n_samples=self._config.importance_samples,
            )
        else:
            bounds = self._stub_importance_sampling(
                self._state.adversarial_replays,
                self._config.importance_samples,
            )
        self._log(f"Importance sampling produced {len(bounds)} bounds")
        return bounds

    def _run_stage_generate_catalog(self) -> Any:
        """Compile a unified race catalog from analysis results."""
        try:
            from . import race as race_mod  # type: ignore[attr-defined]
            cataloger = getattr(race_mod, "build_catalog", None)
        except (ImportError, AttributeError):
            cataloger = None

        if cataloger is not None:
            catalog = cataloger(
                fixpoint_results=self._state.fixpoint_results,
                adversarial_replays=self._state.adversarial_replays,
                probability_bounds=self._state.probability_bounds,
            )
        else:
            catalog = self._stub_build_catalog()
        self._log("Race catalog generated")
        return catalog

    def _run_stage_generate_reports(self) -> List[Any]:
        """Generate human-readable reports in the requested formats."""
        try:
            from . import reporting as rep_mod  # type: ignore[attr-defined]
            reporter = getattr(rep_mod, "generate_report", None)
        except (ImportError, AttributeError):
            reporter = None

        reports: List[Any] = []
        for fmt in self._config.report_formats:
            try:
                if reporter is not None:
                    report = reporter(
                        self._state.race_catalog,
                        format=fmt,
                        output_dir=self._config.output_dir,
                    )
                else:
                    report = self._stub_generate_report(fmt)
                reports.append(report)
                self._log(f"Generated {fmt} report")
            except Exception as exc:
                err = f"Report generation ({fmt}) failed: {exc}"
                self._log(err, level="warning")
                self._state.errors.append(err)
        return reports

    def _run_stage_generate_certificates(self) -> List[Any]:
        """Generate formal proof certificates for verified properties.

        Skipped entirely when ``config.generate_certificates`` is False.
        """
        if not self._config.generate_certificates:
            self._log("Certificate generation disabled; skipping")
            return []

        try:
            from . import race as race_mod  # type: ignore[attr-defined]
            certifier = getattr(race_mod, "generate_certificate", None)
        except (ImportError, AttributeError):
            certifier = None

        certificates: List[Any] = []
        catalog_entries = self._catalog_entries()
        for entry in catalog_entries:
            try:
                if certifier is not None:
                    cert = certifier(
                        entry,
                        fixpoint_results=self._state.fixpoint_results,
                        output_dir=self._config.output_dir,
                    )
                else:
                    cert = self._stub_generate_certificate(entry)
                certificates.append(cert)
            except Exception as exc:
                err = f"Certificate generation failed for entry: {exc}"
                self._log(err, level="warning")
                self._state.errors.append(err)
        self._log(f"Generated {len(certificates)} certificates")
        return certificates

    # -- internal helpers -----------------------------------------------------

    def _should_skip_stage(self, stage: PipelineStage) -> bool:
        """Return ``True`` if *stage* was already completed (e.g. via checkpoint)."""
        return stage in self._state.completed_stages

    def _log(self, msg: str, level: str = "info") -> None:
        """Emit a log message at the requested level."""
        log_fn = getattr(logger, level, logger.info)
        log_fn("[MARACE] %s", msg)

    def _setup_logging(self) -> None:
        """Configure the module-level logger based on config."""
        log_level = logging.DEBUG if self._config.verbose else logging.INFO
        logger.setLevel(log_level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(handler)

    def _apply_stage_result(self, stage: PipelineStage, result: Any) -> None:
        """Store *result* in the appropriate field of ``_state``."""
        mapping = {
            PipelineStage.LOAD_POLICIES: "policies",
            PipelineStage.CONFIGURE_ENV: "environment",
            PipelineStage.PARSE_SPEC: "specification",
            PipelineStage.RECORD_TRACES: "traces",
            PipelineStage.BUILD_HB_GRAPH: "hb_graph",
            PipelineStage.DECOMPOSE_GROUPS: "interaction_groups",
            PipelineStage.ABSTRACT_INTERPRET: "fixpoint_results",
            PipelineStage.ADVERSARIAL_SEARCH: "adversarial_replays",
            PipelineStage.IMPORTANCE_SAMPLING: "probability_bounds",
            PipelineStage.GENERATE_CATALOG: "race_catalog",
            PipelineStage.GENERATE_REPORTS: "reports",
            PipelineStage.GENERATE_CERTIFICATES: "certificates",
        }
        attr = mapping.get(stage)
        if attr is not None:
            setattr(self._state, attr, result)

    @staticmethod
    def _is_critical_stage(stage: PipelineStage) -> bool:
        """Return ``True`` for stages whose failure should abort the pipeline."""
        return stage in {
            PipelineStage.LOAD_POLICIES,
            PipelineStage.CONFIGURE_ENV,
            PipelineStage.RECORD_TRACES,
            PipelineStage.BUILD_HB_GRAPH,
        }

    def _parallel_interpret(
        self,
        interpreter: Any,
        groups: List[Any],
    ) -> Dict[str, Any]:
        """Run *interpreter* over *groups* in parallel threads."""
        results: Dict[str, Any] = {}
        max_workers = self._config.max_workers or min(len(groups), os.cpu_count() or 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_gid = {}
            for idx, group in enumerate(groups):
                gid = self._group_id(group, idx)
                future = executor.submit(
                    interpreter,
                    group,
                    domain=self._config.abstraction_domain,
                    spec=self._state.specification,
                )
                future_to_gid[future] = gid

            for future in as_completed(future_to_gid):
                gid = future_to_gid[future]
                try:
                    results[gid] = future.result()
                except Exception as exc:
                    err = f"Abstract interpretation failed for group {gid}: {exc}"
                    self._log(err, level="warning")
                    self._state.errors.append(err)
        return results

    @staticmethod
    def _group_id(group: Any, index: int) -> str:
        """Derive a string identifier for an interaction group."""
        if isinstance(group, dict) and "id" in group:
            return str(group["id"])
        if hasattr(group, "id"):
            return str(group.id)
        return f"group_{index}"

    def _catalog_entries(self) -> List[Any]:
        """Extract iterable entries from the race catalog."""
        cat = self._state.race_catalog
        if cat is None:
            return []
        if isinstance(cat, dict):
            return cat.get("entries", cat.get("races", []))
        if isinstance(cat, (list, tuple)):
            return list(cat)
        if hasattr(cat, "entries"):
            return list(cat.entries)
        return []

    def _load_traces(self, paths: List[str]) -> List[Any]:
        """Load pre-recorded traces from disk."""
        traces: List[Any] = []
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    traces.extend(data)
                else:
                    traces.append(data)
                self._log(f"Loaded traces from {path}")
            except Exception as exc:
                err = f"Failed to load traces from {path}: {exc}"
                self._log(err, level="warning")
                self._state.errors.append(err)
        if not traces:
            raise RuntimeError("No traces loaded from provided paths")
        return traces

    # -- stub implementations ------------------------------------------------
    # These stubs allow the pipeline to run end-to-end even when the
    # subpackages (policy, env, spec, trace, hb, decomposition, abstract,
    # search, sampling, race, reporting) are not yet fully implemented.

    @staticmethod
    def _stub_load_policy(path: str) -> dict:
        """Stub: load a policy file as a JSON dictionary."""
        p = Path(path)
        if p.exists() and p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {"path": str(p), "type": "stub"}

    @staticmethod
    def _stub_build_environment(env_config: dict) -> dict:
        """Stub: return the environment config as-is."""
        return {"config": env_config, "type": "stub_environment"}

    @staticmethod
    def _stub_parse_spec(spec_path: str) -> dict:
        """Stub: read a spec file as JSON or return a placeholder."""
        p = Path(spec_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as fh:
                try:
                    return json.load(fh)
                except json.JSONDecodeError:
                    return {"raw": fh.read(), "type": "stub_spec"}
        return {"path": str(p), "type": "stub_spec"}

    @staticmethod
    def _stub_record_traces(
        environment: Any,
        policies: List[Any],
        num_samples: int,
    ) -> List[dict]:
        """Stub: generate placeholder trace entries."""
        return [
            {
                "trace_id": i,
                "events": [],
                "agents": len(policies) if policies else 0,
                "type": "stub_trace",
            }
            for i in range(num_samples)
        ]

    @staticmethod
    def _stub_build_hb_graph(traces: List[Any]) -> dict:
        """Stub: return a minimal happens-before graph structure."""
        return {
            "nodes": len(traces),
            "edges": [],
            "type": "stub_hb_graph",
        }

    @staticmethod
    def _stub_decompose_groups(hb_graph: Any) -> List[dict]:
        """Stub: treat the entire graph as a single interaction group."""
        return [{"id": "group_0", "graph": hb_graph, "type": "stub_group"}]

    @staticmethod
    def _stub_interpret_group(
        group: Any,
        *,
        domain: str = "interval",
        spec: Any = None,
    ) -> dict:
        """Stub: return a placeholder fixpoint result."""
        gid = group.get("id", "unknown") if isinstance(group, dict) else "unknown"
        return {
            "group_id": gid,
            "domain": domain,
            "fixpoint_reached": True,
            "violations": [],
            "type": "stub_fixpoint",
        }

    @staticmethod
    def _stub_adversarial_search(
        fixpoint_results: Dict[str, Any],
        budget: int,
    ) -> List[dict]:
        """Stub: return an empty list of adversarial replays."""
        return []

    @staticmethod
    def _stub_importance_sampling(
        replays: List[Any],
        n_samples: int,
    ) -> Dict[str, float]:
        """Stub: return empty probability bounds."""
        return {}

    def _stub_build_catalog(self) -> dict:
        """Stub: compile a minimal race catalog from current state."""
        return {
            "races": [],
            "fixpoint_summary": {
                k: v.get("violations", []) if isinstance(v, dict) else []
                for k, v in self._state.fixpoint_results.items()
            },
            "adversarial_count": len(self._state.adversarial_replays),
            "type": "stub_catalog",
        }

    def _stub_generate_report(self, fmt: str) -> dict:
        """Stub: produce a placeholder report dictionary."""
        output_path = Path(self._config.output_dir) / f"report.{fmt}"
        content = self.summary()
        if fmt == "text":
            output_path.write_text(content, encoding="utf-8")
        elif fmt == "json":
            output_path = output_path.with_suffix(".json")
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "summary": content,
                        "errors": self._state.errors,
                        "stage_times": self._state.stage_times,
                    },
                    fh,
                    indent=2,
                )
        return {"format": fmt, "path": str(output_path), "type": "stub_report"}

    @staticmethod
    def _stub_generate_certificate(entry: Any) -> dict:
        """Stub: return a placeholder certificate for *entry*."""
        entry_id = (
            entry.get("id", "unknown")
            if isinstance(entry, dict)
            else str(entry)
        )
        return {
            "entry_id": entry_id,
            "verified": False,
            "reason": "stub implementation",
            "type": "stub_certificate",
        }
