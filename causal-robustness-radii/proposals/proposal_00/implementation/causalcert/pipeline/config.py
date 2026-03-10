"""
Pipeline run configuration.

Extends the core :class:`PipelineConfig` with runtime settings for
caching, output paths, and reporting options.  Supports loading from
YAML/JSON files and environment-variable overrides.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

from causalcert.types import CITestMethod, PipelineConfig, SolverStrategy


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CITestConfig:
    """Configuration for conditional-independence testing.

    Attributes
    ----------
    method : str
        Primary CI test method.
    alpha : float
        Significance level.
    ensemble_methods : list[str]
        Methods to include in ensemble.
    ensemble_weights : list[float] | None
        Fixed weights, or ``None`` for adaptive.
    adaptive_weights : bool
        Whether to use adaptive weight selection.
    max_conditioning_size : int | None
        Maximum conditioning-set size.
    """

    method: str = "ensemble"
    alpha: float = 0.05
    ensemble_methods: list[str] = field(
        default_factory=lambda: ["partial_correlation", "kernel", "rank"]
    )
    ensemble_weights: list[float] | None = None
    adaptive_weights: bool = True
    max_conditioning_size: int | None = None
    fdr_method: str = "by"
    min_samples_per_test: int = 10


@dataclass(slots=True)
class SolverConfig:
    """Configuration for the robustness-radius solver.

    Attributes
    ----------
    strategy : str
        Solver strategy name.
    k_max : int
        Maximum edit distance.
    time_limit_s : float
        Wall-clock time limit.
    max_treewidth_for_fpt : int
        Treewidth threshold for choosing the FPT solver.
    gap_tolerance : float
        Acceptable optimality gap.
    """

    strategy: str = "auto"
    k_max: int = 10
    time_limit_s: float = 600.0
    max_treewidth_for_fpt: int = 8
    gap_tolerance: float = 0.0
    verbose: bool = False


@dataclass(slots=True)
class EstimationConfig:
    """Configuration for causal-effect estimation.

    Attributes
    ----------
    estimator : str
        Primary estimator (``"aipw"``, ``"ipw"``, ``"regression"``).
    n_folds : int
        Cross-fitting folds.
    propensity_model : str
        Propensity-score model type.
    outcome_model : str
        Outcome-model type.
    ci_alpha : float
        Confidence-interval level.
    """

    estimator: str = "aipw"
    n_folds: int = 5
    propensity_model: str = "logistic"
    outcome_model: str = "linear"
    ci_alpha: float = 0.05


@dataclass(slots=True)
class ReportingConfig:
    """Configuration for report generation.

    Attributes
    ----------
    formats : list[str]
        Output formats.
    detail_level : str
        ``"summary"``, ``"standard"``, or ``"full"``.
    include_plots : bool
        Whether to generate plots.
    output_dir : str | None
        Output directory.
    """

    formats: list[str] = field(default_factory=lambda: ["json"])
    detail_level: str = "standard"
    include_plots: bool = False
    output_dir: str | None = None


@dataclass(slots=True)
class CacheConfig:
    """Configuration for result caching.

    Attributes
    ----------
    enabled : bool
        Whether caching is enabled.
    cache_dir : str
        Cache directory path.
    ttl_seconds : float
        Default TTL for cache entries (0 = no expiry).
    max_size_mb : float
        Maximum cache size.
    """

    enabled: bool = True
    cache_dir: str = ".causalcert_cache"
    ttl_seconds: float = 0.0
    max_size_mb: float = 500.0


@dataclass(slots=True)
class ParallelConfig:
    """Configuration for parallel execution.

    Attributes
    ----------
    n_jobs : int
        Number of parallel workers (``-1`` for all CPUs).
    backend : str
        ``"process"`` or ``"thread"``.
    max_memory_mb : float
        Memory limit.
    chunk_size : int
        Items per parallel chunk.
    """

    n_jobs: int = 1
    backend: str = "thread"
    max_memory_mb: float = 0.0
    chunk_size: int = 50


# ---------------------------------------------------------------------------
# Steps to run
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StepsConfig:
    """Which pipeline steps to execute.

    Attributes
    ----------
    validate : bool
        Validate DAG-data consistency.
    ci_testing : bool
        Run CI tests.
    fragility : bool
        Compute fragility scores.
    radius : bool
        Compute robustness radius.
    estimation : bool
        Estimate causal effects.
    report : bool
        Generate audit report.
    """

    validate: bool = True
    ci_testing: bool = True
    fragility: bool = True
    radius: bool = True
    estimation: bool = True
    report: bool = True


# ---------------------------------------------------------------------------
# Main PipelineRunConfig
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PipelineRunConfig:
    """Full pipeline configuration including runtime settings.

    Attributes
    ----------
    treatment : int
        Treatment variable index.
    outcome : int
        Outcome variable index.
    alpha : float
        Significance level for CI tests.
    ci_method : CITestMethod
        CI test method.
    solver_strategy : SolverStrategy
        Solver strategy.
    max_k : int
        Maximum edit distance to explore.
    n_folds : int
        Number of cross-fitting folds.
    fdr_method : str
        FDR control method.
    n_jobs : int
        Number of parallel workers.
    seed : int
        Random seed.
    cache_dir : str | None
        Cache directory (``None`` to disable caching).
    output_dir : str | None
        Output directory for reports.
    report_formats : list[str]
        Report formats to generate (``"json"``, ``"html"``, ``"latex"``).
    verbose : bool
        Whether to print progress.
    ci_config : CITestConfig
        CI testing configuration.
    solver_config : SolverConfig
        Solver configuration.
    estimation_config : EstimationConfig
        Estimation configuration.
    reporting_config : ReportingConfig
        Reporting configuration.
    cache_config : CacheConfig
        Cache configuration.
    parallel_config : ParallelConfig
        Parallel-execution configuration.
    steps : StepsConfig
        Which steps to run.
    """

    treatment: int = 0
    outcome: int = 1
    alpha: float = 0.05
    ci_method: CITestMethod = CITestMethod.ENSEMBLE
    solver_strategy: SolverStrategy = SolverStrategy.AUTO
    max_k: int = 10
    n_folds: int = 5
    fdr_method: str = "by"
    n_jobs: int = 1
    seed: int = 42
    cache_dir: str | None = ".causalcert_cache"
    output_dir: str | None = None
    report_formats: list[str] = field(default_factory=lambda: ["json"])
    verbose: bool = False

    # Sub-configurations
    ci_config: CITestConfig = field(default_factory=CITestConfig)
    solver_config: SolverConfig = field(default_factory=SolverConfig)
    estimation_config: EstimationConfig = field(default_factory=EstimationConfig)
    reporting_config: ReportingConfig = field(default_factory=ReportingConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    steps: StepsConfig = field(default_factory=StepsConfig)

    def to_pipeline_config(self) -> PipelineConfig:
        """Convert to a core :class:`PipelineConfig`."""
        return PipelineConfig(
            treatment=self.treatment,
            outcome=self.outcome,
            alpha=self.alpha,
            ci_method=self.ci_method,
            solver_strategy=self.solver_strategy,
            max_k=self.max_k,
            n_folds=self.n_folds,
            fdr_method=self.fdr_method,
            n_jobs=self.n_jobs,
            seed=self.seed,
            cache_dir=self.cache_dir,
        )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineRunConfig:
        """Create from a flat or nested dictionary.

        Parameters
        ----------
        d : dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        PipelineRunConfig
        """
        cfg = cls()

        # Map simple top-level keys
        _SIMPLE = {
            "treatment", "outcome", "alpha", "max_k", "n_folds",
            "fdr_method", "n_jobs", "seed", "cache_dir", "output_dir",
            "verbose",
        }
        for k in _SIMPLE:
            if k in d:
                setattr(cfg, k, d[k])

        if "ci_method" in d:
            cfg.ci_method = CITestMethod(d["ci_method"])
        if "solver_strategy" in d:
            cfg.solver_strategy = SolverStrategy(d["solver_strategy"])
        if "report_formats" in d:
            cfg.report_formats = list(d["report_formats"])

        # Sub-configs
        if "ci_testing" in d and isinstance(d["ci_testing"], dict):
            cfg.ci_config = _dict_to_dataclass(CITestConfig, d["ci_testing"])
        if "solver" in d and isinstance(d["solver"], dict):
            cfg.solver_config = _dict_to_dataclass(SolverConfig, d["solver"])
        if "estimation" in d and isinstance(d["estimation"], dict):
            cfg.estimation_config = _dict_to_dataclass(
                EstimationConfig, d["estimation"]
            )
        if "reporting" in d and isinstance(d["reporting"], dict):
            cfg.reporting_config = _dict_to_dataclass(
                ReportingConfig, d["reporting"]
            )
        if "cache" in d and isinstance(d["cache"], dict):
            cfg.cache_config = _dict_to_dataclass(CacheConfig, d["cache"])
        if "parallel" in d and isinstance(d["parallel"], dict):
            cfg.parallel_config = _dict_to_dataclass(ParallelConfig, d["parallel"])
        if "steps" in d and isinstance(d["steps"], dict):
            cfg.steps = _dict_to_dataclass(StepsConfig, d["steps"])

        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> PipelineRunConfig:
        """Load from a JSON configuration file.

        Parameters
        ----------
        path : str | Path
            Path to the JSON file.

        Returns
        -------
        PipelineRunConfig
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls.from_dict(d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineRunConfig:
        """Load from a YAML configuration file.

        Parameters
        ----------
        path : str | Path
            Path to the YAML file.

        Returns
        -------
        PipelineRunConfig
        """
        try:
            import yaml  # type: ignore
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files.")
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        return cls.from_dict(d or {})

    @classmethod
    def from_file(cls, path: str | Path) -> PipelineRunConfig:
        """Load from a JSON or YAML file (auto-detected).

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        PipelineRunConfig
        """
        path = Path(path)
        ext = path.suffix.lower()
        if ext in (".yaml", ".yml"):
            return cls.from_yaml(path)
        return cls.from_json(path)

    @classmethod
    def with_env_overrides(
        cls, base: PipelineRunConfig | None = None
    ) -> PipelineRunConfig:
        """Apply environment-variable overrides on top of *base*.

        Environment variables follow the pattern ``CAUSALCERT_<FIELD>``
        (upper-case).

        Parameters
        ----------
        base : PipelineRunConfig | None

        Returns
        -------
        PipelineRunConfig
        """
        cfg = base or cls()
        prefix = "CAUSALCERT_"

        env_map: dict[str, tuple[str, type]] = {
            "TREATMENT": ("treatment", int),
            "OUTCOME": ("outcome", int),
            "ALPHA": ("alpha", float),
            "MAX_K": ("max_k", int),
            "N_FOLDS": ("n_folds", int),
            "N_JOBS": ("n_jobs", int),
            "SEED": ("seed", int),
            "CACHE_DIR": ("cache_dir", str),
            "OUTPUT_DIR": ("output_dir", str),
            "VERBOSE": ("verbose", _parse_bool),
            "FDR_METHOD": ("fdr_method", str),
        }

        for env_suffix, (attr, conv) in env_map.items():
            val = os.environ.get(f"{prefix}{env_suffix}")
            if val is not None:
                setattr(cfg, attr, conv(val))

        ci_method_env = os.environ.get(f"{prefix}CI_METHOD")
        if ci_method_env:
            cfg.ci_method = CITestMethod(ci_method_env)
        solver_env = os.environ.get(f"{prefix}SOLVER_STRATEGY")
        if solver_env:
            cfg.solver_strategy = SolverStrategy(solver_env)

        return cfg

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dictionary."""
        d: dict[str, Any] = {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "alpha": self.alpha,
            "ci_method": self.ci_method.value,
            "solver_strategy": self.solver_strategy.value,
            "max_k": self.max_k,
            "n_folds": self.n_folds,
            "fdr_method": self.fdr_method,
            "n_jobs": self.n_jobs,
            "seed": self.seed,
            "cache_dir": self.cache_dir,
            "output_dir": self.output_dir,
            "report_formats": self.report_formats,
            "verbose": self.verbose,
            "ci_testing": _dataclass_to_dict(self.ci_config),
            "solver": _dataclass_to_dict(self.solver_config),
            "estimation": _dataclass_to_dict(self.estimation_config),
            "reporting": _dataclass_to_dict(self.reporting_config),
            "cache": _dataclass_to_dict(self.cache_config),
            "parallel": _dataclass_to_dict(self.parallel_config),
            "steps": _dataclass_to_dict(self.steps),
        }
        return d

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialise to JSON.

        Parameters
        ----------
        path : str | Path | None
            If given, write to this file.

        Returns
        -------
        str
            JSON string.
        """
        s = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            Path(path).write_text(s, encoding="utf-8")
        return s

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialise to YAML.

        Parameters
        ----------
        path : str | Path | None

        Returns
        -------
        str
        """
        try:
            import yaml  # type: ignore
        except ImportError:
            raise ImportError("PyYAML is required for YAML output.")
        s = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path is not None:
            Path(path).write_text(s, encoding="utf-8")
        return s

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate configuration and return a list of issues."""
        issues: list[str] = []
        if self.alpha <= 0 or self.alpha >= 1:
            issues.append(f"alpha must be in (0, 1), got {self.alpha}")
        if self.max_k < 1:
            issues.append(f"max_k must be >= 1, got {self.max_k}")
        if self.n_folds < 2:
            issues.append(f"n_folds must be >= 2, got {self.n_folds}")
        if self.treatment == self.outcome:
            issues.append("treatment and outcome must differ")
        if self.treatment < 0:
            issues.append(f"treatment must be >= 0, got {self.treatment}")
        if self.outcome < 0:
            issues.append(f"outcome must be >= 0, got {self.outcome}")
        return issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "on")


def _dict_to_dataclass(cls: type, d: dict[str, Any]) -> Any:
    """Create a dataclass from a dict, ignoring unknown keys."""
    valid = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in d.items() if k in valid}
    return cls(**filtered)


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass to a plain dict."""
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------


def quick_config(
    treatment: int = 0,
    outcome: int = 1,
    **overrides: Any,
) -> PipelineRunConfig:
    """Create a fast-running configuration for quick audits.

    Uses partial correlation only, LP relaxation, and minimal folds.
    """
    cfg = PipelineRunConfig(
        treatment=treatment,
        outcome=outcome,
        ci_method=CITestMethod.PARTIAL_CORRELATION,
        solver_strategy=SolverStrategy.LP_RELAXATION,
        max_k=5,
        n_folds=2,
        n_jobs=1,
    )
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def thorough_config(
    treatment: int = 0,
    outcome: int = 1,
    **overrides: Any,
) -> PipelineRunConfig:
    """Create a thorough configuration for publication-quality audits.

    Uses ensemble CI testing, ILP solver, and 10-fold cross-fitting.
    """
    cfg = PipelineRunConfig(
        treatment=treatment,
        outcome=outcome,
        ci_method=CITestMethod.ENSEMBLE,
        solver_strategy=SolverStrategy.ILP,
        max_k=15,
        n_folds=10,
        n_jobs=-1,
    )
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
