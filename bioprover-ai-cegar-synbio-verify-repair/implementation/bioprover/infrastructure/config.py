"""
Configuration system for BioProver.

Hierarchical dataclass-based configuration with defaults for quick-check,
standard, and thorough verification profiles.  Supports loading from
YAML / JSON files, environment-variable overrides, and validation with
actionable error messages.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field, fields, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SolverBackend(Enum):
    """Supported ODE/SSA solver backends."""
    SCIPY = "scipy"
    ASSIMULO = "assimulo"
    GILLESPY2 = "gillespy2"


class SMTBackend(Enum):
    """Supported SMT solver backends."""
    DREAL = "dreal"
    Z3 = "z3"


class LogLevel(Enum):
    """Log verbosity levels (maps to stdlib logging)."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class VerificationProfile(Enum):
    """Pre-defined verification profiles."""
    QUICK_CHECK = auto()
    STANDARD = auto()
    THOROUGH = auto()


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------

@dataclass
class SolverConfig:
    """Numerical solver settings."""

    backend: SolverBackend = SolverBackend.SCIPY
    ode_method: str = "BDF"
    max_step: float = 0.1
    rtol: float = 1e-6
    atol: float = 1e-9
    t_end: float = 100.0
    num_time_points: int = 1000
    ssa_num_trajectories: int = 1000
    ssa_seed: Optional[int] = None
    stiff_detection: bool = True
    max_integration_retries: int = 3


@dataclass
class CEGARConfig:
    """CEGAR loop settings."""

    max_iterations: int = 50
    initial_granularity: int = 8
    max_granularity: int = 256
    refinement_strategy: str = "counterexample_guided"
    abstraction_timeout_s: float = 60.0
    verification_timeout_s: float = 120.0
    delta_tolerance: float = 1e-3
    use_compositional: bool = True
    parallel_refinement: bool = True
    cache_abstractions: bool = True
    convergence_window: int = 5
    convergence_threshold: float = 1e-4


@dataclass
class RepairConfig:
    """Repair / CEGIS settings."""

    max_cegis_iterations: int = 100
    parameter_search_strategy: str = "bayesian"
    search_bounds_factor: float = 2.0
    max_concurrent_candidates: int = 4
    candidate_batch_size: int = 16
    use_ai_guidance: bool = True
    repair_timeout_s: float = 600.0
    objective: str = "minimize_perturbation"
    perturbation_norm: str = "l2"


@dataclass
class TemporalConfig:
    """Temporal logic and bounded model checking settings."""

    default_bound: int = 100
    time_discretisation: float = 0.1
    supported_logics: List[str] = field(
        default_factory=lambda: ["BLTL", "CSL", "STL"]
    )
    bmc_unrolling_depth: int = 50
    fairness_constraints: bool = False
    use_interval_arithmetic: bool = True


@dataclass
class AIConfig:
    """AI / LLM integration settings."""

    enabled: bool = True
    model_name: str = "gpt-4"
    temperature: float = 0.2
    max_tokens: int = 4096
    api_timeout_s: float = 30.0
    retry_count: int = 3
    cache_responses: bool = True
    cache_dir: str = ".bioprover_cache/ai"
    prompt_template_dir: str = "prompts"
    cost_budget_usd: Optional[float] = None


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class BioProverConfig:
    """Root configuration for the BioProver system."""

    # Sub-configs
    solver: SolverConfig = field(default_factory=SolverConfig)
    cegar: CEGARConfig = field(default_factory=CEGARConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    ai: AIConfig = field(default_factory=AIConfig)

    # Global
    log_level: LogLevel = LogLevel.INFO
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "output"
    random_seed: Optional[int] = 42
    num_workers: int = 4
    memory_limit_mb: int = 8192
    profile: VerificationProfile = VerificationProfile.STANDARD

    # -----------------------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict (enums serialised as their value)."""
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BioProverConfig":
        """Construct from a plain dict, coercing enum strings."""
        return _dict_to_dataclass(cls, data)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "BioProverConfig":
        return cls.from_dict(json.loads(text))

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Return a list of human-readable validation errors (empty = OK)."""
        problems: List[str] = []

        if self.solver.rtol <= 0:
            problems.append("solver.rtol must be positive.")
        if self.solver.atol <= 0:
            problems.append("solver.atol must be positive.")
        if self.solver.t_end <= 0:
            problems.append("solver.t_end must be positive.")
        if self.solver.num_time_points < 2:
            problems.append("solver.num_time_points must be >= 2.")
        if self.solver.ssa_num_trajectories < 1:
            problems.append("solver.ssa_num_trajectories must be >= 1.")

        if self.cegar.max_iterations < 1:
            problems.append("cegar.max_iterations must be >= 1.")
        if self.cegar.initial_granularity < 1:
            problems.append("cegar.initial_granularity must be >= 1.")
        if self.cegar.max_granularity < self.cegar.initial_granularity:
            problems.append(
                "cegar.max_granularity must be >= cegar.initial_granularity."
            )
        if self.cegar.delta_tolerance <= 0:
            problems.append("cegar.delta_tolerance must be positive.")

        if self.repair.max_cegis_iterations < 1:
            problems.append("repair.max_cegis_iterations must be >= 1.")
        if self.repair.search_bounds_factor <= 0:
            problems.append("repair.search_bounds_factor must be positive.")

        if self.temporal.default_bound < 1:
            problems.append("temporal.default_bound must be >= 1.")
        if self.temporal.time_discretisation <= 0:
            problems.append("temporal.time_discretisation must be positive.")

        if self.ai.enabled:
            if self.ai.temperature < 0 or self.ai.temperature > 2:
                problems.append("ai.temperature must be in [0, 2].")
            if self.ai.max_tokens < 1:
                problems.append("ai.max_tokens must be >= 1.")

        if self.num_workers < 1:
            problems.append("num_workers must be >= 1.")
        if self.memory_limit_mb < 64:
            problems.append("memory_limit_mb should be >= 64.")

        return problems


# ---------------------------------------------------------------------------
# Pre-defined profiles
# ---------------------------------------------------------------------------

def quick_check_config() -> BioProverConfig:
    """Fast smoke-test configuration with relaxed tolerances."""
    cfg = BioProverConfig(profile=VerificationProfile.QUICK_CHECK)
    cfg.solver.rtol = 1e-4
    cfg.solver.atol = 1e-6
    cfg.solver.num_time_points = 200
    cfg.solver.ssa_num_trajectories = 100
    cfg.cegar.max_iterations = 10
    cfg.cegar.initial_granularity = 4
    cfg.cegar.max_granularity = 32
    cfg.cegar.verification_timeout_s = 30.0
    cfg.cegar.parallel_refinement = False
    cfg.repair.max_cegis_iterations = 20
    cfg.repair.candidate_batch_size = 4
    cfg.repair.repair_timeout_s = 120.0
    cfg.ai.enabled = False
    cfg.num_workers = 2
    cfg.log_level = LogLevel.WARNING
    return cfg


def standard_config() -> BioProverConfig:
    """Balanced configuration (the default)."""
    return BioProverConfig(profile=VerificationProfile.STANDARD)


def thorough_config() -> BioProverConfig:
    """High-confidence configuration with tight tolerances and many iterations."""
    cfg = BioProverConfig(profile=VerificationProfile.THOROUGH)
    cfg.solver.rtol = 1e-8
    cfg.solver.atol = 1e-12
    cfg.solver.num_time_points = 5000
    cfg.solver.ssa_num_trajectories = 10000
    cfg.cegar.max_iterations = 200
    cfg.cegar.initial_granularity = 16
    cfg.cegar.max_granularity = 1024
    cfg.cegar.verification_timeout_s = 600.0
    cfg.cegar.convergence_window = 10
    cfg.repair.max_cegis_iterations = 500
    cfg.repair.candidate_batch_size = 64
    cfg.repair.repair_timeout_s = 3600.0
    cfg.temporal.bmc_unrolling_depth = 200
    cfg.ai.max_tokens = 8192
    cfg.num_workers = 8
    cfg.memory_limit_mb = 32768
    cfg.log_level = LogLevel.DEBUG
    return cfg


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(
    path: Optional[Union[str, Path]] = None,
    env_prefix: str = "BIOPROVER_",
) -> BioProverConfig:
    """Load configuration from an optional file then overlay env vars.

    Supported file formats: ``.json`` and ``.yaml`` / ``.yml``.
    Environment variables override file values using a flattened naming
    convention: ``BIOPROVER_SOLVER__RTOL=1e-5`` sets ``solver.rtol``.
    """
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        data = _load_file(path)
        cfg = BioProverConfig.from_dict(data)
    else:
        cfg = BioProverConfig()

    _apply_env_overrides(cfg, env_prefix)

    problems = cfg.validate()
    if problems:
        msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {p}" for p in problems
        )
        raise ValueError(msg)

    return cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

T = TypeVar("T")


def _load_file(path: Path) -> Dict[str, Any]:
    """Load a JSON or YAML file into a dict."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]

            return yaml.safe_load(text)  # type: ignore[no-any-return]
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install it with: pip install pyyaml"
            )
    return json.loads(text)


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass tree to a plain dict."""
    if hasattr(obj, "__dataclass_fields__"):
        result: Dict[str, Any] = {}
        for f in fields(obj):
            result[f.name] = _dataclass_to_dict(getattr(obj, f.name))
        return result
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def _coerce_enum(enum_cls: Type[Enum], value: Any) -> Any:
    """Try to convert *value* into an instance of *enum_cls*."""
    if isinstance(value, enum_cls):
        return value
    for member in enum_cls:
        if member.value == value or member.name == value:
            return member
    raise ValueError(f"Cannot coerce {value!r} to {enum_cls.__name__}")


def _dict_to_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively build a dataclass from a dict, coercing enums."""
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        raw = data[f.name]
        ft = f.type if not isinstance(f.type, str) else _resolve_type(cls, f.name)

        if hasattr(ft, "__dataclass_fields__"):
            kwargs[f.name] = _dict_to_dataclass(ft, raw)
        elif isinstance(ft, type) and issubclass(ft, Enum):
            kwargs[f.name] = _coerce_enum(ft, raw)
        else:
            kwargs[f.name] = raw
    return cls(**kwargs)  # type: ignore[call-arg]


def _resolve_type(cls: Type[Any], field_name: str) -> Any:
    """Resolve forward-referenced field types via the module globals."""
    hints = cls.__dataclass_fields__
    f = hints[field_name]
    if isinstance(f.type, str):
        return eval(f.type, globals())  # noqa: S307 – controlled input
    return f.type


def _apply_env_overrides(cfg: BioProverConfig, prefix: str) -> None:
    """Apply ``PREFIX_SECTION__KEY`` environment variables onto *cfg*."""
    sub_map: Dict[str, Any] = {
        "SOLVER": cfg.solver,
        "CEGAR": cfg.cegar,
        "REPAIR": cfg.repair,
        "TEMPORAL": cfg.temporal,
        "AI": cfg.ai,
    }
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        suffix = env_key[len(prefix):]
        parts = suffix.split("__")

        if len(parts) == 2:
            section, attr = parts[0], parts[1].lower()
            target = sub_map.get(section)
            if target is None:
                continue
            _set_field(target, attr, env_val)
        elif len(parts) == 1:
            _set_field(cfg, parts[0].lower(), env_val)


def _set_field(obj: Any, name: str, raw_value: str) -> None:
    """Set a dataclass field, coercing *raw_value* to the field's type."""
    if not hasattr(obj, name):
        return
    current = getattr(obj, name)
    if isinstance(current, bool):
        setattr(obj, name, raw_value.lower() in ("1", "true", "yes"))
    elif isinstance(current, int):
        setattr(obj, name, int(raw_value))
    elif isinstance(current, float):
        setattr(obj, name, float(raw_value))
    elif isinstance(current, Enum):
        setattr(obj, name, _coerce_enum(type(current), raw_value))
    else:
        setattr(obj, name, raw_value)
