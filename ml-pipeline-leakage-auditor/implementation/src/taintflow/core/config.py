"""
taintflow.core.config – Configuration system for TaintFlow.

Supports loading from TOML files, JSON files, environment variables, and
CLI arguments with well-defined merge priority.  Provides pre-built
configuration presets for common use-cases (quick audit, thorough audit,
CI mode).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from taintflow.core.errors import ConfigError, ValidationError

# ===================================================================
#  Severity thresholds
# ===================================================================


@dataclass
class SeverityThresholds:
    """Bit-bound thresholds that map continuous leakage values to severity."""

    negligible_max: float = 1.0
    warning_max: float = 8.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.negligible_max < 0.0:
            errors.append(f"negligible_max must be >= 0, got {self.negligible_max}")
        if self.warning_max < self.negligible_max:
            errors.append(
                f"warning_max ({self.warning_max}) must be >= negligible_max ({self.negligible_max})"
            )
        return errors

    @property
    def critical_min(self) -> float:
        return self.warning_max

    def classify(self, bits: float) -> str:
        if bits <= self.negligible_max:
            return "negligible"
        if bits <= self.warning_max:
            return "warning"
        return "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "negligible_max": self.negligible_max,
            "warning_max": self.warning_max,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SeverityThresholds":
        return cls(
            negligible_max=float(data.get("negligible_max", 1.0)),
            warning_max=float(data.get("warning_max", 8.0)),
        )

    def __repr__(self) -> str:
        return (
            f"Thresholds(negligible≤{self.negligible_max}, "
            f"warning≤{self.warning_max}, critical>{self.warning_max})"
        )


# ===================================================================
#  Channel-capacity settings
# ===================================================================


@dataclass
class ChannelCapacitySettings:
    """Settings controlling how channel capacities are estimated."""

    tier_preference: str = "analytic"
    fallback_tier: str = "sampling"
    analytic_timeout_ms: int = 5_000
    sampling_n_samples: int = 10_000
    sampling_seed: int = 42
    monte_carlo_iterations: int = 1_000
    use_cache: bool = True
    cache_size: int = 4096
    numerical_precision: str = "float64"

    _VALID_TIERS = {"analytic", "sampling", "bounding", "heuristic"}
    _VALID_PRECISIONS = {"float32", "float64", "float128"}

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.tier_preference not in self._VALID_TIERS:
            errors.append(
                f"tier_preference must be one of {self._VALID_TIERS}, got {self.tier_preference!r}"
            )
        if self.fallback_tier not in self._VALID_TIERS:
            errors.append(
                f"fallback_tier must be one of {self._VALID_TIERS}, got {self.fallback_tier!r}"
            )
        if self.analytic_timeout_ms < 0:
            errors.append(f"analytic_timeout_ms must be >= 0, got {self.analytic_timeout_ms}")
        if self.sampling_n_samples < 1:
            errors.append(f"sampling_n_samples must be >= 1, got {self.sampling_n_samples}")
        if self.monte_carlo_iterations < 1:
            errors.append(f"monte_carlo_iterations must be >= 1, got {self.monte_carlo_iterations}")
        if self.cache_size < 0:
            errors.append(f"cache_size must be >= 0, got {self.cache_size}")
        if self.numerical_precision not in self._VALID_PRECISIONS:
            errors.append(
                f"numerical_precision must be one of {self._VALID_PRECISIONS}, "
                f"got {self.numerical_precision!r}"
            )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_preference": self.tier_preference,
            "fallback_tier": self.fallback_tier,
            "analytic_timeout_ms": self.analytic_timeout_ms,
            "sampling_n_samples": self.sampling_n_samples,
            "sampling_seed": self.sampling_seed,
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "use_cache": self.use_cache,
            "cache_size": self.cache_size,
            "numerical_precision": self.numerical_precision,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChannelCapacitySettings":
        return cls(
            tier_preference=str(data.get("tier_preference", "analytic")),
            fallback_tier=str(data.get("fallback_tier", "sampling")),
            analytic_timeout_ms=int(data.get("analytic_timeout_ms", 5_000)),
            sampling_n_samples=int(data.get("sampling_n_samples", 10_000)),
            sampling_seed=int(data.get("sampling_seed", 42)),
            monte_carlo_iterations=int(data.get("monte_carlo_iterations", 1_000)),
            use_cache=bool(data.get("use_cache", True)),
            cache_size=int(data.get("cache_size", 4096)),
            numerical_precision=str(data.get("numerical_precision", "float64")),
        )

    def __repr__(self) -> str:
        return (
            f"ChannelCapacity(tier={self.tier_preference}, "
            f"fallback={self.fallback_tier}, prec={self.numerical_precision})"
        )


# ===================================================================
#  Instrumentation settings
# ===================================================================


@dataclass
class InstrumentationSettings:
    """Settings for the monkey-patching / tracing subsystem."""

    trace_depth: int = 50
    excluded_modules: Tuple[str, ...] = ("taintflow", "logging", "importlib")
    include_numpy: bool = True
    include_pandas: bool = True
    include_sklearn: bool = True
    record_shapes: bool = True
    record_dtypes: bool = True
    record_timing: bool = False
    max_trace_events: int = 500_000

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.trace_depth < 1:
            errors.append(f"trace_depth must be >= 1, got {self.trace_depth}")
        if self.max_trace_events < 1:
            errors.append(f"max_trace_events must be >= 1, got {self.max_trace_events}")
        return errors

    def is_module_excluded(self, module_name: str) -> bool:
        return any(module_name.startswith(excl) for excl in self.excluded_modules)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_depth": self.trace_depth,
            "excluded_modules": list(self.excluded_modules),
            "include_numpy": self.include_numpy,
            "include_pandas": self.include_pandas,
            "include_sklearn": self.include_sklearn,
            "record_shapes": self.record_shapes,
            "record_dtypes": self.record_dtypes,
            "record_timing": self.record_timing,
            "max_trace_events": self.max_trace_events,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InstrumentationSettings":
        return cls(
            trace_depth=int(data.get("trace_depth", 50)),
            excluded_modules=tuple(data.get("excluded_modules", ("taintflow", "logging", "importlib"))),
            include_numpy=bool(data.get("include_numpy", True)),
            include_pandas=bool(data.get("include_pandas", True)),
            include_sklearn=bool(data.get("include_sklearn", True)),
            record_shapes=bool(data.get("record_shapes", True)),
            record_dtypes=bool(data.get("record_dtypes", True)),
            record_timing=bool(data.get("record_timing", False)),
            max_trace_events=int(data.get("max_trace_events", 500_000)),
        )

    def __repr__(self) -> str:
        libs = []
        if self.include_pandas:
            libs.append("pd")
        if self.include_sklearn:
            libs.append("sk")
        if self.include_numpy:
            libs.append("np")
        return f"Instrumentation(depth={self.trace_depth}, libs=[{','.join(libs)}])"


# ===================================================================
#  Report settings
# ===================================================================


@dataclass
class ReportSettings:
    """Settings for report generation."""

    format: str = "text"
    output_path: str = ""
    include_remediation: bool = True
    include_code_snippets: bool = False
    include_dag_visualization: bool = False
    max_features_per_stage: int = 50
    sort_by: str = "severity"
    include_summary: bool = True
    include_config_snapshot: bool = True
    json_indent: int = 2

    _VALID_FORMATS = {"text", "json", "html", "markdown", "sarif"}
    _VALID_SORT = {"severity", "bit_bound", "name", "stage"}

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.format not in self._VALID_FORMATS:
            errors.append(
                f"format must be one of {self._VALID_FORMATS}, got {self.format!r}"
            )
        if self.sort_by not in self._VALID_SORT:
            errors.append(
                f"sort_by must be one of {self._VALID_SORT}, got {self.sort_by!r}"
            )
        if self.max_features_per_stage < 1:
            errors.append(f"max_features_per_stage must be >= 1, got {self.max_features_per_stage}")
        if self.json_indent < 0:
            errors.append(f"json_indent must be >= 0, got {self.json_indent}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "output_path": self.output_path,
            "include_remediation": self.include_remediation,
            "include_code_snippets": self.include_code_snippets,
            "include_dag_visualization": self.include_dag_visualization,
            "max_features_per_stage": self.max_features_per_stage,
            "sort_by": self.sort_by,
            "include_summary": self.include_summary,
            "include_config_snapshot": self.include_config_snapshot,
            "json_indent": self.json_indent,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReportSettings":
        return cls(
            format=str(data.get("format", "text")),
            output_path=str(data.get("output_path", "")),
            include_remediation=bool(data.get("include_remediation", True)),
            include_code_snippets=bool(data.get("include_code_snippets", False)),
            include_dag_visualization=bool(data.get("include_dag_visualization", False)),
            max_features_per_stage=int(data.get("max_features_per_stage", 50)),
            sort_by=str(data.get("sort_by", "severity")),
            include_summary=bool(data.get("include_summary", True)),
            include_config_snapshot=bool(data.get("include_config_snapshot", True)),
            json_indent=int(data.get("json_indent", 2)),
        )

    def __repr__(self) -> str:
        return f"Report(format={self.format}, remediation={self.include_remediation})"


# ===================================================================
#  Main configuration
# ===================================================================


@dataclass
class TaintFlowConfig:
    """Top-level configuration for a TaintFlow audit run.

    Settings are loaded and merged in this priority order (highest wins):
    1. CLI arguments
    2. Environment variables (``TAINTFLOW_*``)
    3. Project-local config file (``taintflow.toml`` / ``taintflow.json``)
    4. User-level config (``~/.config/taintflow/config.toml``)
    5. Built-in defaults (this dataclass)
    """

    # -- analysis parameters -------------------------------------------------
    b_max: float = 64.0
    alpha: float = 0.05
    max_iterations: int = 1000
    use_widening: bool = True
    widening_delay: int = 3
    use_narrowing: bool = True
    narrowing_iterations: int = 5
    epsilon: float = 1e-10

    # -- execution -----------------------------------------------------------
    parallel: bool = False
    n_workers: int = 1
    verbosity: int = 1

    # -- sub-configs ---------------------------------------------------------
    severity: SeverityThresholds = field(default_factory=SeverityThresholds)
    channel: ChannelCapacitySettings = field(default_factory=ChannelCapacitySettings)
    instrumentation: InstrumentationSettings = field(default_factory=InstrumentationSettings)
    report: ReportSettings = field(default_factory=ReportSettings)

    # -- meta ----------------------------------------------------------------
    config_path: str = ""
    profile: str = "default"

    # -----------------------------------------------------------------------
    #  Validation
    # -----------------------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.b_max <= 0:
            errors.append(f"b_max must be > 0, got {self.b_max}")
        if not (0.0 < self.alpha < 1.0):
            errors.append(f"alpha must be in (0, 1), got {self.alpha}")
        if self.max_iterations < 1:
            errors.append(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.widening_delay < 0:
            errors.append(f"widening_delay must be >= 0, got {self.widening_delay}")
        if self.epsilon <= 0:
            errors.append(f"epsilon must be > 0, got {self.epsilon}")
        if self.n_workers < 1:
            errors.append(f"n_workers must be >= 1, got {self.n_workers}")
        if self.verbosity < 0:
            errors.append(f"verbosity must be >= 0, got {self.verbosity}")
        errors.extend(self.severity.validate())
        errors.extend(self.channel.validate())
        errors.extend(self.instrumentation.validate())
        errors.extend(self.report.validate())
        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValidationError(
                "Configuration validation failed",
                violations=errors,
            )

    # -----------------------------------------------------------------------
    #  Loading from various sources
    # -----------------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str | Path) -> "TaintFlowConfig":
        """Load configuration from a TOML file."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}", key="config_path", value=str(path))

        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                raise ConfigError(
                    "Cannot parse TOML: install Python 3.11+ or 'tomli' package.",
                    suggestion="pip install tomli",
                )

        with open(path, "rb") as f:
            data = tomllib.load(f)

        config = cls._from_flat_dict(data.get("taintflow", data))
        config.config_path = str(path)
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "TaintFlowConfig":
        """Load configuration from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}", key="config_path", value=str(path))
        with open(path) as f:
            data = json.load(f)
        config = cls._from_flat_dict(data.get("taintflow", data))
        config.config_path = str(path)
        return config

    @classmethod
    def from_env(cls) -> "TaintFlowConfig":
        """Load configuration from ``TAINTFLOW_*`` environment variables."""
        prefix = "TAINTFLOW_"
        env_data: dict[str, Any] = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                env_data[config_key] = _parse_env_value(value)
        if not env_data:
            return cls()
        return cls._from_flat_dict(env_data)

    @classmethod
    def from_cli_args(cls, **kwargs: Any) -> "TaintFlowConfig":
        """Create config from keyword arguments (typically from Click)."""
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        return cls._from_flat_dict(filtered)

    @classmethod
    def _from_flat_dict(cls, data: Mapping[str, Any]) -> "TaintFlowConfig":
        """Build a config from a potentially nested dict."""
        severity_data = data.get("severity", data.get("severity_thresholds", {}))
        channel_data = data.get("channel", data.get("channel_capacity", {}))
        instr_data = data.get("instrumentation", {})
        report_data = data.get("report", {})

        return cls(
            b_max=float(data.get("b_max", 64.0)),
            alpha=float(data.get("alpha", 0.05)),
            max_iterations=int(data.get("max_iterations", 1000)),
            use_widening=bool(data.get("use_widening", True)),
            widening_delay=int(data.get("widening_delay", 3)),
            use_narrowing=bool(data.get("use_narrowing", True)),
            narrowing_iterations=int(data.get("narrowing_iterations", 5)),
            epsilon=float(data.get("epsilon", 1e-10)),
            parallel=bool(data.get("parallel", False)),
            n_workers=int(data.get("n_workers", 1)),
            verbosity=int(data.get("verbosity", 1)),
            severity=SeverityThresholds.from_dict(severity_data) if severity_data else SeverityThresholds(),
            channel=ChannelCapacitySettings.from_dict(channel_data) if channel_data else ChannelCapacitySettings(),
            instrumentation=InstrumentationSettings.from_dict(instr_data) if instr_data else InstrumentationSettings(),
            report=ReportSettings.from_dict(report_data) if report_data else ReportSettings(),
            profile=str(data.get("profile", "default")),
        )

    # -----------------------------------------------------------------------
    #  Merge multiple configs
    # -----------------------------------------------------------------------

    def merge(self, override: "TaintFlowConfig") -> "TaintFlowConfig":
        """Merge ``override`` on top of ``self`` (override wins on non-default values)."""
        default = TaintFlowConfig()
        merged_data: dict[str, Any] = {}

        for fld in (
            "b_max", "alpha", "max_iterations", "use_widening", "widening_delay",
            "use_narrowing", "narrowing_iterations", "epsilon", "parallel",
            "n_workers", "verbosity", "profile",
        ):
            override_val = getattr(override, fld)
            default_val = getattr(default, fld)
            self_val = getattr(self, fld)
            merged_data[fld] = override_val if override_val != default_val else self_val

        merged_severity = _merge_dataclass(self.severity, override.severity, SeverityThresholds())
        merged_channel = _merge_dataclass(self.channel, override.channel, ChannelCapacitySettings())
        merged_instr = _merge_dataclass(self.instrumentation, override.instrumentation, InstrumentationSettings())
        merged_report = _merge_dataclass(self.report, override.report, ReportSettings())

        return TaintFlowConfig(
            severity=merged_severity,
            channel=merged_channel,
            instrumentation=merged_instr,
            report=merged_report,
            config_path=override.config_path or self.config_path,
            **merged_data,
        )

    @classmethod
    def load(
        cls,
        *,
        config_path: str | Path | None = None,
        cli_overrides: Mapping[str, Any] | None = None,
    ) -> "TaintFlowConfig":
        """Load and merge configuration from all sources.

        Priority (highest wins): CLI > env > file > defaults.
        """
        base = cls()

        if config_path is not None:
            p = Path(config_path)
            if p.suffix in (".toml",):
                file_cfg = cls.from_toml(p)
            elif p.suffix in (".json",):
                file_cfg = cls.from_json(p)
            else:
                raise ConfigError(
                    f"Unsupported config format: {p.suffix}",
                    key="config_path",
                    value=str(p),
                    suggestion="Use .toml or .json format.",
                )
            base = base.merge(file_cfg)
        else:
            for candidate in _default_config_paths():
                if candidate.exists():
                    if candidate.suffix == ".toml":
                        base = base.merge(cls.from_toml(candidate))
                    elif candidate.suffix == ".json":
                        base = base.merge(cls.from_json(candidate))
                    break

        env_cfg = cls.from_env()
        base = base.merge(env_cfg)

        if cli_overrides:
            cli_cfg = cls.from_cli_args(**cli_overrides)
            base = base.merge(cli_cfg)

        base.validate_or_raise()
        return base

    # -----------------------------------------------------------------------
    #  Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "b_max": self.b_max,
            "alpha": self.alpha,
            "max_iterations": self.max_iterations,
            "use_widening": self.use_widening,
            "widening_delay": self.widening_delay,
            "use_narrowing": self.use_narrowing,
            "narrowing_iterations": self.narrowing_iterations,
            "epsilon": self.epsilon,
            "parallel": self.parallel,
            "n_workers": self.n_workers,
            "verbosity": self.verbosity,
            "severity": self.severity.to_dict(),
            "channel": self.channel.to_dict(),
            "instrumentation": self.instrumentation.to_dict(),
            "report": self.report.to_dict(),
            "profile": self.profile,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaintFlowConfig":
        return cls._from_flat_dict(data)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_analysis_config(self) -> "AnalysisConfig":
        from taintflow.core.types import AnalysisConfig
        return AnalysisConfig(
            b_max=self.b_max,
            alpha=self.alpha,
            max_iterations=self.max_iterations,
            use_widening=self.use_widening,
            widening_delay=self.widening_delay,
            use_narrowing=self.use_narrowing,
            narrowing_iterations=self.narrowing_iterations,
            epsilon=self.epsilon,
            track_provenance=True,
            parallel=self.parallel,
            n_workers=self.n_workers,
        )

    # -----------------------------------------------------------------------
    #  Presets
    # -----------------------------------------------------------------------

    @classmethod
    def quick_audit(cls) -> "TaintFlowConfig":
        """Fast preset: fewer iterations, heuristic capacity, no narrowing."""
        return cls(
            b_max=32.0,
            max_iterations=50,
            use_widening=True,
            widening_delay=1,
            use_narrowing=False,
            narrowing_iterations=0,
            epsilon=1e-6,
            verbosity=0,
            channel=ChannelCapacitySettings(
                tier_preference="heuristic",
                fallback_tier="bounding",
                use_cache=True,
            ),
            instrumentation=InstrumentationSettings(
                trace_depth=20,
                record_timing=False,
            ),
            report=ReportSettings(
                format="text",
                include_remediation=False,
                include_code_snippets=False,
            ),
            profile="quick",
        )

    @classmethod
    def thorough_audit(cls) -> "TaintFlowConfig":
        """Thorough preset: more iterations, analytic capacity, narrowing."""
        return cls(
            b_max=64.0,
            max_iterations=5000,
            use_widening=True,
            widening_delay=5,
            use_narrowing=True,
            narrowing_iterations=10,
            epsilon=1e-12,
            verbosity=2,
            channel=ChannelCapacitySettings(
                tier_preference="analytic",
                fallback_tier="sampling",
                sampling_n_samples=100_000,
                monte_carlo_iterations=10_000,
            ),
            instrumentation=InstrumentationSettings(
                trace_depth=100,
                record_timing=True,
                max_trace_events=2_000_000,
            ),
            report=ReportSettings(
                format="json",
                include_remediation=True,
                include_code_snippets=True,
                include_dag_visualization=True,
            ),
            profile="thorough",
        )

    @classmethod
    def ci_mode(cls) -> "TaintFlowConfig":
        """CI preset: machine-readable output, strict thresholds, fast."""
        return cls(
            b_max=64.0,
            max_iterations=200,
            use_widening=True,
            widening_delay=2,
            use_narrowing=True,
            narrowing_iterations=3,
            epsilon=1e-8,
            verbosity=0,
            severity=SeverityThresholds(negligible_max=0.5, warning_max=4.0),
            channel=ChannelCapacitySettings(
                tier_preference="analytic",
                fallback_tier="heuristic",
            ),
            report=ReportSettings(
                format="sarif",
                include_remediation=True,
                include_code_snippets=True,
                include_summary=True,
            ),
            profile="ci",
        )

    # -----------------------------------------------------------------------
    #  Repr
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TaintFlowConfig(B_max={self.b_max}, α={self.alpha}, "
            f"max_iter={self.max_iterations}, profile={self.profile!r})"
        )

    def summary(self) -> str:
        lines = [
            f"TaintFlow Configuration (profile={self.profile})",
            f"  B_max            = {self.b_max}",
            f"  alpha            = {self.alpha}",
            f"  max_iterations   = {self.max_iterations}",
            f"  widening         = {self.use_widening} (delay={self.widening_delay})",
            f"  narrowing        = {self.use_narrowing} (iters={self.narrowing_iterations})",
            f"  epsilon          = {self.epsilon}",
            f"  parallel         = {self.parallel} ({self.n_workers} workers)",
            f"  severity         = {self.severity}",
            f"  channel          = {self.channel}",
            f"  instrumentation  = {self.instrumentation}",
            f"  report           = {self.report}",
        ]
        return "\n".join(lines)


# ===================================================================
#  Helpers
# ===================================================================


def _parse_env_value(raw: str) -> Any:
    """Attempt to parse an environment variable value to a native type."""
    low = raw.lower()
    if low in ("true", "1", "yes"):
        return True
    if low in ("false", "0", "no"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _default_config_paths() -> list[Path]:
    """Return candidate config file paths in search order."""
    candidates: list[Path] = []
    cwd = Path.cwd()
    candidates.append(cwd / "taintflow.toml")
    candidates.append(cwd / "taintflow.json")
    candidates.append(cwd / ".taintflow.toml")
    candidates.append(cwd / ".taintflow.json")
    home = Path.home()
    candidates.append(home / ".config" / "taintflow" / "config.toml")
    candidates.append(home / ".config" / "taintflow" / "config.json")
    return candidates


def _merge_dataclass(base: Any, override: Any, default: Any) -> Any:
    """Merge two dataclass instances: override wins when it differs from default."""
    if not hasattr(base, "__dataclass_fields__"):
        return override
    merged_kwargs: dict[str, Any] = {}
    for fld_name in base.__dataclass_fields__:
        base_val = getattr(base, fld_name)
        over_val = getattr(override, fld_name)
        def_val = getattr(default, fld_name)
        merged_kwargs[fld_name] = over_val if over_val != def_val else base_val
    return type(base)(**merged_kwargs)
