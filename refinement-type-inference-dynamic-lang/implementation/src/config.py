"""
Configuration module for Refinement Type Inference engine.

Provides hierarchical configuration for static analysis of dynamically-typed
languages (Python, TypeScript) using refinement types and counterexample-guided
contract discovery (CEGAR-style abstract interpretation).

Configuration sources (highest to lowest priority):
  1. CLI arguments
  2. Environment variables (REFTYPE_*)
  3. Project-level reftype.toml / reftype.yaml
  4. Built-in profiles (fast, thorough, ci)
  5. Hard-coded defaults
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass, field, asdict, replace
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TargetLanguage(Enum):
    """Languages supported by the analysis engine."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"


class PythonVersion(Enum):
    """Python version targets for type stub resolution."""
    PY38 = "3.8"
    PY39 = "3.9"
    PY310 = "3.10"
    PY311 = "3.11"
    PY312 = "3.12"
    PY313 = "3.13"


class AbstractDomain(Enum):
    """Numeric abstract domains available for widening/narrowing."""
    INTERVALS = "intervals"
    OCTAGON = "octagon"
    POLYHEDRA = "polyhedra"
    CONGRUENCE = "congruence"
    PENTAGONS = "pentagons"


class WideningStrategy(Enum):
    """Widening strategies for fixpoint computation."""
    STANDARD = "standard"
    DELAYED = "delayed"
    THRESHOLD = "threshold"


class NarrowingStrategy(Enum):
    """Narrowing strategies applied after widening reaches a fixpoint."""
    NONE = "none"
    STANDARD = "standard"
    GUIDED = "guided"


class OutputFormat(Enum):
    """Supported output formats."""
    SARIF = "sarif"
    STUBS = "stubs"
    HTML = "html"
    SMTLIB = "smtlib"
    JSON = "json"
    TEXT = "text"


class BugClass(Enum):
    """Categories of bugs the engine can detect."""
    OUT_OF_BOUNDS = "oob"
    NULL_DEREF = "null"
    DIVISION_BY_ZERO = "divzero"
    TYPE_CONFUSION = "type_confusion"
    UNCHECKED_CAST = "unchecked_cast"
    UNREACHABLE_CODE = "unreachable"
    TAINTED_FLOW = "taint"


class Severity(Enum):
    """Finding severity levels, aligned with SARIF."""
    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"
    NONE = "none"


class AnalysisProfile(Enum):
    """Pre-defined analysis profiles."""
    FAST = "fast"
    THOROUGH = "thorough"
    CI = "ci"
    CUSTOM = "custom"


class CacheBackend(Enum):
    """Backends for caching analysis summaries."""
    FILESYSTEM = "filesystem"
    SQLITE = "sqlite"
    NONE = "none"


class IncrementalStrategy(Enum):
    """How incremental analysis decides what to re-analyse."""
    FILE_HASH = "file_hash"
    GIT_DIFF = "git_diff"
    DEPENDENCY_GRAPH = "dependency_graph"


# ---------------------------------------------------------------------------
# Sub-configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BugClassConfig:
    """Toggle and tune individual bug-class detectors.

    Each flag enables or disables the corresponding checker.  The
    ``severity_override`` map lets users raise or lower the severity of
    individual classes for their project.
    """
    oob: bool = True
    null: bool = True
    divzero: bool = True
    type_confusion: bool = True
    unchecked_cast: bool = False
    unreachable: bool = False
    taint: bool = False
    severity_overrides: Dict[str, Severity] = field(default_factory=dict)

    @property
    def enabled_classes(self) -> FrozenSet[BugClass]:
        """Return the set of enabled bug classes."""
        mapping: Dict[str, BugClass] = {
            "oob": BugClass.OUT_OF_BOUNDS,
            "null": BugClass.NULL_DEREF,
            "divzero": BugClass.DIVISION_BY_ZERO,
            "type_confusion": BugClass.TYPE_CONFUSION,
            "unchecked_cast": BugClass.UNCHECKED_CAST,
            "unreachable": BugClass.UNREACHABLE_CODE,
            "taint": BugClass.TAINTED_FLOW,
        }
        enabled: Set[BugClass] = set()
        for attr, cls in mapping.items():
            if getattr(self, attr):
                enabled.add(cls)
        return frozenset(enabled)

    def severity_for(self, bug_class: BugClass) -> Severity:
        """Return the effective severity for *bug_class*."""
        override = self.severity_overrides.get(bug_class.value)
        if override is not None:
            return override
        if bug_class in (BugClass.OUT_OF_BOUNDS, BugClass.NULL_DEREF,
                         BugClass.DIVISION_BY_ZERO, BugClass.TYPE_CONFUSION):
            return Severity.ERROR
        return Severity.WARNING


@dataclass(frozen=True)
class OutputConfig:
    """Controls which output artefacts the engine produces."""
    sarif: bool = True
    stubs: bool = False
    html: bool = False
    smtlib: bool = False
    json_summary: bool = False
    text: bool = True
    sarif_file: Path = Path("reftype-results.sarif")
    stubs_dir: Path = Path(".reftype/stubs")
    html_dir: Path = Path(".reftype/html")
    smtlib_dir: Path = Path(".reftype/smt")
    json_file: Path = Path("reftype-results.json")
    pretty_print: bool = True
    sarif_version: str = "2.1.0"
    include_provenance: bool = True

    @property
    def enabled_formats(self) -> FrozenSet[OutputFormat]:
        flags: Dict[str, OutputFormat] = {
            "sarif": OutputFormat.SARIF,
            "stubs": OutputFormat.STUBS,
            "html": OutputFormat.HTML,
            "smtlib": OutputFormat.SMTLIB,
            "json_summary": OutputFormat.JSON,
            "text": OutputFormat.TEXT,
        }
        return frozenset(fmt for attr, fmt in flags.items() if getattr(self, attr))


@dataclass(frozen=True)
class PerformanceConfig:
    """Tuning knobs that affect speed vs. precision trade-offs."""
    per_function_timeout_s: float = 30.0
    max_cegar_iterations: int = 50
    max_call_depth: int = 8
    max_loop_unrolls: int = 3
    parallel_workers: int = 0  # 0 = auto (os.cpu_count)
    chunk_size: int = 64
    use_summary_cache: bool = True
    lazy_import_resolution: bool = True
    skip_unreachable_functions: bool = True
    max_disjuncts: int = 16
    widening_delay: int = 3
    max_smt_query_size: int = 100_000
    smt_timeout_ms: int = 5_000

    @property
    def effective_workers(self) -> int:
        if self.parallel_workers > 0:
            return self.parallel_workers
        return max(1, (os.cpu_count() or 1))


@dataclass(frozen=True)
class DomainConfig:
    """Abstract domain selection and composition."""
    numeric_domain: AbstractDomain = AbstractDomain.INTERVALS
    enable_octagon: bool = False
    enable_polyhedra: bool = False
    enable_congruence: bool = False
    widening_strategy: WideningStrategy = WideningStrategy.DELAYED
    narrowing_strategy: NarrowingStrategy = NarrowingStrategy.STANDARD
    widening_thresholds: Tuple[int, ...] = (0, 1, 16, 256, 65536)
    join_precision: int = 2
    reduced_product: bool = False
    string_domain_max_length: int = 64
    collection_domain_max_size: int = 32

    @property
    def active_numeric_domains(self) -> FrozenSet[AbstractDomain]:
        """Return the set of active numeric abstract domains."""
        domains: Set[AbstractDomain] = {self.numeric_domain}
        if self.enable_octagon:
            domains.add(AbstractDomain.OCTAGON)
        if self.enable_polyhedra:
            domains.add(AbstractDomain.POLYHEDRA)
        if self.enable_congruence:
            domains.add(AbstractDomain.CONGRUENCE)
        return frozenset(domains)


@dataclass(frozen=True)
class IncrementalConfig:
    """Settings for incremental (re-)analysis."""
    enabled: bool = False
    strategy: IncrementalStrategy = IncrementalStrategy.FILE_HASH
    cache_dir: Path = Path(".reftype/cache")
    cache_backend: CacheBackend = CacheBackend.FILESYSTEM
    max_cache_size_mb: int = 512
    invalidation_depth: int = 2
    track_transitive_deps: bool = True
    git_base_ref: Optional[str] = None
    store_summaries: bool = True
    summary_ttl_hours: int = 168  # one week

    def resolve_cache_dir(self, project_root: Path) -> Path:
        if self.cache_dir.is_absolute():
            return self.cache_dir
        return project_root / self.cache_dir


# ---------------------------------------------------------------------------
# Stdlib model paths
# ---------------------------------------------------------------------------

_DEFAULT_STDLIB_ROOTS: Dict[TargetLanguage, List[str]] = {
    TargetLanguage.PYTHON: [
        "models/python/stdlib",
        "models/python/typeshed",
        "models/python/builtins.pyi",
    ],
    TargetLanguage.TYPESCRIPT: [
        "models/typescript/lib",
        "models/typescript/dom.d.ts",
    ],
}

_ENV_PREFIX = "REFTYPE_"


def _default_stub_dirs() -> List[Path]:
    return [Path("typings"), Path(".reftype/stubs")]


def _default_exclude_patterns() -> List[str]:
    return [
        "__pycache__",
        ".git",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        "*.egg-info",
        "venv",
        ".venv",
        "dist",
        "build",
    ]


# ---------------------------------------------------------------------------
# Top-level AnalysisConfig
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """Complete configuration for a single analysis run.

    Instances are typically built via :class:`ConfigMerger` which layers
    CLI flags on top of file-based settings and built-in defaults.
    """

    # -- target specification ------------------------------------------------
    target_paths: List[Path] = field(default_factory=lambda: [Path(".")])
    language: TargetLanguage = TargetLanguage.PYTHON
    python_version: PythonVersion = PythonVersion.PY312
    exclude_patterns: List[str] = field(default_factory=_default_exclude_patterns)
    include_patterns: List[str] = field(default_factory=list)
    follow_imports: bool = True
    stub_dirs: List[Path] = field(default_factory=_default_stub_dirs)
    stdlib_model_paths: List[Path] = field(default_factory=list)

    # -- analysis tuning -----------------------------------------------------
    confidence_threshold: float = 0.7
    profile: AnalysisProfile = AnalysisProfile.CUSTOM
    bug_classes: BugClassConfig = field(default_factory=BugClassConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    incremental: IncrementalConfig = field(default_factory=IncrementalConfig)

    # -- output --------------------------------------------------------------
    output: OutputConfig = field(default_factory=OutputConfig)
    verbose: int = 0
    quiet: bool = False
    color: bool = True

    # -- misc ----------------------------------------------------------------
    project_root: Path = field(default_factory=Path.cwd)
    config_file: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.stdlib_model_paths:
            base = Path(__file__).resolve().parent.parent
            raw = _DEFAULT_STDLIB_ROOTS.get(self.language, [])
            self.stdlib_model_paths = [base / p for p in raw]

    # -- convenience helpers -------------------------------------------------

    def should_analyse(self, path: Path) -> bool:
        """Return *True* if *path* should be included in analysis."""
        rel = str(path)
        for pat in self.exclude_patterns:
            if pat in rel:
                return False
        if self.include_patterns:
            return any(pat in rel for pat in self.include_patterns)
        return True

    def effective_timeout(self) -> float:
        """Per-function timeout factoring in the active profile."""
        return self.performance.per_function_timeout_s

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entire config tree to a plain dict."""
        return _serialise(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisConfig":
        """Reconstruct an AnalysisConfig from a plain dict."""
        return _deserialise_analysis_config(data)

    @classmethod
    def from_json(cls, text: str) -> "AnalysisConfig":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialise(obj: Any) -> Any:
    """Recursively convert dataclass / enum / Path trees to JSON-safe dicts."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, frozenset):
        return sorted(_serialise(v) for v in obj)
    if isinstance(obj, (set,)):
        return sorted(_serialise(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _serialise(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialise(v) for k, v in asdict(obj).items()}
    return obj


def _enum_lookup(enum_cls: type, value: Any) -> Any:
    """Best-effort enum member lookup."""
    if isinstance(value, enum_cls):
        return value
    for member in enum_cls:
        if member.value == value or member.name == value:
            return member
    raise ValueError(f"Unknown {enum_cls.__name__} value: {value!r}")


def _deserialise_bug_class_config(d: Dict[str, Any]) -> BugClassConfig:
    overrides_raw = d.get("severity_overrides", {})
    overrides = {k: _enum_lookup(Severity, v) for k, v in overrides_raw.items()}
    return BugClassConfig(
        oob=d.get("oob", True),
        null=d.get("null", True),
        divzero=d.get("divzero", True),
        type_confusion=d.get("type_confusion", True),
        unchecked_cast=d.get("unchecked_cast", False),
        unreachable=d.get("unreachable", False),
        taint=d.get("taint", False),
        severity_overrides=overrides,
    )


def _deserialise_output_config(d: Dict[str, Any]) -> OutputConfig:
    return OutputConfig(
        sarif=d.get("sarif", True),
        stubs=d.get("stubs", False),
        html=d.get("html", False),
        smtlib=d.get("smtlib", False),
        json_summary=d.get("json_summary", False),
        text=d.get("text", True),
        sarif_file=Path(d["sarif_file"]) if "sarif_file" in d else OutputConfig.sarif_file,
        stubs_dir=Path(d["stubs_dir"]) if "stubs_dir" in d else OutputConfig.stubs_dir,
        html_dir=Path(d["html_dir"]) if "html_dir" in d else OutputConfig.html_dir,
        smtlib_dir=Path(d["smtlib_dir"]) if "smtlib_dir" in d else OutputConfig.smtlib_dir,
        json_file=Path(d["json_file"]) if "json_file" in d else OutputConfig.json_file,
        pretty_print=d.get("pretty_print", True),
        sarif_version=d.get("sarif_version", "2.1.0"),
        include_provenance=d.get("include_provenance", True),
    )


def _deserialise_performance_config(d: Dict[str, Any]) -> PerformanceConfig:
    return PerformanceConfig(
        per_function_timeout_s=float(d.get("per_function_timeout_s", 30.0)),
        max_cegar_iterations=int(d.get("max_cegar_iterations", 50)),
        max_call_depth=int(d.get("max_call_depth", 8)),
        max_loop_unrolls=int(d.get("max_loop_unrolls", 3)),
        parallel_workers=int(d.get("parallel_workers", 0)),
        chunk_size=int(d.get("chunk_size", 64)),
        use_summary_cache=d.get("use_summary_cache", True),
        lazy_import_resolution=d.get("lazy_import_resolution", True),
        skip_unreachable_functions=d.get("skip_unreachable_functions", True),
        max_disjuncts=int(d.get("max_disjuncts", 16)),
        widening_delay=int(d.get("widening_delay", 3)),
        max_smt_query_size=int(d.get("max_smt_query_size", 100_000)),
        smt_timeout_ms=int(d.get("smt_timeout_ms", 5_000)),
    )


def _deserialise_domain_config(d: Dict[str, Any]) -> DomainConfig:
    thresholds = d.get("widening_thresholds", (0, 1, 16, 256, 65536))
    return DomainConfig(
        numeric_domain=_enum_lookup(AbstractDomain, d.get("numeric_domain", "intervals")),
        enable_octagon=d.get("enable_octagon", False),
        enable_polyhedra=d.get("enable_polyhedra", False),
        enable_congruence=d.get("enable_congruence", False),
        widening_strategy=_enum_lookup(WideningStrategy, d.get("widening_strategy", "delayed")),
        narrowing_strategy=_enum_lookup(NarrowingStrategy, d.get("narrowing_strategy", "standard")),
        widening_thresholds=tuple(int(x) for x in thresholds),
        join_precision=int(d.get("join_precision", 2)),
        reduced_product=d.get("reduced_product", False),
        string_domain_max_length=int(d.get("string_domain_max_length", 64)),
        collection_domain_max_size=int(d.get("collection_domain_max_size", 32)),
    )


def _deserialise_incremental_config(d: Dict[str, Any]) -> IncrementalConfig:
    return IncrementalConfig(
        enabled=d.get("enabled", False),
        strategy=_enum_lookup(IncrementalStrategy, d.get("strategy", "file_hash")),
        cache_dir=Path(d.get("cache_dir", ".reftype/cache")),
        cache_backend=_enum_lookup(CacheBackend, d.get("cache_backend", "filesystem")),
        max_cache_size_mb=int(d.get("max_cache_size_mb", 512)),
        invalidation_depth=int(d.get("invalidation_depth", 2)),
        track_transitive_deps=d.get("track_transitive_deps", True),
        git_base_ref=d.get("git_base_ref"),
        store_summaries=d.get("store_summaries", True),
        summary_ttl_hours=int(d.get("summary_ttl_hours", 168)),
    )


def _deserialise_analysis_config(d: Dict[str, Any]) -> AnalysisConfig:
    cfg = AnalysisConfig(
        target_paths=[Path(p) for p in d.get("target_paths", ["."])],
        language=_enum_lookup(TargetLanguage, d.get("language", "python")),
        python_version=_enum_lookup(PythonVersion, d.get("python_version", "3.12")),
        exclude_patterns=d.get("exclude_patterns", _default_exclude_patterns()),
        include_patterns=d.get("include_patterns", []),
        follow_imports=d.get("follow_imports", True),
        stub_dirs=[Path(p) for p in d.get("stub_dirs", ["typings", ".reftype/stubs"])],
        stdlib_model_paths=[Path(p) for p in d.get("stdlib_model_paths", [])],
        confidence_threshold=float(d.get("confidence_threshold", 0.7)),
        profile=_enum_lookup(AnalysisProfile, d.get("profile", "custom")),
        bug_classes=_deserialise_bug_class_config(d.get("bug_classes", {})),
        domain=_deserialise_domain_config(d.get("domain", {})),
        performance=_deserialise_performance_config(d.get("performance", {})),
        incremental=_deserialise_incremental_config(d.get("incremental", {})),
        output=_deserialise_output_config(d.get("output", {})),
        verbose=int(d.get("verbose", 0)),
        quiet=d.get("quiet", False),
        color=d.get("color", True),
        project_root=Path(d.get("project_root", ".")),
        config_file=Path(d["config_file"]) if d.get("config_file") else None,
    )
    return cfg


# ---------------------------------------------------------------------------
# Config file loaders
# ---------------------------------------------------------------------------

class TOMLConfigLoader:
    """Load configuration from a ``reftype.toml`` file.

    Uses the stdlib ``tomllib`` (3.11+) with a fallback to ``tomli``.
    If neither is available the loader raises :class:`ImportError`.
    """

    @staticmethod
    def _load_toml(path: Path) -> Dict[str, Any]:
        import sys
        text = path.read_text(encoding="utf-8")
        if sys.version_info >= (3, 11):
            import tomllib  # type: ignore[import-not-found]
            return tomllib.loads(text)
        try:
            import tomli  # type: ignore[import-not-found]
            return tomli.loads(text)
        except ImportError:
            raise ImportError(
                "TOML support requires Python ≥ 3.11 or the 'tomli' package."
            )

    @classmethod
    def load(cls, path: Path) -> AnalysisConfig:
        """Parse *path* and return a fully hydrated :class:`AnalysisConfig`."""
        raw = cls._load_toml(path)
        tool_section = raw.get("tool", {}).get("reftype", raw)
        cfg = _deserialise_analysis_config(tool_section)
        cfg.config_file = path
        return cfg

    @classmethod
    def discover(cls, start: Path) -> Optional[Path]:
        """Walk up from *start* looking for ``reftype.toml``."""
        current = start.resolve()
        for _ in range(64):
            candidate = current / "reftype.toml"
            if candidate.is_file():
                return candidate
            pyproject = current / "pyproject.toml"
            if pyproject.is_file():
                try:
                    data = cls._load_toml(pyproject)
                    if "tool" in data and "reftype" in data["tool"]:
                        return pyproject
                except Exception:
                    pass
            parent = current.parent
            if parent == current:
                break
            current = parent
        return None


class YAMLConfigLoader:
    """Load configuration from a ``reftype.yaml`` or ``reftype.yml`` file.

    Requires the ``pyyaml`` package.
    """

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "YAML config support requires the 'pyyaml' package."
            )
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a mapping at top level of {path}")
        return data

    @classmethod
    def load(cls, path: Path) -> AnalysisConfig:
        raw = cls._load_yaml(path)
        cfg = _deserialise_analysis_config(raw)
        cfg.config_file = path
        return cfg

    @classmethod
    def discover(cls, start: Path) -> Optional[Path]:
        """Walk up from *start* looking for ``reftype.yaml`` / ``.yml``."""
        current = start.resolve()
        for _ in range(64):
            for name in ("reftype.yaml", "reftype.yml"):
                candidate = current / name
                if candidate.is_file():
                    return candidate
            parent = current.parent
            if parent == current:
                break
            current = parent
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ConfigValidationError(Exception):
    """Raised when a configuration fails validation."""

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__(
            "Configuration validation failed:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )


class ConfigValidator:
    """Validate an :class:`AnalysisConfig` for internal consistency."""

    def __init__(self) -> None:
        self._errors: List[str] = []

    def validate(self, cfg: AnalysisConfig) -> List[str]:
        """Return a list of validation error strings (empty == valid)."""
        self._errors = []
        self._check_confidence(cfg)
        self._check_timeouts(cfg)
        self._check_cegar(cfg)
        self._check_paths(cfg)
        self._check_domains(cfg)
        self._check_incremental(cfg)
        self._check_output(cfg)
        self._check_workers(cfg)
        return list(self._errors)

    def validate_or_raise(self, cfg: AnalysisConfig) -> None:
        """Like :meth:`validate` but raises on failure."""
        errors = self.validate(cfg)
        if errors:
            raise ConfigValidationError(errors)

    # -- individual checks ---------------------------------------------------

    def _check_confidence(self, cfg: AnalysisConfig) -> None:
        if not (0.0 <= cfg.confidence_threshold <= 1.0):
            self._errors.append(
                f"confidence_threshold must be in [0, 1], got {cfg.confidence_threshold}"
            )

    def _check_timeouts(self, cfg: AnalysisConfig) -> None:
        t = cfg.performance.per_function_timeout_s
        if t <= 0:
            self._errors.append(
                f"per_function_timeout_s must be positive, got {t}"
            )
        if cfg.performance.smt_timeout_ms <= 0:
            self._errors.append("smt_timeout_ms must be positive")

    def _check_cegar(self, cfg: AnalysisConfig) -> None:
        n = cfg.performance.max_cegar_iterations
        if n < 1:
            self._errors.append(
                f"max_cegar_iterations must be ≥ 1, got {n}"
            )
        if cfg.performance.max_call_depth < 1:
            self._errors.append("max_call_depth must be ≥ 1")

    def _check_paths(self, cfg: AnalysisConfig) -> None:
        if not cfg.target_paths:
            self._errors.append("At least one target path is required")
        for p in cfg.target_paths:
            resolved = (cfg.project_root / p) if not p.is_absolute() else p
            if not resolved.exists():
                self._errors.append(f"Target path does not exist: {resolved}")

    def _check_domains(self, cfg: AnalysisConfig) -> None:
        if cfg.domain.enable_polyhedra and not cfg.domain.enable_octagon:
            self._errors.append(
                "Polyhedra domain requires octagon to be enabled as well "
                "(for reduced product stability)"
            )
        if cfg.domain.join_precision < 0:
            self._errors.append("join_precision must be non-negative")

    def _check_incremental(self, cfg: AnalysisConfig) -> None:
        inc = cfg.incremental
        if inc.enabled and inc.strategy == IncrementalStrategy.GIT_DIFF:
            if inc.git_base_ref is None:
                self._errors.append(
                    "git_diff incremental strategy requires git_base_ref"
                )
        if inc.max_cache_size_mb < 1:
            self._errors.append("max_cache_size_mb must be ≥ 1")

    def _check_output(self, cfg: AnalysisConfig) -> None:
        if not cfg.output.enabled_formats:
            self._errors.append("At least one output format must be enabled")

    def _check_workers(self, cfg: AnalysisConfig) -> None:
        w = cfg.performance.parallel_workers
        if w < 0:
            self._errors.append("parallel_workers must be ≥ 0 (0 = auto)")


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

def _apply_env_overrides(cfg: AnalysisConfig) -> AnalysisConfig:
    """Apply ``REFTYPE_*`` environment variables on top of *cfg*.

    Supported variables
    -------------------
    REFTYPE_CONFIDENCE    – float, overrides confidence_threshold
    REFTYPE_TIMEOUT       – float, per-function timeout in seconds
    REFTYPE_WORKERS       – int, parallel workers (0 = auto)
    REFTYPE_PROFILE       – str, analysis profile name
    REFTYPE_VERBOSE       – int, verbosity level
    REFTYPE_CACHE_DIR     – str, cache directory path
    REFTYPE_COLOR         – 0/1, enable coloured output
    REFTYPE_PYTHON_VERSION – str, e.g. "3.12"
    REFTYPE_EXCLUDE       – comma-separated exclude patterns
    REFTYPE_MAX_CEGAR     – int, max contract discovery iterations
    REFTYPE_OCTAGON       – 0/1, enable octagon domain
    """

    def _env(name: str) -> Optional[str]:
        return os.environ.get(f"{_ENV_PREFIX}{name}")

    val = _env("CONFIDENCE")
    if val is not None:
        cfg.confidence_threshold = float(val)

    val = _env("TIMEOUT")
    if val is not None:
        cfg.performance = replace(
            cfg.performance, per_function_timeout_s=float(val)
        )

    val = _env("WORKERS")
    if val is not None:
        cfg.performance = replace(
            cfg.performance, parallel_workers=int(val)
        )

    val = _env("PROFILE")
    if val is not None:
        cfg.profile = _enum_lookup(AnalysisProfile, val)

    val = _env("VERBOSE")
    if val is not None:
        cfg.verbose = int(val)

    val = _env("CACHE_DIR")
    if val is not None:
        cfg.incremental = replace(
            cfg.incremental, cache_dir=Path(val)
        )

    val = _env("COLOR")
    if val is not None:
        cfg.color = val not in ("0", "false", "no", "")

    val = _env("PYTHON_VERSION")
    if val is not None:
        cfg.python_version = _enum_lookup(PythonVersion, val)

    val = _env("EXCLUDE")
    if val is not None:
        cfg.exclude_patterns = [p.strip() for p in val.split(",") if p.strip()]

    val = _env("MAX_CEGAR")
    if val is not None:
        cfg.performance = replace(
            cfg.performance, max_cegar_iterations=int(val)
        )

    val = _env("OCTAGON")
    if val is not None:
        cfg.domain = replace(
            cfg.domain, enable_octagon=val not in ("0", "false", "no", "")
        )

    return cfg


# ---------------------------------------------------------------------------
# Predefined profiles
# ---------------------------------------------------------------------------

_PROFILE_OVERRIDES: Dict[AnalysisProfile, Dict[str, Any]] = {
    AnalysisProfile.FAST: {
        "performance": {
            "per_function_timeout_s": 5.0,
            "max_cegar_iterations": 10,
            "max_call_depth": 4,
            "max_loop_unrolls": 1,
            "max_disjuncts": 4,
            "smt_timeout_ms": 1_000,
            "skip_unreachable_functions": True,
        },
        "domain": {
            "numeric_domain": "intervals",
            "enable_octagon": False,
            "widening_strategy": "standard",
            "narrowing_strategy": "none",
        },
        "confidence_threshold": 0.85,
        "bug_classes": {
            "oob": True,
            "null": True,
            "divzero": True,
            "type_confusion": True,
            "unchecked_cast": False,
            "unreachable": False,
            "taint": False,
        },
    },
    AnalysisProfile.THOROUGH: {
        "performance": {
            "per_function_timeout_s": 120.0,
            "max_cegar_iterations": 200,
            "max_call_depth": 16,
            "max_loop_unrolls": 8,
            "max_disjuncts": 64,
            "smt_timeout_ms": 30_000,
            "skip_unreachable_functions": False,
        },
        "domain": {
            "numeric_domain": "octagon",
            "enable_octagon": True,
            "enable_congruence": True,
            "widening_strategy": "threshold",
            "narrowing_strategy": "guided",
            "reduced_product": True,
        },
        "confidence_threshold": 0.5,
        "bug_classes": {
            "oob": True,
            "null": True,
            "divzero": True,
            "type_confusion": True,
            "unchecked_cast": True,
            "unreachable": True,
            "taint": True,
        },
    },
    AnalysisProfile.CI: {
        "performance": {
            "per_function_timeout_s": 15.0,
            "max_cegar_iterations": 30,
            "max_call_depth": 6,
            "max_loop_unrolls": 2,
            "max_disjuncts": 8,
            "smt_timeout_ms": 3_000,
        },
        "domain": {
            "numeric_domain": "intervals",
            "enable_octagon": False,
            "widening_strategy": "delayed",
            "narrowing_strategy": "standard",
        },
        "confidence_threshold": 0.8,
        "output": {
            "sarif": True,
            "text": True,
            "html": False,
            "stubs": False,
        },
        "incremental": {
            "enabled": True,
            "strategy": "git_diff",
            "git_base_ref": "origin/main",
        },
        "color": False,
        "quiet": True,
    },
}


def _apply_profile(cfg: AnalysisConfig) -> AnalysisConfig:
    """Apply profile overrides if *cfg.profile* is not CUSTOM."""
    overrides = _PROFILE_OVERRIDES.get(cfg.profile)
    if overrides is None:
        return cfg
    merged = _deserialise_analysis_config({
        **cfg.to_dict(),
        **overrides,
    })
    merged.profile = cfg.profile
    merged.config_file = cfg.config_file
    return merged


# ---------------------------------------------------------------------------
# ConfigMerger – layers CLI ▸ env ▸ file ▸ profile ▸ defaults
# ---------------------------------------------------------------------------

class ConfigMerger:
    """Merge multiple configuration sources into one :class:`AnalysisConfig`.

    Typical usage::

        merger = ConfigMerger()
        cfg = merger.build(
            cli_overrides={"confidence_threshold": 0.9, "verbose": 2},
            config_path=Path("reftype.toml"),
            profile=AnalysisProfile.CI,
        )
    """

    def build(
        self,
        cli_overrides: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path] = None,
        profile: Optional[AnalysisProfile] = None,
        project_root: Optional[Path] = None,
    ) -> AnalysisConfig:
        """Build a fully resolved configuration.

        Parameters
        ----------
        cli_overrides:
            Raw key/value pairs from CLI argument parsing.
        config_path:
            Explicit path to a config file.  ``None`` triggers auto-discovery.
        profile:
            Override the analysis profile.
        project_root:
            Override the project root directory.
        """
        root = (project_root or Path.cwd()).resolve()

        # 1. Start from defaults
        cfg = AnalysisConfig(project_root=root)

        # 2. Load file-based config (toml or yaml)
        file_cfg = self._load_file(config_path, root)
        if file_cfg is not None:
            cfg = file_cfg
            cfg.project_root = root

        # 3. Apply profile
        if profile is not None:
            cfg.profile = profile
        cfg = _apply_profile(cfg)

        # 4. Apply environment variable overrides
        cfg = _apply_env_overrides(cfg)

        # 5. Apply CLI overrides (highest priority)
        if cli_overrides:
            cfg = self._apply_cli(cfg, cli_overrides)

        return cfg

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _load_file(
        explicit: Optional[Path], root: Path
    ) -> Optional[AnalysisConfig]:
        if explicit is not None:
            return _load_config_file(explicit)
        # Auto-discover
        toml_path = TOMLConfigLoader.discover(root)
        if toml_path is not None:
            return TOMLConfigLoader.load(toml_path)
        yaml_path = YAMLConfigLoader.discover(root)
        if yaml_path is not None:
            return YAMLConfigLoader.load(yaml_path)
        return None

    @staticmethod
    def _apply_cli(
        cfg: AnalysisConfig, overrides: Dict[str, Any]
    ) -> AnalysisConfig:
        """Overlay flat CLI key=value pairs onto *cfg*."""
        simple_fields = {
            "confidence_threshold", "verbose", "quiet", "color",
            "follow_imports",
        }
        for key in simple_fields:
            if key in overrides:
                setattr(cfg, key, overrides[key])

        if "target_paths" in overrides:
            cfg.target_paths = [Path(p) for p in overrides["target_paths"]]

        if "exclude_patterns" in overrides:
            cfg.exclude_patterns = list(overrides["exclude_patterns"])

        if "language" in overrides:
            cfg.language = _enum_lookup(TargetLanguage, overrides["language"])

        if "python_version" in overrides:
            cfg.python_version = _enum_lookup(
                PythonVersion, overrides["python_version"]
            )

        # Nested performance overrides
        perf_keys = {
            "per_function_timeout_s", "max_cegar_iterations",
            "max_call_depth", "parallel_workers",
        }
        perf_updates: Dict[str, Any] = {}
        for k in perf_keys:
            if k in overrides:
                perf_updates[k] = overrides[k]
        if perf_updates:
            cfg.performance = replace(cfg.performance, **perf_updates)

        # Domain overrides
        if "enable_octagon" in overrides:
            cfg.domain = replace(
                cfg.domain, enable_octagon=bool(overrides["enable_octagon"])
            )

        # Output overrides
        if "sarif_file" in overrides:
            cfg.output = replace(
                cfg.output, sarif_file=Path(overrides["sarif_file"])
            )

        return cfg


# ---------------------------------------------------------------------------
# Dispatcher for loading config files by extension
# ---------------------------------------------------------------------------

def _load_config_file(path: Path) -> AnalysisConfig:
    """Load a config from *path*, selecting loader by file extension."""
    suffix = path.suffix.lower()
    if suffix == ".toml":
        return TOMLConfigLoader.load(path)
    if suffix in (".yaml", ".yml"):
        return YAMLConfigLoader.load(path)
    if suffix == ".json":
        text = path.read_text(encoding="utf-8")
        return AnalysisConfig.from_json(text)
    raise ValueError(f"Unsupported config file format: {suffix}")


# ---------------------------------------------------------------------------
# Quick-start factory functions
# ---------------------------------------------------------------------------

def fast_config(**overrides: Any) -> AnalysisConfig:
    """Return a config tuned for speed (low timeouts, simple domains)."""
    merger = ConfigMerger()
    return merger.build(
        cli_overrides=overrides, profile=AnalysisProfile.FAST
    )


def thorough_config(**overrides: Any) -> AnalysisConfig:
    """Return a config tuned for maximum precision."""
    merger = ConfigMerger()
    return merger.build(
        cli_overrides=overrides, profile=AnalysisProfile.THOROUGH
    )


def ci_config(**overrides: Any) -> AnalysisConfig:
    """Return a config tuned for CI pipelines (incremental, SARIF output)."""
    merger = ConfigMerger()
    return merger.build(
        cli_overrides=overrides, profile=AnalysisProfile.CI
    )


def default_config(**overrides: Any) -> AnalysisConfig:
    """Return the default config with optional overrides."""
    merger = ConfigMerger()
    return merger.build(cli_overrides=overrides)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def dump_config_summary(cfg: AnalysisConfig) -> str:
    """Return a human-readable multi-line summary of *cfg*."""
    lines = [
        "Refinement Type Inference – configuration summary",
        "=" * 50,
        f"  Language:          {cfg.language.value}",
        f"  Python version:    {cfg.python_version.value}",
        f"  Profile:           {cfg.profile.value}",
        f"  Targets:           {', '.join(str(p) for p in cfg.target_paths)}",
        f"  Confidence:        {cfg.confidence_threshold:.0%}",
        f"  Timeout/fn:        {cfg.performance.per_function_timeout_s:.1f}s",
        f"  Contract discovery iterations:  {cfg.performance.max_cegar_iterations}",
        f"  Workers:           {cfg.performance.effective_workers}",
        f"  Numeric domain:    {cfg.domain.numeric_domain.value}",
        f"  Octagon:           {'yes' if cfg.domain.enable_octagon else 'no'}",
        f"  Incremental:       {'yes' if cfg.incremental.enabled else 'no'}",
        f"  Bug classes:       {', '.join(b.value for b in sorted(cfg.bug_classes.enabled_classes, key=lambda b: b.value))}",
        f"  Output formats:    {', '.join(f.value for f in sorted(cfg.output.enabled_formats, key=lambda f: f.value))}",
    ]
    if cfg.config_file:
        lines.append(f"  Config file:       {cfg.config_file}")
    return "\n".join(lines)
