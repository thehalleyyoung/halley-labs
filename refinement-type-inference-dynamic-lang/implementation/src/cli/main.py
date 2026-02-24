from __future__ import annotations

"""
reftype.cli.main — Main CLI entry point and application logic.

Provides the ``ReftypeCliApp`` class with argparse-based subcommands for
analysing Python / TypeScript code via CEGAR-based refinement type inference.
"""

import argparse
import contextlib
import enum
import fnmatch
import glob as _glob
import hashlib
import io
import json
import logging
import multiprocessing
import os
import pathlib
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Type,
    Union,
)

# ---------------------------------------------------------------------------
# Locally-defined domain types (no imports from other project modules)
# ---------------------------------------------------------------------------

logger = logging.getLogger("reftype.cli")

_VERSION = "0.1.0"


class Severity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class Language(enum.Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    AUTO = "auto"


class OutputFormat(enum.Enum):
    PYI = "pyi"
    DTS = "dts"
    SARIF = "sarif"
    HTML = "html"
    JSON = "json"


@dataclass
class SourceLocation:
    file: str
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    def __str__(self) -> str:
        loc = f"{self.file}:{self.line}:{self.column}"
        if self.end_line is not None:
            loc += f"-{self.end_line}:{self.end_column}"
        return loc


@dataclass
class RefinementType:
    base: str
    predicate: str
    constraints: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.predicate:
            return f"{{{self.base} | {self.predicate}}}"
        return self.base


@dataclass
class Bug:
    id: str
    message: str
    severity: Severity
    location: SourceLocation
    category: str
    refinement_type: Optional[RefinementType] = None
    fix_suggestion: Optional[str] = None
    cegar_trace: Optional[List[str]] = None

    def fingerprint(self) -> str:
        raw = f"{self.category}:{self.location.file}:{self.message}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class FunctionContract:
    name: str
    file: str
    line: int
    params: Dict[str, RefinementType] = field(default_factory=dict)
    return_type: Optional[RefinementType] = None
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    file: str
    language: Language
    bugs: List[Bug] = field(default_factory=list)
    contracts: List[FunctionContract] = field(default_factory=list)
    duration_ms: float = 0.0
    functions_analyzed: int = 0
    cegar_iterations: int = 0
    timed_out: bool = False


@dataclass
class AnalysisSummary:
    total_files: int = 0
    total_functions: int = 0
    total_bugs: int = 0
    bugs_by_severity: Dict[str, int] = field(default_factory=dict)
    bugs_by_category: Dict[str, int] = field(default_factory=dict)
    total_contracts: int = 0
    total_cegar_iterations: int = 0
    duration_ms: float = 0.0
    files_timed_out: int = 0

    def merge(self, result: AnalysisResult) -> None:
        self.total_files += 1
        self.total_functions += result.functions_analyzed
        self.total_bugs += len(result.bugs)
        for b in result.bugs:
            sev = b.severity.value
            self.bugs_by_severity[sev] = self.bugs_by_severity.get(sev, 0) + 1
            self.bugs_by_category[b.category] = (
                self.bugs_by_category.get(b.category, 0) + 1
            )
        self.total_contracts += len(result.contracts)
        self.total_cegar_iterations += result.cegar_iterations
        self.duration_ms += result.duration_ms
        if result.timed_out:
            self.files_timed_out += 1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DomainSettings:
    interval_precision: int = 64
    octagon_enabled: bool = False
    polyhedra_enabled: bool = False
    widening_delay: int = 3
    narrowing_iterations: int = 2


@dataclass
class CegarSettings:
    max_iterations: int = 50
    refinement_strategy: str = "counterexample-guided"
    interpolation_enabled: bool = True
    predicate_abstraction: bool = True
    lazy_abstraction: bool = False


@dataclass
class IncrementalSettings:
    enabled: bool = True
    cache_dir: str = ".reftype-cache"
    hash_algorithm: str = "sha256"
    max_cache_age_hours: int = 168


@dataclass
class ParallelSettings:
    workers: int = 0  # 0 = auto (cpu_count)
    chunk_size: int = 4
    timeout_per_file: float = 60.0


@dataclass
class Configuration:
    """Complete configuration for a reftype analysis run."""

    paths: List[str] = field(default_factory=lambda: ["."])
    include_patterns: List[str] = field(default_factory=lambda: ["**/*.py", "**/*.ts"])
    exclude_patterns: List[str] = field(
        default_factory=lambda: [
            "node_modules/**",
            ".venv/**",
            "__pycache__/**",
            "*.egg-info/**",
            "dist/**",
            "build/**",
        ]
    )
    language: Language = Language.AUTO
    output_format: OutputFormat = OutputFormat.JSON
    output_file: Optional[str] = None
    verbosity: int = 0
    config_file: Optional[str] = None

    bug_classes: List[str] = field(
        default_factory=lambda: [
            "null-deref",
            "index-out-of-bounds",
            "division-by-zero",
            "type-mismatch",
            "unreachable-code",
            "unused-refinement",
        ]
    )
    min_severity: Severity = Severity.WARNING
    max_functions: int = 0  # 0 = unlimited
    timeout: float = 300.0

    domain: DomainSettings = field(default_factory=DomainSettings)
    cegar: CegarSettings = field(default_factory=CegarSettings)
    incremental: IncrementalSettings = field(default_factory=IncrementalSettings)
    parallel: ParallelSettings = field(default_factory=ParallelSettings)

    baseline_file: Optional[str] = None
    fail_on_new_bugs: bool = False
    telemetry_enabled: bool = False

    def effective_workers(self) -> int:
        if self.parallel.workers > 0:
            return self.parallel.workers
        return max(1, (os.cpu_count() or 1) - 1)


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------


class ConfigLoader:
    """Loads configuration from files, merges with CLI args, validates."""

    SEARCH_FILES = (".reftype.toml", "pyproject.toml", "package.json")

    def __init__(self) -> None:
        self._raw: Dict[str, Any] = {}

    # ------------------------------------------------------------------

    def find_config_file(self, start_dir: str = ".") -> Optional[str]:
        current = pathlib.Path(start_dir).resolve()
        while True:
            for name in self.SEARCH_FILES:
                candidate = current / name
                if candidate.is_file():
                    return str(candidate)
            parent = current.parent
            if parent == current:
                break
            current = parent
        return None

    # ------------------------------------------------------------------

    def load(self, path: Optional[str] = None) -> Dict[str, Any]:
        if path is None:
            path = self.find_config_file()
        if path is None:
            return {}
        p = pathlib.Path(path)
        if not p.exists():
            logger.warning("Config file %s does not exist", path)
            return {}
        if p.name == ".reftype.toml" or p.suffix == ".toml":
            return self._load_toml(p)
        if p.name == "pyproject.toml":
            data = self._load_toml(p)
            return data.get("tool", {}).get("reftype", {})
        if p.name == "package.json":
            return self._load_package_json(p)
        return {}

    def _load_toml(self, p: pathlib.Path) -> Dict[str, Any]:
        try:
            import tomllib  # 3.11+
        except ModuleNotFoundError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ModuleNotFoundError:
                return self._load_toml_fallback(p)
        with open(p, "rb") as fh:
            return tomllib.load(fh)

    @staticmethod
    def _load_toml_fallback(p: pathlib.Path) -> Dict[str, Any]:
        """Minimal TOML key=value parser when no library is available."""
        data: Dict[str, Any] = {}
        section = data
        with open(p) as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                hdr = re.match(r"^\[(.+)]$", line)
                if hdr:
                    parts = hdr.group(1).split(".")
                    section = data
                    for part in parts:
                        section = section.setdefault(part.strip(), {})
                    continue
                m = re.match(r'^(\w+)\s*=\s*"(.*)"\s*$', line)
                if m:
                    section[m.group(1)] = m.group(2)
                    continue
                m = re.match(r"^(\w+)\s*=\s*(\d+)\s*$", line)
                if m:
                    section[m.group(1)] = int(m.group(2))
                    continue
                m = re.match(r"^(\w+)\s*=\s*(true|false)\s*$", line, re.I)
                if m:
                    section[m.group(1)] = m.group(2).lower() == "true"
        return data

    @staticmethod
    def _load_package_json(p: pathlib.Path) -> Dict[str, Any]:
        with open(p) as fh:
            data = json.load(fh)
        return data.get("reftype", {})

    # ------------------------------------------------------------------

    def merge(self, file_cfg: Dict[str, Any], cli_args: argparse.Namespace) -> Configuration:
        cfg = Configuration()
        mapping: Dict[str, str] = {
            "paths": "paths",
            "include": "include_patterns",
            "exclude": "exclude_patterns",
            "language": "language",
            "format": "output_format",
            "output": "output_file",
            "verbose": "verbosity",
            "config": "config_file",
            "bug_classes": "bug_classes",
            "min_severity": "min_severity",
            "max_functions": "max_functions",
            "timeout": "timeout",
            "baseline": "baseline_file",
            "fail_on_new_bugs": "fail_on_new_bugs",
            "workers": "parallel.workers",
            "incremental": "incremental.enabled",
        }
        for src_key, dst_key in mapping.items():
            val = file_cfg.get(src_key)
            if val is not None:
                self._set_nested(cfg, dst_key, val)

        for src_key, dst_key in mapping.items():
            val = getattr(cli_args, src_key, None)
            if val is not None:
                self._set_nested(cfg, dst_key, val)

        domain_raw = file_cfg.get("domain", {})
        if isinstance(domain_raw, dict):
            for k, v in domain_raw.items():
                if hasattr(cfg.domain, k):
                    setattr(cfg.domain, k, v)

        cegar_raw = file_cfg.get("cegar", {})
        if isinstance(cegar_raw, dict):
            for k, v in cegar_raw.items():
                if hasattr(cfg.cegar, k):
                    setattr(cfg.cegar, k, v)

        return cfg

    @staticmethod
    def _set_nested(obj: Any, key: str, value: Any) -> None:
        parts = key.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        attr = parts[-1]
        field_val = getattr(obj, attr, None)
        if isinstance(field_val, enum.Enum):
            if isinstance(value, str):
                value = type(field_val)(value)
        setattr(obj, attr, value)

    # ------------------------------------------------------------------

    def validate(self, cfg: Configuration) -> List[str]:
        errors: List[str] = []
        if cfg.timeout <= 0:
            errors.append("timeout must be positive")
        if cfg.max_functions < 0:
            errors.append("max_functions must be >= 0")
        if cfg.parallel.workers < 0:
            errors.append("workers must be >= 0")
        if cfg.cegar.max_iterations < 1:
            errors.append("cegar.max_iterations must be >= 1")
        return errors


# ---------------------------------------------------------------------------
# LoggingSetup
# ---------------------------------------------------------------------------


class LoggingSetup:
    """Configure logging with verbosity levels."""

    LEVELS = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }

    @classmethod
    def configure(cls, verbosity: int = 0, log_file: Optional[str] = None) -> None:
        level = cls.LEVELS.get(min(verbosity, 2), logging.DEBUG)
        handlers: List[logging.Handler] = []

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(fmt)
        handlers.append(stream_handler)

        if log_file:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            handlers.append(fh)

        root = logging.getLogger("reftype")
        root.setLevel(level)
        for h in root.handlers[:]:
            root.removeHandler(h)
        for h in handlers:
            root.addHandler(h)


# ---------------------------------------------------------------------------
# ProgressReporter
# ---------------------------------------------------------------------------


class ProgressReporter:
    """Progress bars / status output with non-TTY fallback."""

    def __init__(self, total: int = 0, stream: TextIO = sys.stderr) -> None:
        self.total = total
        self.current = 0
        self.stream = stream
        self._is_tty = hasattr(stream, "isatty") and stream.isatty()
        self._start = time.monotonic()
        self._lock = threading.Lock()

    def update(self, n: int = 1, message: str = "") -> None:
        with self._lock:
            self.current += n
            if self._is_tty:
                self._render_bar(message)
            elif message:
                self.stream.write(f"[{self.current}/{self.total}] {message}\n")
                self.stream.flush()

    def _render_bar(self, message: str) -> None:
        width = shutil.get_terminal_size((80, 24)).columns - 30
        width = max(10, width)
        if self.total > 0:
            frac = self.current / self.total
            filled = int(width * frac)
            bar = "█" * filled + "░" * (width - filled)
            pct = f"{frac * 100:5.1f}%"
        else:
            bar = "░" * width
            pct = "  ?%"
        elapsed = time.monotonic() - self._start
        eta = ""
        if self.total > 0 and self.current > 0:
            remaining = elapsed / self.current * (self.total - self.current)
            eta = f" ETA {remaining:.0f}s"
        line = f"\r{bar} {pct} ({self.current}/{self.total}){eta}  {message[:30]}"
        self.stream.write(line)
        self.stream.flush()
        if self.current >= self.total > 0:
            self.stream.write("\n")

    def finish(self, message: str = "Done") -> None:
        with self._lock:
            elapsed = time.monotonic() - self._start
            if self._is_tty:
                self.stream.write(f"\r{'':80}\r")
            self.stream.write(f"{message} in {elapsed:.1f}s\n")
            self.stream.flush()


# ---------------------------------------------------------------------------
# ResultPrinter
# ---------------------------------------------------------------------------


_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "cyan": "\033[36m",
    "gray": "\033[90m",
    "magenta": "\033[35m",
}


def _c(text: str, color: str, *, use_color: bool = True) -> str:
    if not use_color:
        return text
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


class ResultPrinter:
    """Formats and prints analysis results to terminal (colorised)."""

    def __init__(self, stream: TextIO = sys.stdout, color: bool = True) -> None:
        self.stream = stream
        self.color = color and hasattr(stream, "isatty") and stream.isatty()

    def print_bug(self, bug: Bug) -> None:
        sev_colors = {
            Severity.ERROR: "red",
            Severity.WARNING: "yellow",
            Severity.INFO: "cyan",
            Severity.HINT: "gray",
        }
        col = sev_colors.get(bug.severity, "reset")
        header = _c(f"[{bug.severity.value.upper()}]", col, use_color=self.color)
        loc = _c(str(bug.location), "bold", use_color=self.color)
        self.stream.write(f"{header} {loc}: {bug.message}\n")
        if bug.category:
            self.stream.write(
                f"  category: {_c(bug.category, 'magenta', use_color=self.color)}\n"
            )
        if bug.refinement_type:
            self.stream.write(f"  type: {bug.refinement_type}\n")
        if bug.fix_suggestion:
            self.stream.write(
                f"  fix: {_c(bug.fix_suggestion, 'green', use_color=self.color)}\n"
            )

    def print_contract(self, contract: FunctionContract) -> None:
        name = _c(contract.name, "bold", use_color=self.color)
        self.stream.write(f"  {name}(")
        parts = []
        for pname, ptype in contract.params.items():
            parts.append(f"{pname}: {ptype}")
        self.stream.write(", ".join(parts))
        ret = f" -> {contract.return_type}" if contract.return_type else ""
        self.stream.write(f"){ret}\n")
        for pre in contract.preconditions:
            self.stream.write(f"    requires {pre}\n")
        for post in contract.postconditions:
            self.stream.write(f"    ensures  {post}\n")

    def print_result(self, result: AnalysisResult) -> None:
        header = _c(f"── {result.file} ", "bold", use_color=self.color)
        lang = _c(f"[{result.language.value}]", "cyan", use_color=self.color)
        self.stream.write(f"\n{header}{lang}\n")
        if result.bugs:
            self.stream.write(
                _c(f"  {len(result.bugs)} bug(s) found:\n", "red", use_color=self.color)
            )
            for bug in result.bugs:
                self.print_bug(bug)
        if result.contracts:
            self.stream.write(
                _c(
                    f"  {len(result.contracts)} contract(s) inferred:\n",
                    "green",
                    use_color=self.color,
                )
            )
            for c in result.contracts:
                self.print_contract(c)
        meta = (
            f"  analyzed {result.functions_analyzed} functions, "
            f"{result.cegar_iterations} CEGAR iterations, "
            f"{result.duration_ms:.0f}ms"
        )
        if result.timed_out:
            meta += _c(" (TIMEOUT)", "yellow", use_color=self.color)
        self.stream.write(_c(meta, "gray", use_color=self.color) + "\n")

    def print_summary(self, summary: AnalysisSummary) -> None:
        self.stream.write("\n")
        self.stream.write(
            _c("═══ Analysis Summary ═══\n", "bold", use_color=self.color)
        )
        self.stream.write(f"  Files analysed : {summary.total_files}\n")
        self.stream.write(f"  Functions      : {summary.total_functions}\n")
        self.stream.write(f"  Bugs found     : {summary.total_bugs}\n")
        if summary.bugs_by_severity:
            for sev, cnt in sorted(summary.bugs_by_severity.items()):
                self.stream.write(f"    {sev:10s}: {cnt}\n")
        if summary.bugs_by_category:
            self.stream.write("  By category:\n")
            for cat, cnt in sorted(
                summary.bugs_by_category.items(), key=lambda x: -x[1]
            ):
                self.stream.write(f"    {cat:25s}: {cnt}\n")
        self.stream.write(f"  Contracts      : {summary.total_contracts}\n")
        self.stream.write(f"  CEGAR iters    : {summary.total_cegar_iterations}\n")
        self.stream.write(f"  Duration       : {summary.duration_ms:.0f}ms\n")
        if summary.files_timed_out:
            self.stream.write(
                _c(
                    f"  Timed-out files: {summary.files_timed_out}\n",
                    "yellow",
                    use_color=self.color,
                )
            )
        self.stream.write("\n")


# ---------------------------------------------------------------------------
# ErrorHandler
# ---------------------------------------------------------------------------


class ErrorHandler:
    """Graceful error handling with optional crash reports."""

    CRASH_DIR = ".reftype-crashes"

    def handle(self, exc: BaseException, context: str = "") -> int:
        if isinstance(exc, KeyboardInterrupt):
            sys.stderr.write("\nInterrupted.\n")
            return 130
        if isinstance(exc, SystemExit):
            return exc.code if isinstance(exc.code, int) else 1
        msg = f"reftype: internal error"
        if context:
            msg += f" ({context})"
        msg += f": {exc}"
        sys.stderr.write(f"{msg}\n")
        report_path = self._write_crash_report(exc, context)
        if report_path:
            sys.stderr.write(f"Crash report written to {report_path}\n")
        return 2

    def _write_crash_report(self, exc: BaseException, context: str) -> Optional[str]:
        try:
            d = pathlib.Path(self.CRASH_DIR)
            d.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = d / f"crash-{ts}-{uuid.uuid4().hex[:8]}.txt"
            with open(path, "w") as fh:
                fh.write(f"reftype {_VERSION} crash report\n")
                fh.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
                fh.write(f"Python: {sys.version}\n")
                fh.write(f"Platform: {platform.platform()}\n")
                fh.write(f"Context: {context}\n\n")
                fh.write("Traceback:\n")
                traceback.print_exception(type(exc), exc, exc.__traceback__, file=fh)
            return str(path)
        except OSError:
            return None


# ---------------------------------------------------------------------------
# SignalHandler
# ---------------------------------------------------------------------------


class SignalHandler:
    """Handles SIGINT / SIGTERM gracefully."""

    def __init__(self) -> None:
        self._interrupted = threading.Event()
        self._original_sigint: Any = None
        self._original_sigterm: Any = None

    def install(self) -> None:
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def uninstall(self) -> None:
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)

    def _handler(self, signum: int, frame: Any) -> None:
        self._interrupted.set()
        logger.info("Received signal %d, shutting down…", signum)

    @property
    def interrupted(self) -> bool:
        return self._interrupted.is_set()


# ---------------------------------------------------------------------------
# LanguageDetector
# ---------------------------------------------------------------------------


class LanguageDetector:
    """Auto-detect language from file extensions, shebangs, config."""

    EXT_MAP: Dict[str, Language] = {
        ".py": Language.PYTHON,
        ".pyi": Language.PYTHON,
        ".ts": Language.TYPESCRIPT,
        ".tsx": Language.TYPESCRIPT,
        ".mts": Language.TYPESCRIPT,
        ".cts": Language.TYPESCRIPT,
    }

    SHEBANG_PATTERNS: List[Tuple[str, Language]] = [
        ("python", Language.PYTHON),
        ("node", Language.TYPESCRIPT),
        ("ts-node", Language.TYPESCRIPT),
        ("deno", Language.TYPESCRIPT),
    ]

    def detect_file(self, path: str) -> Optional[Language]:
        ext = pathlib.Path(path).suffix.lower()
        lang = self.EXT_MAP.get(ext)
        if lang is not None:
            return lang
        return self._detect_shebang(path)

    def _detect_shebang(self, path: str) -> Optional[Language]:
        try:
            with open(path, "r", errors="ignore") as fh:
                first = fh.readline(256)
        except OSError:
            return None
        if not first.startswith("#!"):
            return None
        for pattern, lang in self.SHEBANG_PATTERNS:
            if pattern in first:
                return lang
        return None

    def detect_project(self, directory: str) -> Language:
        d = pathlib.Path(directory)
        py_count = len(list(d.rglob("*.py")))
        ts_count = len(list(d.rglob("*.ts"))) + len(list(d.rglob("*.tsx")))
        if py_count > ts_count:
            return Language.PYTHON
        if ts_count > py_count:
            return Language.TYPESCRIPT
        if (d / "pyproject.toml").exists() or (d / "setup.py").exists():
            return Language.PYTHON
        if (d / "tsconfig.json").exists() or (d / "package.json").exists():
            return Language.TYPESCRIPT
        return Language.PYTHON


# ---------------------------------------------------------------------------
# FileDiscovery
# ---------------------------------------------------------------------------


class FileDiscovery:
    """Discover source files, respecting .gitignore and config exclusions."""

    def __init__(
        self,
        include: Sequence[str] = ("**/*.py", "**/*.ts"),
        exclude: Sequence[str] = (),
        respect_gitignore: bool = True,
    ) -> None:
        self.include = list(include)
        self.exclude = list(exclude)
        self.respect_gitignore = respect_gitignore
        self._gitignore_patterns: Optional[List[str]] = None

    def discover(self, roots: Sequence[str]) -> List[str]:
        found: List[str] = []
        for root in roots:
            p = pathlib.Path(root)
            if p.is_file():
                if self._matches_include(str(p)):
                    found.append(str(p.resolve()))
            elif p.is_dir():
                found.extend(self._walk(p))
            else:
                logger.warning("Path does not exist: %s", root)
        return sorted(set(found))

    def _walk(self, directory: pathlib.Path) -> Generator[str, None, None]:
        gitignore = self._load_gitignore(directory) if self.respect_gitignore else []
        for pat in self.include:
            for match in directory.glob(pat):
                if not match.is_file():
                    continue
                rel = str(match.relative_to(directory))
                if self._is_excluded(rel, gitignore):
                    continue
                yield str(match.resolve())

    def _matches_include(self, path: str) -> bool:
        for pat in self.include:
            if fnmatch.fnmatch(path, pat) or pathlib.Path(path).match(pat):
                return True
        return True  # single file explicitly given

    def _is_excluded(self, rel: str, gitignore: List[str]) -> bool:
        for pat in self.exclude:
            if fnmatch.fnmatch(rel, pat):
                return True
        for pat in gitignore:
            if fnmatch.fnmatch(rel, pat):
                return True
        return False

    def _load_gitignore(self, directory: pathlib.Path) -> List[str]:
        if self._gitignore_patterns is not None:
            return self._gitignore_patterns
        patterns: List[str] = []
        gi = directory / ".gitignore"
        if gi.is_file():
            with open(gi) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
        self._gitignore_patterns = patterns
        return patterns


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------


class CacheManager:
    """Manages the analysis cache directory for incremental analysis."""

    def __init__(self, cache_dir: str = ".reftype-cache") -> None:
        self.cache_dir = pathlib.Path(cache_dir)

    def ensure(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, file_path: str) -> str:
        return hashlib.sha256(file_path.encode()).hexdigest()

    def _hash_file(self, path: str) -> str:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
        except OSError:
            return ""
        return h.hexdigest()

    def get(self, file_path: str) -> Optional[AnalysisResult]:
        key = self._key(file_path)
        meta_path = self.cache_dir / f"{key}.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("source_hash") != self._hash_file(file_path):
            return None
        max_age = data.get("max_age_hours", 168) * 3600
        if time.time() - data.get("timestamp", 0) > max_age:
            return None
        return self._deserialise_result(data.get("result", {}), file_path)

    def put(self, file_path: str, result: AnalysisResult) -> None:
        self.ensure()
        key = self._key(file_path)
        meta_path = self.cache_dir / f"{key}.json"
        data = {
            "source_hash": self._hash_file(file_path),
            "timestamp": time.time(),
            "max_age_hours": 168,
            "result": self._serialise_result(result),
        }
        try:
            with open(meta_path, "w") as fh:
                json.dump(data, fh)
        except OSError as exc:
            logger.warning("Failed to write cache for %s: %s", file_path, exc)

    def invalidate(self, file_path: str) -> None:
        key = self._key(file_path)
        meta_path = self.cache_dir / f"{key}.json"
        meta_path.unlink(missing_ok=True)

    def clear(self) -> int:
        count = 0
        if self.cache_dir.exists():
            for f in self.cache_dir.iterdir():
                if f.suffix == ".json":
                    f.unlink()
                    count += 1
        return count

    @staticmethod
    def _serialise_result(result: AnalysisResult) -> Dict[str, Any]:
        bugs = []
        for b in result.bugs:
            bugs.append(
                {
                    "id": b.id,
                    "message": b.message,
                    "severity": b.severity.value,
                    "location": {
                        "file": b.location.file,
                        "line": b.location.line,
                        "column": b.location.column,
                    },
                    "category": b.category,
                    "fix_suggestion": b.fix_suggestion,
                }
            )
        contracts = []
        for c in result.contracts:
            contracts.append(
                {
                    "name": c.name,
                    "file": c.file,
                    "line": c.line,
                    "params": {k: str(v) for k, v in c.params.items()},
                    "return_type": str(c.return_type) if c.return_type else None,
                    "preconditions": c.preconditions,
                    "postconditions": c.postconditions,
                }
            )
        return {
            "file": result.file,
            "language": result.language.value,
            "bugs": bugs,
            "contracts": contracts,
            "duration_ms": result.duration_ms,
            "functions_analyzed": result.functions_analyzed,
            "cegar_iterations": result.cegar_iterations,
            "timed_out": result.timed_out,
        }

    @staticmethod
    def _deserialise_result(data: Dict[str, Any], file_path: str) -> AnalysisResult:
        bugs = []
        for bd in data.get("bugs", []):
            loc_d = bd.get("location", {})
            bugs.append(
                Bug(
                    id=bd.get("id", ""),
                    message=bd.get("message", ""),
                    severity=Severity(bd.get("severity", "warning")),
                    location=SourceLocation(
                        file=loc_d.get("file", file_path),
                        line=loc_d.get("line", 0),
                        column=loc_d.get("column", 0),
                    ),
                    category=bd.get("category", ""),
                    fix_suggestion=bd.get("fix_suggestion"),
                )
            )
        contracts = []
        for cd in data.get("contracts", []):
            contracts.append(
                FunctionContract(
                    name=cd.get("name", ""),
                    file=cd.get("file", file_path),
                    line=cd.get("line", 0),
                    preconditions=cd.get("preconditions", []),
                    postconditions=cd.get("postconditions", []),
                )
            )
        return AnalysisResult(
            file=file_path,
            language=Language(data.get("language", "python")),
            bugs=bugs,
            contracts=contracts,
            duration_ms=data.get("duration_ms", 0),
            functions_analyzed=data.get("functions_analyzed", 0),
            cegar_iterations=data.get("cegar_iterations", 0),
            timed_out=data.get("timed_out", False),
        )


# ---------------------------------------------------------------------------
# PluginLoader
# ---------------------------------------------------------------------------


class PluginLoader:
    """Loads analysis plugins from entry points."""

    ENTRY_POINT_GROUP = "reftype.plugins"

    def __init__(self) -> None:
        self._plugins: Dict[str, Any] = {}

    def discover(self) -> List[str]:
        names: List[str] = []
        try:
            if sys.version_info >= (3, 10):
                from importlib.metadata import entry_points

                eps = entry_points(group=self.ENTRY_POINT_GROUP)
            else:
                from importlib.metadata import entry_points

                all_eps = entry_points()
                eps = all_eps.get(self.ENTRY_POINT_GROUP, [])
            for ep in eps:
                names.append(ep.name)
        except Exception as exc:
            logger.debug("Plugin discovery failed: %s", exc)
        return names

    def load(self, name: str) -> Any:
        if name in self._plugins:
            return self._plugins[name]
        try:
            if sys.version_info >= (3, 10):
                from importlib.metadata import entry_points

                eps = entry_points(group=self.ENTRY_POINT_GROUP)
            else:
                from importlib.metadata import entry_points

                all_eps = entry_points()
                eps = all_eps.get(self.ENTRY_POINT_GROUP, [])
            for ep in eps:
                if ep.name == name:
                    plugin = ep.load()
                    self._plugins[name] = plugin
                    return plugin
        except Exception as exc:
            logger.warning("Failed to load plugin %s: %s", name, exc)
        return None

    def load_all(self) -> Dict[str, Any]:
        for name in self.discover():
            self.load(name)
        return dict(self._plugins)


# ---------------------------------------------------------------------------
# ParallelExecutor
# ---------------------------------------------------------------------------


def _analyze_file_worker(args: Tuple[str, str]) -> Dict[str, Any]:
    """Worker function executed in a child process.

    Uses the real liquid type / shape analysis engine instead of regex
    pattern matching.  Falls back to basic AST parsing when the engine
    is unavailable or raises.
    """
    file_path, language_str = args
    start = time.monotonic()
    lang = Language(language_str) if language_str != "auto" else Language.PYTHON
    bugs: List[Dict[str, Any]] = []
    contracts: List[Dict[str, Any]] = []
    functions_analyzed = 0
    cegar_iterations = 0
    timed_out = False

    # Map from liquid / shape engine kinds to CLI category strings
    _KIND_TO_CLI_CAT = {
        "NULL_DEREF": "null-deref",
        "DIV_BY_ZERO": "division-by-zero",
        "INDEX_OOB": "index-out-of-bounds",
        "TYPE_ERROR": "type-error",
        "ATTRIBUTE_ERROR": "attribute-error",
        "PRECONDITION_VIOLATION": "precondition-violation",
        "UNSAT_CONSTRAINT": "type-error",
        # Shape error kinds
        "DIM_MISMATCH": "shape-error",
        "NDIM_MISMATCH": "shape-error",
        "RESHAPE_INVALID": "shape-error",
        "BROADCAST_FAIL": "shape-error",
        "MATMUL_INCOMPAT": "shape-error",
        "CAT_INCOMPAT": "shape-error",
        "CONV_INCOMPAT": "shape-error",
    }

    try:
        with open(file_path, "r", errors="replace") as fh:
            source = fh.read()

        analysis_ok = False

        # ── Try the real analysis engine ──────────────────────────────
        if lang == Language.PYTHON:
            try:
                from ..api import analyze, liquid_analyze, _HAS_LIQUID
                from ..tensor_shapes import analyze_shapes

                # Run liquid type analysis (Z3-backed if available)
                if _HAS_LIQUID:
                    liq_result = liquid_analyze(source, filename=file_path)
                else:
                    liq_result = analyze(source, filename=file_path)

                for b in liq_result.bugs:
                    cat = b.category.value if hasattr(b.category, "value") else str(b.category)
                    bugs.append(
                        {
                            "id": f"{cat}-{b.location.line}",
                            "message": b.message,
                            "severity": b.severity,
                            "location": {
                                "file": file_path,
                                "line": b.location.line,
                                "column": b.location.column,
                            },
                            "category": cat.replace("_", "-"),
                        }
                    )
                functions_analyzed += liq_result.functions_analyzed
                cegar_iterations += getattr(liq_result, "guards_harvested", 0)

                # Extract contracts if available
                liq_contracts = getattr(liq_result, "_liquid_contracts", None)
                if liq_contracts:
                    for name, contract in liq_contracts.items():
                        contracts.append(
                            {
                                "name": name,
                                "file": file_path,
                                "line": getattr(contract, "line", 0),
                                "params": {
                                    p: str(t) for p, t in contract.params.items()
                                } if hasattr(contract, "params") else {},
                                "return_type": str(contract.return_type) if hasattr(contract, "return_type") else None,
                                "preconditions": [str(p) for p in contract.preconditions] if hasattr(contract, "preconditions") else [],
                                "postconditions": [str(p) for p in contract.postconditions] if hasattr(contract, "postconditions") else [],
                            }
                        )

                # Run tensor shape analysis
                try:
                    shape_result = analyze_shapes(source)
                    for err in shape_result.errors:
                        kind_name = err.kind.name if hasattr(err.kind, "name") else str(err.kind)
                        bugs.append(
                            {
                                "id": f"shape-{err.line}",
                                "message": err.message,
                                "severity": err.severity,
                                "location": {
                                    "file": file_path,
                                    "line": err.line,
                                    "column": err.col,
                                },
                                "category": _KIND_TO_CLI_CAT.get(kind_name, "shape-error"),
                            }
                        )
                    functions_analyzed += shape_result.functions_analyzed
                except Exception as exc:
                    logger.warning("Shape analysis failed for %s: %s", file_path, exc)

                analysis_ok = True

            except Exception as exc:
                logger.warning("Analysis engine unavailable for %s: %s", file_path, exc)

        # ── Fallback: basic AST-level parsing ─────────────────────────
        if not analysis_ok:
            import ast as _ast
            try:
                tree = _ast.parse(source)
                for node in _ast.walk(tree):
                    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                        functions_analyzed += 1
                        contracts.append(
                            {
                                "name": node.name,
                                "file": file_path,
                                "line": node.lineno,
                                "params": {},
                                "return_type": None,
                                "preconditions": [],
                                "postconditions": [],
                            }
                        )
                        cegar_iterations += 1
            except _ast.error:
                pass

        elapsed = (time.monotonic() - start) * 1000

    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        bugs.append(
            {
                "id": "parse-error",
                "message": str(exc),
                "severity": "error",
                "location": {"file": file_path, "line": 1, "column": 0},
                "category": "parse-error",
            }
        )

    return {
        "file": file_path,
        "language": lang.value,
        "bugs": bugs,
        "contracts": contracts,
        "duration_ms": elapsed,
        "functions_analyzed": functions_analyzed,
        "cegar_iterations": cegar_iterations,
        "timed_out": timed_out,
    }


class ParallelExecutor:
    """Parallel analysis of multiple files using multiprocessing."""

    def __init__(
        self,
        workers: int = 0,
        chunk_size: int = 4,
        timeout_per_file: float = 60.0,
        language: Language = Language.AUTO,
    ) -> None:
        self.workers = workers if workers > 0 else max(1, (os.cpu_count() or 1) - 1)
        self.chunk_size = chunk_size
        self.timeout_per_file = timeout_per_file
        self.language = language
        self._detector = LanguageDetector()

    def execute(
        self,
        files: Sequence[str],
        progress: Optional[ProgressReporter] = None,
        signal_handler: Optional[SignalHandler] = None,
    ) -> List[AnalysisResult]:
        if not files:
            return []

        tasks: List[Tuple[str, str]] = []
        for f in files:
            if self.language == Language.AUTO:
                detected = self._detector.detect_file(f)
                lang_str = detected.value if detected else "python"
            else:
                lang_str = self.language.value
            tasks.append((f, lang_str))

        if len(files) == 1 or self.workers == 1:
            return self._execute_sequential(tasks, progress, signal_handler)

        return self._execute_parallel(tasks, progress, signal_handler)

    def _execute_sequential(
        self,
        tasks: List[Tuple[str, str]],
        progress: Optional[ProgressReporter],
        signal_handler: Optional[SignalHandler],
    ) -> List[AnalysisResult]:
        results: List[AnalysisResult] = []
        for task in tasks:
            if signal_handler and signal_handler.interrupted:
                break
            raw = _analyze_file_worker(task)
            results.append(self._raw_to_result(raw))
            if progress:
                progress.update(1, task[0].rsplit("/", 1)[-1])
        return results

    def _execute_parallel(
        self,
        tasks: List[Tuple[str, str]],
        progress: Optional[ProgressReporter],
        signal_handler: Optional[SignalHandler],
    ) -> List[AnalysisResult]:
        results: List[AnalysisResult] = []
        try:
            with multiprocessing.Pool(processes=self.workers) as pool:
                for raw in pool.imap_unordered(
                    _analyze_file_worker, tasks, chunksize=self.chunk_size
                ):
                    if signal_handler and signal_handler.interrupted:
                        pool.terminate()
                        break
                    results.append(self._raw_to_result(raw))
                    if progress:
                        progress.update(1, raw["file"].rsplit("/", 1)[-1])
        except Exception as exc:
            logger.error("Parallel execution failed: %s", exc)
            return self._execute_sequential(tasks, progress, signal_handler)
        return results

    @staticmethod
    def _raw_to_result(raw: Dict[str, Any]) -> AnalysisResult:
        bugs = []
        for bd in raw.get("bugs", []):
            loc_d = bd.get("location", {})
            bugs.append(
                Bug(
                    id=bd.get("id", ""),
                    message=bd.get("message", ""),
                    severity=Severity(bd.get("severity", "warning")),
                    location=SourceLocation(
                        file=loc_d.get("file", ""),
                        line=loc_d.get("line", 0),
                        column=loc_d.get("column", 0),
                    ),
                    category=bd.get("category", ""),
                )
            )
        contracts = []
        for cd in raw.get("contracts", []):
            contracts.append(
                FunctionContract(
                    name=cd.get("name", ""),
                    file=cd.get("file", ""),
                    line=cd.get("line", 0),
                    preconditions=cd.get("preconditions", []),
                    postconditions=cd.get("postconditions", []),
                )
            )
        return AnalysisResult(
            file=raw["file"],
            language=Language(raw.get("language", "python")),
            bugs=bugs,
            contracts=contracts,
            duration_ms=raw.get("duration_ms", 0),
            functions_analyzed=raw.get("functions_analyzed", 0),
            cegar_iterations=raw.get("cegar_iterations", 0),
            timed_out=raw.get("timed_out", False),
        )


# ---------------------------------------------------------------------------
# ExitCodeManager
# ---------------------------------------------------------------------------


class ExitCodeManager:
    """Determines exit code from analysis results."""

    EXIT_SUCCESS = 0
    EXIT_BUGS_FOUND = 1
    EXIT_ERROR = 2

    def __init__(self, fail_on_new_bugs: bool = False) -> None:
        self.fail_on_new_bugs = fail_on_new_bugs

    def compute(
        self,
        summary: AnalysisSummary,
        new_bugs: Optional[int] = None,
    ) -> int:
        if summary.files_timed_out > 0 and summary.total_files == summary.files_timed_out:
            return self.EXIT_ERROR
        if self.fail_on_new_bugs and new_bugs is not None:
            return self.EXIT_BUGS_FOUND if new_bugs > 0 else self.EXIT_SUCCESS
        if summary.total_bugs > 0:
            has_errors = summary.bugs_by_severity.get("error", 0) > 0
            if has_errors:
                return self.EXIT_BUGS_FOUND
        return self.EXIT_SUCCESS


# ---------------------------------------------------------------------------
# TelemetryCollector
# ---------------------------------------------------------------------------


class TelemetryCollector:
    """Optional anonymous usage telemetry (opt-in only)."""

    TELEMETRY_FILE = ".reftype-telemetry.json"

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._events: List[Dict[str, Any]] = []
        self._session_id = uuid.uuid4().hex[:12]

    def record(self, event: str, **kwargs: Any) -> None:
        if not self.enabled:
            return
        self._events.append(
            {
                "event": event,
                "session": self._session_id,
                "timestamp": time.time(),
                **kwargs,
            }
        )

    def flush(self) -> None:
        if not self.enabled or not self._events:
            return
        try:
            p = pathlib.Path(self.TELEMETRY_FILE)
            existing: List[Dict[str, Any]] = []
            if p.exists():
                with open(p) as fh:
                    existing = json.load(fh)
            existing.extend(self._events)
            # keep only last 1000 events
            existing = existing[-1000:]
            with open(p, "w") as fh:
                json.dump(existing, fh)
            self._events.clear()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# UpdateChecker
# ---------------------------------------------------------------------------


class UpdateChecker:
    """Checks for newer versions (non-blocking, best-effort)."""

    PYPI_URL = "https://pypi.org/pypi/reftype/json"
    CACHE_FILE = ".reftype-update-check"
    CHECK_INTERVAL = 86400  # 1 day

    def __init__(self) -> None:
        self._latest: Optional[str] = None

    def should_check(self) -> bool:
        p = pathlib.Path(self.CACHE_FILE)
        if not p.exists():
            return True
        try:
            ts = float(p.read_text().strip())
            return (time.time() - ts) > self.CHECK_INTERVAL
        except (ValueError, OSError):
            return True

    def check_async(self) -> None:
        if not self.should_check():
            return
        t = threading.Thread(target=self._do_check, daemon=True)
        t.start()

    def _do_check(self) -> None:
        try:
            import urllib.request

            with urllib.request.urlopen(self.PYPI_URL, timeout=5) as resp:
                data = json.load(resp)
                self._latest = data.get("info", {}).get("version")
            p = pathlib.Path(self.CACHE_FILE)
            p.write_text(str(time.time()))
        except Exception as exc:
            logger.warning("Update check failed: %s", exc)

    def notify_if_outdated(self, current: str = _VERSION) -> Optional[str]:
        if self._latest and self._latest != current:
            return (
                f"A newer version of reftype is available: {self._latest} "
                f"(current: {current}). Run `pip install --upgrade reftype` to update."
            )
        return None


# ---------------------------------------------------------------------------
# SARIF / HTML / JSON output helpers
# ---------------------------------------------------------------------------


class SarifGenerator:
    """Generates SARIF 2.1.0 output for GitHub Advanced Security."""

    SARIF_VERSION = "2.1.0"
    SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"

    def generate(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        rules: Dict[str, Dict[str, Any]] = {}
        sarif_results: List[Dict[str, Any]] = []

        for result in results:
            for bug in result.bugs:
                rule_id = bug.category
                if rule_id not in rules:
                    rules[rule_id] = {
                        "id": rule_id,
                        "name": rule_id.replace("-", " ").title(),
                        "shortDescription": {"text": f"Refinement type: {rule_id}"},
                        "defaultConfiguration": {
                            "level": self._severity_to_level(bug.severity)
                        },
                    }
                sarif_results.append(self._bug_to_result(bug, rule_id))

        return {
            "$schema": self.SCHEMA,
            "version": self.SARIF_VERSION,
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "reftype",
                            "version": _VERSION,
                            "informationUri": "https://github.com/reftype/reftype",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": sarif_results,
                }
            ],
        }

    @staticmethod
    def _severity_to_level(sev: Severity) -> str:
        return {
            Severity.ERROR: "error",
            Severity.WARNING: "warning",
            Severity.INFO: "note",
            Severity.HINT: "note",
        }.get(sev, "warning")

    @staticmethod
    def _bug_to_result(bug: Bug, rule_id: str) -> Dict[str, Any]:
        return {
            "ruleId": rule_id,
            "level": SarifGenerator._severity_to_level(bug.severity),
            "message": {"text": bug.message},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": bug.location.file,
                            "uriBaseId": "%SRCROOT%",
                        },
                        "region": {
                            "startLine": bug.location.line,
                            "startColumn": bug.location.column + 1,
                        },
                    }
                }
            ],
            "fingerprints": {"reftype/v1": bug.fingerprint()},
            "fixes": (
                [
                    {
                        "description": {"text": bug.fix_suggestion},
                        "artifactChanges": [],
                    }
                ]
                if bug.fix_suggestion
                else []
            ),
        }


class HtmlReportGenerator:
    """Generates a self-contained HTML analysis report."""

    def generate(
        self, results: List[AnalysisResult], summary: AnalysisSummary
    ) -> str:
        bugs_html = []
        for result in results:
            for bug in result.bugs:
                sev_class = bug.severity.value
                bugs_html.append(
                    f'<tr class="{sev_class}">'
                    f"<td>{bug.severity.value}</td>"
                    f"<td>{bug.location}</td>"
                    f"<td>{bug.category}</td>"
                    f"<td>{bug.message}</td>"
                    f"<td>{bug.fix_suggestion or ''}</td>"
                    f"</tr>"
                )

        contracts_html = []
        for result in results:
            for c in result.contracts:
                params = ", ".join(f"{k}: {v}" for k, v in c.params.items())
                ret = str(c.return_type) if c.return_type else "—"
                contracts_html.append(
                    f"<tr>"
                    f"<td>{c.file}:{c.line}</td>"
                    f"<td>{c.name}</td>"
                    f"<td>{params}</td>"
                    f"<td>{ret}</td>"
                    f"</tr>"
                )

        return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <title>reftype Analysis Report</title>
        <style>
          body {{ font-family: system-ui, sans-serif; margin: 2em; }}
          h1 {{ color: #333; }}
          table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
          th {{ background: #f4f4f4; }}
          .error {{ background: #ffe0e0; }}
          .warning {{ background: #fff3cd; }}
          .info {{ background: #d1ecf1; }}
          .hint {{ background: #f0f0f0; }}
          .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1em; }}
          .card {{ background: #f8f9fa; border-radius: 8px; padding: 1em; }}
          .card h3 {{ margin: 0 0 0.5em; }}
          .card .value {{ font-size: 2em; font-weight: bold; }}
        </style>
        </head>
        <body>
        <h1>reftype Analysis Report</h1>
        <p>Generated {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        <div class="summary">
          <div class="card"><h3>Files</h3><div class="value">{summary.total_files}</div></div>
          <div class="card"><h3>Functions</h3><div class="value">{summary.total_functions}</div></div>
          <div class="card"><h3>Bugs</h3><div class="value">{summary.total_bugs}</div></div>
          <div class="card"><h3>Contracts</h3><div class="value">{summary.total_contracts}</div></div>
          <div class="card"><h3>Duration</h3><div class="value">{summary.duration_ms:.0f}ms</div></div>
        </div>
        <h2>Bugs ({len(bugs_html)})</h2>
        <table>
        <tr><th>Severity</th><th>Location</th><th>Category</th><th>Message</th><th>Fix</th></tr>
        {"".join(bugs_html)}
        </table>
        <h2>Inferred Contracts ({len(contracts_html)})</h2>
        <table>
        <tr><th>Location</th><th>Function</th><th>Parameters</th><th>Return</th></tr>
        {"".join(contracts_html)}
        </table>
        </body>
        </html>
        """)


class JsonOutputGenerator:
    """Generates JSON output."""

    def generate(
        self, results: List[AnalysisResult], summary: AnalysisSummary
    ) -> Dict[str, Any]:
        return {
            "version": _VERSION,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": {
                "total_files": summary.total_files,
                "total_functions": summary.total_functions,
                "total_bugs": summary.total_bugs,
                "bugs_by_severity": summary.bugs_by_severity,
                "bugs_by_category": summary.bugs_by_category,
                "total_contracts": summary.total_contracts,
                "total_cegar_iterations": summary.total_cegar_iterations,
                "duration_ms": summary.duration_ms,
            },
            "results": [CacheManager._serialise_result(r) for r in results],
        }


class PyiGenerator:
    """Generates .pyi stub files from inferred contracts."""

    def generate(self, contracts: List[FunctionContract]) -> str:
        lines: List[str] = ["# Auto-generated by reftype", ""]
        for c in contracts:
            params = ", ".join(f"{k}: {v}" for k, v in c.params.items())
            ret = f" -> {c.return_type}" if c.return_type else ""
            lines.append(f"def {c.name}({params}){ret}: ...")
            for pre in c.preconditions:
                lines.append(f"    # requires: {pre}")
            for post in c.postconditions:
                lines.append(f"    # ensures: {post}")
            lines.append("")
        return "\n".join(lines)


class DtsGenerator:
    """Generates .d.ts declaration files from inferred contracts."""

    TYPE_MAP: Dict[str, str] = {
        "int": "number",
        "float": "number",
        "str": "string",
        "bool": "boolean",
        "None": "void",
        "NoneType": "void",
        "list": "Array<any>",
        "dict": "Record<string, any>",
    }

    def generate(self, contracts: List[FunctionContract]) -> str:
        lines: List[str] = ["// Auto-generated by reftype", ""]
        for c in contracts:
            params = ", ".join(
                f"{k}: {self._map_type(str(v))}" for k, v in c.params.items()
            )
            ret = self._map_type(str(c.return_type)) if c.return_type else "void"
            lines.append(f"declare function {c.name}({params}): {ret};")
            for pre in c.preconditions:
                lines.append(f"  // requires: {pre}")
            for post in c.postconditions:
                lines.append(f"  // ensures: {post}")
            lines.append("")
        return "\n".join(lines)

    def _map_type(self, t: str) -> str:
        base = t.split("|")[0].strip().lstrip("{").split()[0]
        return self.TYPE_MAP.get(base, "any")


# ---------------------------------------------------------------------------
# Subcommand Protocol & Implementations
# ---------------------------------------------------------------------------


class Command(Protocol):
    """Protocol for CLI subcommands."""

    def register(self, parser: argparse.ArgumentParser) -> None: ...
    def execute(self, args: argparse.Namespace) -> int: ...


# ── AnalyzeCommand ────────────────────────────────────────────────────────


class AnalyzeCommand:
    """Analyze files and directories for refinement type bugs."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("paths", nargs="*", default=["."], help="Files or directories")
        parser.add_argument(
            "-l", "--language", choices=["python", "typescript", "auto"], default="auto"
        )
        parser.add_argument(
            "-f",
            "--format",
            choices=["pyi", "dts", "sarif", "html", "json"],
            default="json",
        )
        parser.add_argument("-o", "--output", help="Output file (default: stdout)")
        parser.add_argument("-v", "--verbose", action="count", default=0)
        parser.add_argument("-c", "--config", help="Config file path")
        parser.add_argument("--include", nargs="*", help="Include patterns")
        parser.add_argument("--exclude", nargs="*", help="Exclude patterns")
        parser.add_argument("--max-functions", type=int, default=0)
        parser.add_argument("--timeout", type=float, default=300.0)
        parser.add_argument("-w", "--workers", type=int, default=0)
        parser.add_argument("--incremental", action="store_true", default=False)
        parser.add_argument("--baseline", help="Baseline file for comparison")
        parser.add_argument("--no-color", action="store_true")
        parser.add_argument("--fail-on-new-bugs", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        LoggingSetup.configure(getattr(args, "verbose", 0))
        loader = ConfigLoader()
        file_cfg = loader.load(getattr(args, "config", None))
        cfg = loader.merge(file_cfg, args)
        errors = loader.validate(cfg)
        if errors:
            for e in errors:
                sys.stderr.write(f"Config error: {e}\n")
            return 2

        signal_handler = SignalHandler()
        signal_handler.install()

        telemetry = TelemetryCollector(enabled=cfg.telemetry_enabled)
        telemetry.record("analyze_start", files=len(cfg.paths))

        try:
            include = getattr(args, "include", None) or cfg.include_patterns
            exclude = getattr(args, "exclude", None) or cfg.exclude_patterns
            discovery = FileDiscovery(include=include, exclude=exclude)
            files = discovery.discover(getattr(args, "paths", cfg.paths))

            if not files:
                sys.stderr.write("No files found to analyse.\n")
                return 0

            if cfg.max_functions > 0:
                files = files[: cfg.max_functions]

            logger.info("Analysing %d files with %d workers", len(files), cfg.effective_workers())

            cache = CacheManager(cfg.incremental.cache_dir) if cfg.incremental.enabled else None

            cached_results: List[AnalysisResult] = []
            files_to_analyze = list(files)
            if cache:
                remaining = []
                for f in files_to_analyze:
                    cached = cache.get(f)
                    if cached:
                        cached_results.append(cached)
                    else:
                        remaining.append(f)
                files_to_analyze = remaining
                if cached_results:
                    logger.info("Using %d cached results", len(cached_results))

            progress = ProgressReporter(total=len(files_to_analyze))
            executor = ParallelExecutor(
                workers=cfg.effective_workers(),
                chunk_size=cfg.parallel.chunk_size,
                timeout_per_file=cfg.parallel.timeout_per_file,
                language=cfg.language,
            )

            new_results = executor.execute(files_to_analyze, progress, signal_handler)
            progress.finish()

            if cache:
                for r in new_results:
                    cache.put(r.file, r)

            all_results = cached_results + new_results
            summary = AnalysisSummary()
            for r in all_results:
                summary.merge(r)

            # Baseline comparison
            new_bug_count: Optional[int] = None
            if cfg.baseline_file:
                new_bug_count = self._compare_baseline(cfg.baseline_file, all_results)

            # Output
            self._write_output(args, cfg, all_results, summary)

            # Terminal summary (only when stdout is a TTY, not piped)
            use_color = not getattr(args, "no_color", False)
            printer = ResultPrinter(stream=sys.stderr, color=use_color)
            if getattr(args, "verbose", 0) >= 1:
                for r in all_results:
                    printer.print_result(r)
            printer.print_summary(summary)

            telemetry.record(
                "analyze_end",
                bugs=summary.total_bugs,
                files=summary.total_files,
                duration=summary.duration_ms,
            )
            telemetry.flush()

            exit_mgr = ExitCodeManager(fail_on_new_bugs=cfg.fail_on_new_bugs)
            return exit_mgr.compute(summary, new_bug_count)

        except Exception as exc:
            return ErrorHandler().handle(exc, "analyze")
        finally:
            signal_handler.uninstall()

    def _compare_baseline(
        self, baseline_path: str, results: List[AnalysisResult]
    ) -> int:
        try:
            with open(baseline_path) as fh:
                baseline_data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load baseline: %s", exc)
            return 0

        baseline_fps: Set[str] = set()
        for r in baseline_data.get("results", []):
            for b in r.get("bugs", []):
                fp = hashlib.sha256(
                    f"{b.get('category', '')}:{r.get('file', '')}:{b.get('message', '')}".encode()
                ).hexdigest()[:16]
                baseline_fps.add(fp)

        new_bugs = 0
        for result in results:
            for bug in result.bugs:
                if bug.fingerprint() not in baseline_fps:
                    new_bugs += 1
        return new_bugs

    def _write_output(
        self,
        args: argparse.Namespace,
        cfg: Configuration,
        results: List[AnalysisResult],
        summary: AnalysisSummary,
    ) -> None:
        fmt = getattr(args, "format", cfg.output_format.value)
        if isinstance(fmt, OutputFormat):
            fmt = fmt.value
        output_path = getattr(args, "output", cfg.output_file)

        content: str
        if fmt == "sarif":
            data = SarifGenerator().generate(results)
            content = json.dumps(data, indent=2)
        elif fmt == "html":
            content = HtmlReportGenerator().generate(results, summary)
        elif fmt == "json":
            data = JsonOutputGenerator().generate(results, summary)
            content = json.dumps(data, indent=2)
        elif fmt == "pyi":
            all_contracts = [c for r in results for c in r.contracts]
            content = PyiGenerator().generate(all_contracts)
        elif fmt == "dts":
            all_contracts = [c for r in results for c in r.contracts]
            content = DtsGenerator().generate(all_contracts)
        else:
            data = JsonOutputGenerator().generate(results, summary)
            content = json.dumps(data, indent=2)

        if output_path:
            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as fh:
                fh.write(content)
            logger.info("Output written to %s", output_path)
        else:
            sys.stdout.write(content + "\n")


# ── WatchCommand ──────────────────────────────────────────────────────────


class WatchCommand:
    """File watcher for incremental development."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("paths", nargs="*", default=["."])
        parser.add_argument("-l", "--language", default="auto")
        parser.add_argument("--debounce", type=float, default=0.5)
        parser.add_argument(
            "--editor", choices=["vim", "emacs", "vscode"], default=None
        )
        parser.add_argument("-v", "--verbose", action="count", default=0)
        parser.add_argument("-c", "--config", help="Config file path")

    def execute(self, args: argparse.Namespace) -> int:
        LoggingSetup.configure(getattr(args, "verbose", 0))
        loader = ConfigLoader()
        file_cfg = loader.load(getattr(args, "config", None))
        cfg = loader.merge(file_cfg, args)

        signal_handler = SignalHandler()
        signal_handler.install()

        debounce = getattr(args, "debounce", 0.5)
        printer = ResultPrinter()

        discovery = FileDiscovery(
            include=cfg.include_patterns, exclude=cfg.exclude_patterns
        )
        files = discovery.discover(getattr(args, "paths", cfg.paths))
        if not files:
            sys.stderr.write("No files found to watch.\n")
            return 0

        sys.stderr.write(f"Watching {len(files)} files (debounce={debounce}s)…\n")

        mtimes: Dict[str, float] = {}
        for f in files:
            try:
                mtimes[f] = os.path.getmtime(f)
            except OSError:
                pass

        executor = ParallelExecutor(workers=1, language=cfg.language)

        try:
            while not signal_handler.interrupted:
                changed: List[str] = []
                for f in files:
                    try:
                        mt = os.path.getmtime(f)
                    except OSError:
                        continue
                    prev = mtimes.get(f)
                    if prev is None or mt > prev:
                        mtimes[f] = mt
                        if prev is not None:
                            changed.append(f)

                if changed:
                    sys.stderr.write(
                        f"\n[{time.strftime('%H:%M:%S')}] "
                        f"Detected changes in {len(changed)} file(s)\n"
                    )
                    results = executor.execute(changed)
                    for r in results:
                        printer.print_result(r)
                    summary = AnalysisSummary()
                    for r in results:
                        summary.merge(r)
                    if summary.total_bugs > 0:
                        sys.stderr.write(
                            _c(
                                f"⚠ {summary.total_bugs} bug(s) found\n",
                                "yellow",
                                use_color=True,
                            )
                        )
                    else:
                        sys.stderr.write(
                            _c("✓ No bugs found\n", "green", use_color=True)
                        )

                time.sleep(debounce)
        except KeyboardInterrupt:
            pass
        finally:
            signal_handler.uninstall()

        sys.stderr.write("\nWatch stopped.\n")
        return 0


# ── CiCheckCommand ────────────────────────────────────────────────────────


class CiCheckCommand:
    """CI pipeline mode with exit codes and SARIF output."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("paths", nargs="*", default=["."])
        parser.add_argument("-l", "--language", default="auto")
        parser.add_argument("--baseline", help="Baseline file")
        parser.add_argument("--sarif-output", help="SARIF output path")
        parser.add_argument("--max-new-bugs", type=int, default=0)
        parser.add_argument("--max-total-bugs", type=int, default=None)
        parser.add_argument("--fail-on-new-bugs", action="store_true", default=True)
        parser.add_argument("-v", "--verbose", action="count", default=0)
        parser.add_argument("-c", "--config", help="Config file path")
        parser.add_argument("-w", "--workers", type=int, default=0)
        parser.add_argument("--timeout", type=float, default=300.0)
        parser.add_argument(
            "-f", "--format",
            choices=["sarif", "json"],
            default="sarif",
        )

    def execute(self, args: argparse.Namespace) -> int:
        LoggingSetup.configure(getattr(args, "verbose", 0))
        loader = ConfigLoader()
        file_cfg = loader.load(getattr(args, "config", None))
        cfg = loader.merge(file_cfg, args)
        cfg.fail_on_new_bugs = getattr(args, "fail_on_new_bugs", True)

        try:
            discovery = FileDiscovery(
                include=cfg.include_patterns, exclude=cfg.exclude_patterns
            )
            files = discovery.discover(getattr(args, "paths", cfg.paths))
            if not files:
                sys.stderr.write("No files found.\n")
                return 0

            executor = ParallelExecutor(
                workers=cfg.effective_workers(),
                language=cfg.language,
            )
            results = executor.execute(files)

            summary = AnalysisSummary()
            for r in results:
                summary.merge(r)

            # SARIF output
            sarif_path = getattr(args, "sarif_output", None)
            if sarif_path:
                sarif = SarifGenerator().generate(results)
                pathlib.Path(sarif_path).parent.mkdir(parents=True, exist_ok=True)
                with open(sarif_path, "w") as fh:
                    json.dump(sarif, fh, indent=2)
                logger.info("SARIF written to %s", sarif_path)

            # Baseline comparison
            new_bugs = 0
            baseline_path = getattr(args, "baseline", cfg.baseline_file)
            if baseline_path:
                analyze_cmd = AnalyzeCommand()
                new_bugs = analyze_cmd._compare_baseline(baseline_path, results)
                sys.stderr.write(f"New bugs since baseline: {new_bugs}\n")

            max_new = getattr(args, "max_new_bugs", 0)
            max_total = getattr(args, "max_total_bugs", None)

            # Thresholds
            if max_total is not None and summary.total_bugs > max_total:
                sys.stderr.write(
                    f"FAIL: {summary.total_bugs} bugs exceed threshold of {max_total}\n"
                )
                return 1
            if cfg.fail_on_new_bugs and new_bugs > max_new:
                sys.stderr.write(
                    f"FAIL: {new_bugs} new bugs exceed threshold of {max_new}\n"
                )
                return 1

            # CI annotations
            self._emit_annotations(results)

            sys.stderr.write(
                f"CI check passed: {summary.total_bugs} total bugs, {new_bugs} new\n"
            )
            return 0

        except Exception as exc:
            return ErrorHandler().handle(exc, "ci-check")

    @staticmethod
    def _emit_annotations(results: List[AnalysisResult]) -> None:
        """Emit GitHub Actions annotations."""
        is_gha = os.environ.get("GITHUB_ACTIONS") == "true"
        if not is_gha:
            return
        for result in results:
            for bug in result.bugs:
                level = "error" if bug.severity == Severity.ERROR else "warning"
                msg = bug.message.replace("\n", "%0A")
                print(
                    f"::{level} file={bug.location.file},"
                    f"line={bug.location.line},"
                    f"col={bug.location.column}::{msg}"
                )


# ── InitCommand ───────────────────────────────────────────────────────────


class InitCommand:
    """Initialise .reftype.toml configuration in the project."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--language", choices=["python", "typescript", "auto"], default="auto"
        )
        parser.add_argument("--force", action="store_true")
        parser.add_argument("directory", nargs="?", default=".")

    def execute(self, args: argparse.Namespace) -> int:
        directory = getattr(args, "directory", ".")
        config_path = pathlib.Path(directory) / ".reftype.toml"

        if config_path.exists() and not getattr(args, "force", False):
            sys.stderr.write(
                f"{config_path} already exists. Use --force to overwrite.\n"
            )
            return 1

        lang_str = getattr(args, "language", "auto")
        if lang_str == "auto":
            detector = LanguageDetector()
            detected = detector.detect_project(directory)
            lang_str = detected.value

        if lang_str == "python":
            include = '["**/*.py"]'
            exclude = '["__pycache__/**", ".venv/**", "*.egg-info/**", "dist/**"]'
        else:
            include = '["**/*.ts", "**/*.tsx"]'
            exclude = '["node_modules/**", "dist/**", "build/**"]'

        template = textwrap.dedent(f"""\
        # reftype configuration
        # https://github.com/reftype/reftype

        [reftype]
        language = "{lang_str}"
        include = {include}
        exclude = {exclude}

        [reftype.domain]
        interval_precision = 64
        octagon_enabled = false

        [reftype.cegar]
        max_iterations = 50
        refinement_strategy = "counterexample-guided"
        interpolation_enabled = true

        [reftype.incremental]
        enabled = true
        cache_dir = ".reftype-cache"

        [reftype.parallel]
        workers = 0  # auto-detect
        """)

        with open(config_path, "w") as fh:
            fh.write(template)

        sys.stdout.write(f"Created {config_path}\n")
        sys.stdout.write(f"Detected language: {lang_str}\n")
        return 0


# ── ReportCommand ─────────────────────────────────────────────────────────


class ReportCommand:
    """Generate analysis reports in various formats."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="JSON analysis results file")
        parser.add_argument(
            "-f", "--format", choices=["html", "sarif", "json"], default="html"
        )
        parser.add_argument("-o", "--output", help="Output file")

    def execute(self, args: argparse.Namespace) -> int:
        input_path = getattr(args, "input", None)
        if not input_path or not pathlib.Path(input_path).exists():
            sys.stderr.write(f"Input file not found: {input_path}\n")
            return 2

        try:
            with open(input_path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            sys.stderr.write(f"Failed to read input: {exc}\n")
            return 2

        results = [
            CacheManager._deserialise_result(r, r.get("file", ""))
            for r in data.get("results", [])
        ]
        summary = AnalysisSummary()
        for r in results:
            summary.merge(r)

        fmt = getattr(args, "format", "html")
        if fmt == "html":
            content = HtmlReportGenerator().generate(results, summary)
        elif fmt == "sarif":
            sarif = SarifGenerator().generate(results)
            content = json.dumps(sarif, indent=2)
        else:
            content = json.dumps(
                JsonOutputGenerator().generate(results, summary), indent=2
            )

        output_path = getattr(args, "output", None)
        if output_path:
            with open(output_path, "w") as fh:
                fh.write(content)
            sys.stdout.write(f"Report written to {output_path}\n")
        else:
            sys.stdout.write(content + "\n")
        return 0


# ── ExportCommand ─────────────────────────────────────────────────────────


class ExportCommand:
    """Export inferred contracts to various formats."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("input", help="JSON analysis results file")
        parser.add_argument(
            "-f", "--format", choices=["pyi", "dts", "json"], default="pyi"
        )
        parser.add_argument("-o", "--output", help="Output file")

    def execute(self, args: argparse.Namespace) -> int:
        input_path = getattr(args, "input", None)
        if not input_path or not pathlib.Path(input_path).exists():
            sys.stderr.write(f"Input file not found: {input_path}\n")
            return 2

        try:
            with open(input_path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            sys.stderr.write(f"Failed to read input: {exc}\n")
            return 2

        results = [
            CacheManager._deserialise_result(r, r.get("file", ""))
            for r in data.get("results", [])
        ]
        all_contracts = [c for r in results for c in r.contracts]

        fmt = getattr(args, "format", "pyi")
        if fmt == "pyi":
            content = PyiGenerator().generate(all_contracts)
        elif fmt == "dts":
            content = DtsGenerator().generate(all_contracts)
        else:
            content = json.dumps(
                [CacheManager._serialise_result(r) for r in results], indent=2
            )

        output_path = getattr(args, "output", None)
        if output_path:
            with open(output_path, "w") as fh:
                fh.write(content)
            sys.stdout.write(f"Exported to {output_path}\n")
        else:
            sys.stdout.write(content + "\n")
        return 0


# ── DiffCommand ───────────────────────────────────────────────────────────


class DiffCommand:
    """Compare two analysis results, show new/fixed/changed bugs."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("before", help="Before analysis results (JSON)")
        parser.add_argument("after", help="After analysis results (JSON)")
        parser.add_argument("--no-color", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        before_path = getattr(args, "before", None)
        after_path = getattr(args, "after", None)

        for label, path in [("before", before_path), ("after", after_path)]:
            if not path or not pathlib.Path(path).exists():
                sys.stderr.write(f"{label} file not found: {path}\n")
                return 2

        try:
            with open(before_path) as fh:
                before_data = json.load(fh)
            with open(after_path) as fh:
                after_data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            sys.stderr.write(f"Failed to read files: {exc}\n")
            return 2

        before_fps: Dict[str, Dict[str, Any]] = {}
        for r in before_data.get("results", []):
            for b in r.get("bugs", []):
                fp = hashlib.sha256(
                    f"{b.get('category', '')}:{r.get('file', '')}:{b.get('message', '')}".encode()
                ).hexdigest()[:16]
                before_fps[fp] = {**b, "file": r.get("file", "")}

        after_fps: Dict[str, Dict[str, Any]] = {}
        for r in after_data.get("results", []):
            for b in r.get("bugs", []):
                fp = hashlib.sha256(
                    f"{b.get('category', '')}:{r.get('file', '')}:{b.get('message', '')}".encode()
                ).hexdigest()[:16]
                after_fps[fp] = {**b, "file": r.get("file", "")}

        new_fps = set(after_fps.keys()) - set(before_fps.keys())
        fixed_fps = set(before_fps.keys()) - set(after_fps.keys())

        use_color = not getattr(args, "no_color", False)

        sys.stdout.write(
            _c(f"\n═══ Analysis Diff ═══\n", "bold", use_color=use_color)
        )

        if new_fps:
            sys.stdout.write(
                _c(f"\n  New bugs ({len(new_fps)}):\n", "red", use_color=use_color)
            )
            for fp in sorted(new_fps):
                b = after_fps[fp]
                sys.stdout.write(
                    f"    + [{b.get('severity', 'warning')}] "
                    f"{b.get('file', '')}:{b.get('location', {}).get('line', '?')}: "
                    f"{b.get('message', '')}\n"
                )

        if fixed_fps:
            sys.stdout.write(
                _c(f"\n  Fixed bugs ({len(fixed_fps)}):\n", "green", use_color=use_color)
            )
            for fp in sorted(fixed_fps):
                b = before_fps[fp]
                sys.stdout.write(
                    f"    - [{b.get('severity', 'warning')}] "
                    f"{b.get('file', '')}:{b.get('location', {}).get('line', '?')}: "
                    f"{b.get('message', '')}\n"
                )

        unchanged = set(before_fps.keys()) & set(after_fps.keys())
        sys.stdout.write(f"\n  Summary: +{len(new_fps)} new, -{len(fixed_fps)} fixed, {len(unchanged)} unchanged\n\n")

        return 1 if new_fps else 0


# ── ServerCommand ─────────────────────────────────────────────────────────


class ServerCommand:
    """Start the Language Server Protocol server."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--transport", choices=["stdio", "tcp"], default="stdio"
        )
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=2087)
        parser.add_argument("-v", "--verbose", action="count", default=0)
        parser.add_argument("--log-file", help="Log file path")

    def execute(self, args: argparse.Namespace) -> int:
        LoggingSetup.configure(
            getattr(args, "verbose", 0),
            log_file=getattr(args, "log_file", None),
        )
        transport = getattr(args, "transport", "stdio")

        try:
            from .lsp_server import ReftypeLspServer

            server = ReftypeLspServer()
            if transport == "stdio":
                server.run_stdio()
            else:
                host = getattr(args, "host", "127.0.0.1")
                port = getattr(args, "port", 2087)
                server.run_tcp(host, port)
            return 0
        except ImportError:
            sys.stderr.write("LSP server module not available.\n")
            return 2
        except Exception as exc:
            return ErrorHandler().handle(exc, "server")


# ── VersionCommand ────────────────────────────────────────────────────────


class VersionCommand:
    """Show version information."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--json", action="store_true", dest="as_json")

    def execute(self, args: argparse.Namespace) -> int:
        info = {
            "version": _VERSION,
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
        }
        if getattr(args, "as_json", False):
            sys.stdout.write(json.dumps(info, indent=2) + "\n")
        else:
            sys.stdout.write(f"reftype {_VERSION}\n")
            sys.stdout.write(f"Python {sys.version}\n")
            sys.stdout.write(f"Platform {platform.platform()}\n")
        return 0


# ── ConfigCommand ─────────────────────────────────────────────────────────


class ConfigCommand:
    """Show or edit configuration."""

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "action", choices=["show", "path", "defaults"], nargs="?", default="show"
        )
        parser.add_argument("-c", "--config", help="Config file path")

    def execute(self, args: argparse.Namespace) -> int:
        action = getattr(args, "action", "show")
        loader = ConfigLoader()

        if action == "path":
            path = loader.find_config_file()
            if path:
                sys.stdout.write(f"{path}\n")
            else:
                sys.stdout.write("No config file found.\n")
            return 0

        if action == "defaults":
            cfg = Configuration()
            sys.stdout.write(json.dumps(asdict(cfg), indent=2, default=str) + "\n")
            return 0

        # show
        config_path = getattr(args, "config", None) or loader.find_config_file()
        if config_path and pathlib.Path(config_path).exists():
            file_cfg = loader.load(config_path)
            sys.stdout.write(f"# Loaded from {config_path}\n")
            sys.stdout.write(json.dumps(file_cfg, indent=2) + "\n")
        else:
            sys.stdout.write("No configuration file found. Using defaults.\n")
            cfg = Configuration()
            sys.stdout.write(json.dumps(asdict(cfg), indent=2, default=str) + "\n")
        return 0


# ── VerifyCommand ─────────────────────────────────────────────────────────


class VerifyCommand:
    """Verify nn.Module architecture via constraint-based verification.

    Extracts the computation graph from an nn.Module class, then verifies
    shape compatibility, device consistency, and gradient flow using Z3-backed
    symbolic constraint propagation. Produces either a safety certificate or a counterexample trace.
    """

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("file", help="Python file containing nn.Module class")
        parser.add_argument(
            "--input-shape", "-s", action="append", default=[],
            help="Input shape as name=dim1,dim2,... (e.g., x=batch,3,224,224)"
        )
        parser.add_argument(
            "--no-device-check", action="store_true",
            help="Disable device consistency checking"
        )
        parser.add_argument(
            "--no-phase-check", action="store_true",
            help="Disable train/eval phase checking"
        )
        parser.add_argument(
            "--cegar-iterations", type=int, default=10,
            help="Max CEGAR refinement iterations (default: 10)"
        )
        parser.add_argument(
            "--format", "-f", choices=["text", "json", "sarif"], default="text",
            help="Output format"
        )

    def execute(self, args: argparse.Namespace) -> int:
        filepath = pathlib.Path(args.file)
        if not filepath.exists():
            sys.stderr.write(f"File not found: {args.file}\n")
            return 1

        try:
            source = filepath.read_text(encoding="utf-8")
        except Exception as e:
            sys.stderr.write(f"Cannot read file: {e}\n")
            return 1

        # Parse input shapes
        input_shapes: Dict[str, tuple] = {}
        for spec in args.input_shape:
            if "=" not in spec:
                sys.stderr.write(f"Invalid shape spec: {spec} (use name=d1,d2,...)\n")
                return 1
            name, dims_str = spec.split("=", 1)
            dims = []
            for d in dims_str.split(","):
                d = d.strip()
                try:
                    dims.append(int(d))
                except ValueError:
                    dims.append(d)  # symbolic dim
            input_shapes[name] = tuple(dims)

        try:
            from src.api import verify_architecture
            result = verify_architecture(
                source,
                input_shapes=input_shapes,
                check_devices=not args.no_device_check,
                check_phases=not args.no_phase_check,
                max_cegar_iterations=args.cegar_iterations,
                filename=str(filepath),
            )
        except RuntimeError as e:
            sys.stderr.write(f"Error: {e}\n")
            return 1

        fmt = getattr(args, "format", "text")
        if fmt == "json":
            out = {
                "file": str(filepath),
                "bugs": [
                    {"line": b.location.line, "message": b.message, "severity": b.severity}
                    for b in result.bugs
                ],
                "duration_ms": result.duration_ms,
                "status": "SAFE" if not result.bugs else "UNSAFE",
            }
            sys.stdout.write(json.dumps(out, indent=2) + "\n")
        elif fmt == "text":
            if not result.bugs:
                sys.stdout.write(
                    f"✓ {filepath.name}: Architecture verified safe "
                    f"({result.duration_ms:.1f}ms)\n"
                )
                # Show discovered contracts if any
                contracts = getattr(result, "_shape_contracts", [])
                if contracts:
                    sys.stdout.write(f"  Discovered {len(contracts)} shape contracts:\n")
                    for c in contracts[:5]:
                        sys.stdout.write(f"    {c}\n")
            else:
                sys.stdout.write(
                    f"✗ {filepath.name}: {len(result.bugs)} verification errors "
                    f"({result.duration_ms:.1f}ms)\n"
                )
                for b in result.bugs:
                    sys.stdout.write(
                        f"  L{b.location.line}: {b.message}\n"
                    )
        else:
            sarif = result.to_sarif()
            sys.stdout.write(json.dumps(sarif, indent=2) + "\n")

        return 0 if not result.bugs else 1


# ---------------------------------------------------------------------------
# ReftypeCliApp — main application
# ---------------------------------------------------------------------------


class ReftypeCliApp:
    """Main CLI application class that wires subcommands to argparse."""

    COMMANDS: Dict[str, Callable[[], Command]] = {
        "analyze": lambda: AnalyzeCommand(),
        "verify": lambda: VerifyCommand(),
        "watch": lambda: WatchCommand(),
        "ci-check": lambda: CiCheckCommand(),
        "init": lambda: InitCommand(),
        "report": lambda: ReportCommand(),
        "export": lambda: ExportCommand(),
        "diff": lambda: DiffCommand(),
        "server": lambda: ServerCommand(),
        "version": lambda: VersionCommand(),
        "config": lambda: ConfigCommand(),
    }

    def __init__(self) -> None:
        self.parser = self._build_parser()
        self._update_checker = UpdateChecker()

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="reftype",
            description=(
                "Refinement type inference for dynamically-typed languages "
                "(Python & TypeScript) using CEGAR."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""\
            examples:
              reftype analyze .
              reftype analyze src/ --format sarif -o results.sarif
              reftype watch src/ --debounce 1.0
              reftype ci-check --baseline baseline.json --sarif-output results.sarif
              reftype init --language python
              reftype server --transport stdio
              reftype diff before.json after.json
            """),
        )
        parser.add_argument(
            "--version", action="version", version=f"reftype {_VERSION}"
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        for name, factory in self.COMMANDS.items():
            cmd = factory()
            sub = subparsers.add_parser(name, help=self._command_help(name))
            cmd.register(sub)

        return parser

    @staticmethod
    def _command_help(name: str) -> str:
        helps: Dict[str, str] = {
            "analyze": "Analyse files/directories for refinement type bugs",
            "verify": "Verify nn.Module architecture via constraint-based verification",
            "watch": "Watch files for changes and re-analyse incrementally",
            "ci-check": "Run analysis in CI mode with exit codes",
            "init": "Initialise .reftype.toml configuration",
            "report": "Generate analysis reports",
            "export": "Export inferred contracts",
            "diff": "Compare two analysis results",
            "server": "Start the LSP server",
            "version": "Show version information",
            "config": "Show or edit configuration",
        }
        return helps.get(name, "")

    def run(self, argv: Optional[Sequence[str]] = None) -> int:
        args = self.parser.parse_args(argv)
        command_name = getattr(args, "command", None)

        if not command_name:
            self.parser.print_help()
            return 0

        self._update_checker.check_async()

        factory = self.COMMANDS.get(command_name)
        if factory is None:
            sys.stderr.write(f"Unknown command: {command_name}\n")
            return 2

        cmd = factory()
        try:
            exit_code = cmd.execute(args)
        except Exception as exc:
            exit_code = ErrorHandler().handle(exc, command_name)

        update_msg = self._update_checker.notify_if_outdated()
        if update_msg:
            sys.stderr.write(f"\n{update_msg}\n")

        return exit_code


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    app = ReftypeCliApp()
    return app.run(argv)


if __name__ == "__main__":
    sys.exit(main())
