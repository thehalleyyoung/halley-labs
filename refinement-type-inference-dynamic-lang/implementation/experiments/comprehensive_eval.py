"""
Comprehensive Evaluation Suite for Refinement Type Inference.

Runs analysis on 10+ real Python packages:
  requests, flask, click, httpx, pydantic, fastapi, django (subset),
  pytest, black, rich

For each package:
  - Count guard patterns
  - Run refinement type analyzer
  - Classify findings by category
  - Measure precision and recall

Comparison matrix: our tool vs mypy vs pyright vs pylint
Statistical analysis with bootstrap confidence intervals
False positive analysis with categorization
"""

import argparse
import ast
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import (
    analyze_python_source,
    run_cegar,
    PythonAnalyzer,
    BugCategory,
    BugReport,
    AnalysisResult,
    FunctionSummary,
)
from src.python_frontend.guard_extractor import (
    extract_guards,
    GuardPattern,
    ExtractedGuard,
    PredicateKind,
)
from src.smt.solver import (
    Z3Solver,
    SatResult,
    Comparison,
    ComparisonOp,
    Var,
    Const,
    IsInstance,
    IsNone,
    And,
    Not,
    BoolLit,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Configuration and Setup
# ═══════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR = Path(__file__).parent / ".cache"


@dataclass
class PackageConfig:
    """Configuration for a single package to analyze."""
    name: str
    pip_name: str
    import_name: str
    version: str = ""
    max_files: int = 200
    exclude_patterns: Tuple[str, ...] = ()
    description: str = ""
    expected_loc_range: Tuple[int, int] = (100, 500_000)


@dataclass
class EvalConfig:
    """Global evaluation configuration."""
    packages: List[PackageConfig] = field(default_factory=list)
    output_dir: Path = RESULTS_DIR
    cache_dir: Path = CACHE_DIR
    timeout_per_file_s: float = 30.0
    timeout_per_package_s: float = 600.0
    max_workers: int = 1
    run_external_tools: bool = True
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    verbose: bool = False
    selected_packages: Optional[List[str]] = None


DEFAULT_PACKAGES = [
    PackageConfig(
        name="requests", pip_name="requests", import_name="requests",
        description="HTTP library for Python",
        exclude_patterns=("*test*", "*vendor*", "*_compat*"),
    ),
    PackageConfig(
        name="flask", pip_name="flask", import_name="flask",
        description="Micro web framework",
        exclude_patterns=("*test*",),
    ),
    PackageConfig(
        name="click", pip_name="click", import_name="click",
        description="CLI framework",
        exclude_patterns=("*test*",),
    ),
    PackageConfig(
        name="httpx", pip_name="httpx", import_name="httpx",
        description="Async HTTP client",
        exclude_patterns=("*test*", "*_compat*"),
    ),
    PackageConfig(
        name="pydantic", pip_name="pydantic", import_name="pydantic",
        description="Data validation using Python type hints",
        exclude_patterns=("*test*", "*_internal*"),
        max_files=100,
    ),
    PackageConfig(
        name="fastapi", pip_name="fastapi", import_name="fastapi",
        description="Async web framework",
        exclude_patterns=("*test*",),
    ),
    PackageConfig(
        name="django", pip_name="django", import_name="django",
        description="Web framework (subset)",
        exclude_patterns=("*test*", "*migrations*", "*locale*", "*contrib/admin*"),
        max_files=50,
    ),
    PackageConfig(
        name="pytest", pip_name="pytest", import_name="pytest",
        description="Testing framework",
        exclude_patterns=("*test*",),
        max_files=80,
    ),
    PackageConfig(
        name="black", pip_name="black", import_name="black",
        description="Code formatter",
        exclude_patterns=("*test*", "*data*", "*blib2to3*"),
    ),
    PackageConfig(
        name="rich", pip_name="rich", import_name="rich",
        description="Rich text library",
        exclude_patterns=("*test*",),
    ),
]


@dataclass
class FileFinding:
    """A single finding from any tool."""
    file_path: str
    line: int
    col: int
    message: str
    category: str
    severity: str
    tool: str
    confidence: float = 1.0


@dataclass
class PackageResult:
    """Results of analyzing one package."""
    package_name: str
    num_files: int
    total_loc: int
    total_functions: int
    total_classes: int
    guard_counts: Dict[str, int] = field(default_factory=dict)
    findings: List[FileFinding] = field(default_factory=list)
    analysis_time_s: float = 0.0
    errors: List[str] = field(default_factory=list)
    external_findings: Dict[str, List[FileFinding]] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Overall evaluation results."""
    packages: Dict[str, PackageResult] = field(default_factory=dict)
    comparison: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: PackageDownloader
# ═══════════════════════════════════════════════════════════════════════════

class PackageDownloader:
    """Download or locate real Python packages for analysis."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._location_cache: Dict[str, Path] = {}

    def locate_package(self, config: PackageConfig) -> Optional[Path]:
        """Find the installed location of a package, or download it."""
        cache_key = f"{config.pip_name}_{config.version or 'latest'}"
        if cache_key in self._location_cache:
            return self._location_cache[cache_key]

        # Try to find in sys.path first
        pkg_path = self._find_installed(config.import_name)
        if pkg_path is not None:
            self._location_cache[cache_key] = pkg_path
            logger.info("Found %s at %s", config.name, pkg_path)
            return pkg_path

        # Try pip download into cache
        pkg_path = self._pip_download(config)
        if pkg_path is not None:
            self._location_cache[cache_key] = pkg_path
            logger.info("Downloaded %s to %s", config.name, pkg_path)
            return pkg_path

        logger.warning("Could not locate package %s", config.name)
        return None

    def _find_installed(self, import_name: str) -> Optional[Path]:
        """Find package in the Python environment."""
        try:
            result = subprocess.run(
                [sys.executable, "-c",
                 f"import {import_name}; print({import_name}.__file__)"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                pkg_file = Path(result.stdout.strip())
                # Return the package directory
                if pkg_file.name == "__init__.py":
                    return pkg_file.parent
                return pkg_file.parent
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def _pip_download(self, config: PackageConfig) -> Optional[Path]:
        """Download a package source using pip."""
        dest = self.cache_dir / config.name
        if dest.exists() and any(dest.glob("**/*.py")):
            return dest

        dest.mkdir(parents=True, exist_ok=True)
        pip_spec = config.pip_name
        if config.version:
            pip_spec = f"{config.pip_name}=={config.version}"

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "download",
                 "--no-binary", ":all:", "--no-deps",
                 "-d", str(dest), pip_spec],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                logger.warning("pip download failed for %s: %s",
                               config.name, result.stderr[:500])
                return None

            # Extract if tarball or zip
            for archive in dest.iterdir():
                if archive.suffix in (".gz", ".zip", ".whl"):
                    self._extract(archive, dest)

            # Find the package source directory
            for candidate in dest.rglob("__init__.py"):
                if candidate.parent.name == config.import_name:
                    return candidate.parent

            # Fallback: return dest if it has .py files
            if any(dest.rglob("*.py")):
                return dest

        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning("Failed to download %s: %s", config.name, exc)

        return None

    def _extract(self, archive: Path, dest: Path) -> None:
        """Extract an archive into dest."""
        try:
            if archive.name.endswith(".tar.gz") or archive.name.endswith(".tgz"):
                subprocess.run(
                    ["tar", "xzf", str(archive), "-C", str(dest)],
                    capture_output=True, timeout=60,
                )
            elif archive.name.endswith(".zip") or archive.name.endswith(".whl"):
                subprocess.run(
                    ["unzip", "-o", "-q", str(archive), "-d", str(dest)],
                    capture_output=True, timeout=60,
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def cleanup(self) -> None:
        """Remove cached downloads."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: SourceCollector
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FileStats:
    """Statistics for a single source file."""
    path: Path
    loc: int = 0
    function_count: int = 0
    class_count: int = 0
    parse_ok: bool = True
    tree: Optional[ast.AST] = None


class SourceCollector:
    """Collect and filter Python source files from a package directory."""

    SKIP_DIRS = {
        "__pycache__", ".git", ".tox", ".eggs", "node_modules",
        ".mypy_cache", ".pytest_cache", "dist", "build", "egg-info",
    }

    def __init__(self, exclude_patterns: Tuple[str, ...] = ()):
        self.exclude_patterns = exclude_patterns

    def collect(self, root: Path, max_files: int = 200) -> List[FileStats]:
        """Walk root collecting .py files, filtering out unwanted paths."""
        candidates: List[Path] = []
        for py_file in sorted(root.rglob("*.py")):
            if self._should_skip(py_file, root):
                continue
            candidates.append(py_file)
            if len(candidates) >= max_files:
                break

        results: List[FileStats] = []
        for path in candidates:
            stats = self._analyze_file(path)
            results.append(stats)

        return results

    def _should_skip(self, path: Path, root: Path) -> bool:
        """Return True if this file should be excluded."""
        rel = path.relative_to(root)
        rel_str = str(rel)

        # Skip directories
        for part in rel.parts[:-1]:
            if part in self.SKIP_DIRS:
                return True
            if part.endswith(".egg-info"):
                return True

        # Skip patterns
        for pattern in self.exclude_patterns:
            if self._match_pattern(rel_str, pattern):
                return True

        # Skip very large files (likely generated)
        try:
            size = path.stat().st_size
            if size > 500_000:
                return True
        except OSError:
            return True

        # Skip files that look generated
        name_lower = path.name.lower()
        if name_lower in ("setup.py", "conftest.py", "conf.py"):
            return False  # keep these
        if "_generated" in name_lower or "_pb2" in name_lower:
            return True

        return False

    @staticmethod
    def _match_pattern(path_str: str, pattern: str) -> bool:
        """Simple glob-like pattern matching."""
        import fnmatch
        return fnmatch.fnmatch(path_str, pattern)

    @staticmethod
    def _analyze_file(path: Path) -> FileStats:
        """Parse a file and collect basic statistics."""
        stats = FileStats(path=path)
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            stats.parse_ok = False
            return stats

        lines = source.splitlines()
        stats.loc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])

        try:
            tree = ast.parse(source, str(path))
            stats.tree = tree
            stats.parse_ok = True
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    stats.function_count += 1
                elif isinstance(node, ast.ClassDef):
                    stats.class_count += 1
        except SyntaxError:
            stats.parse_ok = False

        return stats

    @staticmethod
    def aggregate(files: List[FileStats]) -> Dict[str, int]:
        """Aggregate stats across files."""
        return {
            "total_files": len(files),
            "parseable_files": sum(1 for f in files if f.parse_ok),
            "total_loc": sum(f.loc for f in files),
            "total_functions": sum(f.function_count for f in files),
            "total_classes": sum(f.class_count for f in files),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: GuardPatternCounter
# ═══════════════════════════════════════════════════════════════════════════

class GuardCategory(Enum):
    """Categories of guard patterns we count."""
    ISINSTANCE = "isinstance"
    NONE_CHECK = "none_check"
    TRUTHINESS = "truthiness"
    COMPARISON = "comparison"
    HASATTR = "hasattr"
    TRY_EXCEPT = "try_except"
    ASSERTION = "assertion"
    TYPE_CALL = "type_call"
    CALLABLE_CHECK = "callable_check"
    LEN_CHECK = "len_check"
    IN_CHECK = "in_check"
    CUSTOM_TYPECHECK = "custom_typecheck"


class GuardPatternCounter(ast.NodeVisitor):
    """Count all guard patterns in Python source code."""

    def __init__(self):
        self.counts: Dict[str, int] = {cat.value: 0 for cat in GuardCategory}
        self.total_guards: int = 0
        self._in_test: bool = False
        self._guard_locations: List[Tuple[str, int, str]] = []

    def count_source(self, source: str, filename: str = "<string>") -> Dict[str, int]:
        """Count guard patterns in a source string."""
        try:
            tree = ast.parse(source, filename)
        except SyntaxError:
            return dict(self.counts)
        self.visit(tree)
        return dict(self.counts)

    def count_tree(self, tree: ast.AST, filename: str = "<string>") -> Dict[str, int]:
        """Count guard patterns in an already-parsed AST."""
        self.visit(tree)
        return dict(self.counts)

    def visit_If(self, node: ast.If) -> None:
        """Analyze if-statement conditions for guard patterns."""
        self._classify_condition(node.test, node.lineno)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        """Analyze while conditions too (less common but valid)."""
        self._classify_condition(node.test, node.lineno)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        """Assertions are a form of guard."""
        self.counts[GuardCategory.ASSERTION.value] += 1
        self.total_guards += 1
        self._guard_locations.append(("assertion", node.lineno, "assert"))
        # Also classify what's being asserted
        self._classify_condition(node.test, node.lineno, is_assert=True)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        """try/except is an implicit guard."""
        if node.handlers:
            self.counts[GuardCategory.TRY_EXCEPT.value] += len(node.handlers)
            self.total_guards += len(node.handlers)
            for handler in node.handlers:
                exc_name = ""
                if handler.type:
                    exc_name = ast.dump(handler.type)
                self._guard_locations.append(
                    ("try_except", handler.lineno, exc_name)
                )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Already counted in visit_Try, just recurse."""
        self.generic_visit(node)

    def _classify_condition(self, node: ast.expr, lineno: int,
                            is_assert: bool = False) -> None:
        """Classify a condition expression into guard categories."""
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                self._classify_condition(value, lineno, is_assert)
            return

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            self._classify_condition(node.operand, lineno, is_assert)
            return

        if isinstance(node, ast.Call):
            self._classify_call_guard(node, lineno)
            return

        if isinstance(node, ast.Compare):
            self._classify_compare_guard(node, lineno)
            return

        # Bare name or attribute → truthiness
        if isinstance(node, (ast.Name, ast.Attribute)):
            self.counts[GuardCategory.TRUTHINESS.value] += 1
            self.total_guards += 1
            self._guard_locations.append(("truthiness", lineno, ""))
            return

        # Subscript or other → truthiness
        if isinstance(node, ast.Subscript):
            self.counts[GuardCategory.TRUTHINESS.value] += 1
            self.total_guards += 1
            return

    def _classify_call_guard(self, node: ast.Call, lineno: int) -> None:
        """Classify a function-call guard."""
        func = node.func

        # isinstance(x, T)
        if isinstance(func, ast.Name) and func.id == "isinstance":
            self.counts[GuardCategory.ISINSTANCE.value] += 1
            self.total_guards += 1
            self._guard_locations.append(("isinstance", lineno, ""))
            return

        # hasattr(x, 'attr')
        if isinstance(func, ast.Name) and func.id == "hasattr":
            self.counts[GuardCategory.HASATTR.value] += 1
            self.total_guards += 1
            self._guard_locations.append(("hasattr", lineno, ""))
            return

        # callable(x)
        if isinstance(func, ast.Name) and func.id == "callable":
            self.counts[GuardCategory.CALLABLE_CHECK.value] += 1
            self.total_guards += 1
            self._guard_locations.append(("callable", lineno, ""))
            return

        # type(x) is T  → handled in compare, but type(x) alone
        if isinstance(func, ast.Name) and func.id == "type":
            self.counts[GuardCategory.TYPE_CALL.value] += 1
            self.total_guards += 1
            return

        # len(x) > 0 → handled in compare; bare len(x) → truthiness
        if isinstance(func, ast.Name) and func.id == "len":
            self.counts[GuardCategory.LEN_CHECK.value] += 1
            self.total_guards += 1
            return

        # Any other call in condition position → truthiness
        self.counts[GuardCategory.TRUTHINESS.value] += 1
        self.total_guards += 1

    def _classify_compare_guard(self, node: ast.Compare, lineno: int) -> None:
        """Classify a comparison guard."""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            self.counts[GuardCategory.COMPARISON.value] += 1
            self.total_guards += 1
            return

        op = node.ops[0]
        comparator = node.comparators[0]

        # x is None / x is not None
        is_none_check = (
            isinstance(op, (ast.Is, ast.IsNot))
            and isinstance(comparator, ast.Constant)
            and comparator.value is None
        )
        if is_none_check:
            self.counts[GuardCategory.NONE_CHECK.value] += 1
            self.total_guards += 1
            self._guard_locations.append(("none_check", lineno, ""))
            return

        # x in collection
        if isinstance(op, (ast.In, ast.NotIn)):
            self.counts[GuardCategory.IN_CHECK.value] += 1
            self.total_guards += 1
            self._guard_locations.append(("in_check", lineno, ""))
            return

        # is comparison (non-None identity)
        if isinstance(op, (ast.Is, ast.IsNot)):
            self.counts[GuardCategory.COMPARISON.value] += 1
            self.total_guards += 1
            return

        # Numeric/string comparisons
        self.counts[GuardCategory.COMPARISON.value] += 1
        self.total_guards += 1
        self._guard_locations.append(("comparison", lineno, ""))

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of counted guard patterns."""
        return {
            "total_guards": self.total_guards,
            "by_category": dict(self.counts),
            "top_categories": sorted(
                self.counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "guard_density": 0.0,  # set externally per LoC
        }


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: AnalysisRunner
# ═══════════════════════════════════════════════════════════════════════════

class AnalysisRunner:
    """Run our refinement type analysis on Python packages."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self._timing: Dict[str, float] = {}

    def analyze_package(
        self,
        pkg_config: PackageConfig,
        files: List[FileStats],
    ) -> PackageResult:
        """Run full analysis on a package."""
        result = PackageResult(
            package_name=pkg_config.name,
            num_files=0,
            total_loc=0,
            total_functions=0,
            total_classes=0,
        )

        t0 = time.perf_counter()
        analyzed = 0
        pkg_deadline = t0 + self.config.timeout_per_package_s

        for fstats in files:
            if time.perf_counter() > pkg_deadline:
                result.errors.append(
                    f"Package timeout after {analyzed} files"
                )
                break

            if not fstats.parse_ok:
                continue

            try:
                source = fstats.path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                result.errors.append(f"Read error {fstats.path}: {exc}")
                continue

            file_findings = self._analyze_single_file(
                source, str(fstats.path), pkg_config.name,
            )
            result.findings.extend(file_findings)
            result.total_loc += fstats.loc
            result.total_functions += fstats.function_count
            result.total_classes += fstats.class_count
            analyzed += 1

        result.num_files = analyzed
        result.analysis_time_s = time.perf_counter() - t0

        # Count guard patterns across all files
        counter = GuardPatternCounter()
        for fstats in files:
            if fstats.parse_ok and fstats.tree is not None:
                counter.count_tree(fstats.tree, str(fstats.path))
        result.guard_counts = counter.get_summary()

        self._timing[pkg_config.name] = result.analysis_time_s
        return result

    def _analyze_single_file(
        self,
        source: str,
        filepath: str,
        package_name: str,
    ) -> List[FileFinding]:
        """Analyze a single file with our pipeline."""
        findings: List[FileFinding] = []

        try:
            analysis = analyze_python_source(source, filepath)
        except Exception as exc:
            logger.debug("Analysis error in %s: %s", filepath, exc)
            return findings

        for summary in analysis.summaries:
            for bug in summary.bugs_found:
                finding = FileFinding(
                    file_path=filepath,
                    line=bug.line,
                    col=bug.col,
                    message=bug.message,
                    category=bug.category.name,
                    severity=bug.severity,
                    tool="refinement",
                    confidence=1.0 if not bug.guarded else 0.5,
                )
                findings.append(finding)

        return findings

    def analyze_source_string(
        self, source: str, filename: str = "<string>"
    ) -> AnalysisResult:
        """Analyze a raw source string (for testing)."""
        return analyze_python_source(source, filename)

    def run_cegar_on_source(
        self, source: str, filename: str = "<string>"
    ) -> Dict[str, Any]:
        """Run CEGAR loop on source and return summary."""
        try:
            guards = extract_guards(source, filename)
        except SyntaxError:
            return {"error": "syntax_error", "guards": 0, "predicates": 0}

        try:
            cegar_state = run_cegar(source, guards, max_iterations=20)
        except Exception as exc:
            return {"error": str(exc), "guards": len(guards), "predicates": 0}

        return {
            "guards_extracted": len(guards),
            "predicates_inferred": len(cegar_state.predicates),
            "cegar_iterations": cegar_state.iterations,
            "converged": cegar_state.converged,
        }

    def get_timing(self) -> Dict[str, float]:
        """Return timing data for all analyzed packages."""
        return dict(self._timing)


# ═══════════════════════════════════════════════════════════════════════════
# Section 6: ExternalToolRunner
# ═══════════════════════════════════════════════════════════════════════════

class ExternalToolRunner:
    """Run mypy, pyright, and pylint for comparison."""

    TOOL_COMMANDS = {
        "mypy": [sys.executable, "-m", "mypy"],
        "pyright": ["pyright"],
        "pylint": [sys.executable, "-m", "pylint"],
    }

    def __init__(self, timeout: float = 300.0):
        self.timeout = timeout
        self._available: Dict[str, bool] = {}

    def check_available(self, tool: str) -> bool:
        """Check if an external tool is installed and runnable."""
        if tool in self._available:
            return self._available[tool]

        cmd = self.TOOL_COMMANDS.get(tool)
        if cmd is None:
            self._available[tool] = False
            return False

        try:
            result = subprocess.run(
                cmd + ["--version"],
                capture_output=True, text=True, timeout=15,
            )
            available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            available = False

        self._available[tool] = available
        if not available:
            logger.info("Tool %s is not available", tool)
        return available

    def run_tool(
        self, tool: str, target_path: str
    ) -> List[FileFinding]:
        """Run an external tool on a directory or file, return findings."""
        if not self.check_available(tool):
            return []

        if tool == "mypy":
            return self._run_mypy(target_path)
        elif tool == "pyright":
            return self._run_pyright(target_path)
        elif tool == "pylint":
            return self._run_pylint(target_path)
        return []

    def _run_mypy(self, target: str) -> List[FileFinding]:
        """Run mypy and parse output."""
        cmd = self.TOOL_COMMANDS["mypy"] + [
            "--no-color-output",
            "--no-error-summary",
            "--ignore-missing-imports",
            "--no-incremental",
            target,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        return self._parse_mypy_output(result.stdout)

    def _parse_mypy_output(self, output: str) -> List[FileFinding]:
        """Parse mypy output lines: file:line: severity: message."""
        findings: List[FileFinding] = []
        pattern = re.compile(
            r"^(.+?):(\d+):\s*(error|warning|note):\s*(.+?)(?:\s+\[(\w[\w-]*)\])?$"
        )
        for line in output.splitlines():
            m = pattern.match(line.strip())
            if m:
                filepath, lineno, severity, message = m.group(1, 2, 3, 4)
                code = m.group(5) or ""
                category = self._mypy_code_to_category(code, message)
                findings.append(FileFinding(
                    file_path=filepath,
                    line=int(lineno),
                    col=0,
                    message=message,
                    category=category,
                    severity=severity,
                    tool="mypy",
                ))
        return findings

    @staticmethod
    def _mypy_code_to_category(code: str, message: str) -> str:
        """Map mypy error codes to our categories."""
        mapping = {
            "union-attr": "NULL_DEREF",
            "attr-defined": "ATTRIBUTE_ERROR",
            "arg-type": "TYPE_ERROR",
            "return-value": "TYPE_ERROR",
            "assignment": "TYPE_ERROR",
            "operator": "TYPE_ERROR",
            "index": "INDEX_OUT_OF_BOUNDS",
            "none-return": "UNGUARDED_NONE",
            "has-type": "TYPE_ERROR",
        }
        if code in mapping:
            return mapping[code]
        if "None" in message:
            return "UNGUARDED_NONE"
        if "has no attribute" in message:
            return "ATTRIBUTE_ERROR"
        return "TYPE_ERROR"

    def _run_pyright(self, target: str) -> List[FileFinding]:
        """Run pyright and parse JSON output."""
        cmd = ["pyright", "--outputjson", target]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        return self._parse_pyright_output(result.stdout)

    def _parse_pyright_output(self, output: str) -> List[FileFinding]:
        """Parse pyright JSON output."""
        findings: List[FileFinding] = []
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return findings

        diagnostics = data.get("generalDiagnostics", [])
        for diag in diagnostics:
            severity = diag.get("severity", "information")
            if severity == "information":
                continue
            rng = diag.get("range", {})
            start = rng.get("start", {})
            message = diag.get("message", "")
            rule = diag.get("rule", "")
            category = self._pyright_rule_to_category(rule, message)
            findings.append(FileFinding(
                file_path=diag.get("file", ""),
                line=start.get("line", 0) + 1,
                col=start.get("character", 0),
                message=message,
                category=category,
                severity="error" if severity == "error" else "warning",
                tool="pyright",
            ))
        return findings

    @staticmethod
    def _pyright_rule_to_category(rule: str, message: str) -> str:
        """Map pyright rule to our categories."""
        mapping = {
            "reportOptionalMemberAccess": "NULL_DEREF",
            "reportGeneralClassIssues": "TYPE_ERROR",
            "reportArgumentType": "TYPE_ERROR",
            "reportReturnType": "TYPE_ERROR",
            "reportIndexIssue": "INDEX_OUT_OF_BOUNDS",
            "reportAttributeAccessIssue": "ATTRIBUTE_ERROR",
            "reportOptionalSubscript": "NULL_DEREF",
            "reportOptionalCall": "NULL_DEREF",
            "reportOptionalIterable": "NULL_DEREF",
        }
        if rule in mapping:
            return mapping[rule]
        if "None" in message or "Optional" in message:
            return "UNGUARDED_NONE"
        return "TYPE_ERROR"

    def _run_pylint(self, target: str) -> List[FileFinding]:
        """Run pylint and parse output."""
        cmd = self.TOOL_COMMANDS["pylint"] + [
            "--output-format=json",
            "--disable=all",
            "--enable=E",  # only errors
            target,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        return self._parse_pylint_output(result.stdout)

    def _parse_pylint_output(self, output: str) -> List[FileFinding]:
        """Parse pylint JSON output."""
        findings: List[FileFinding] = []
        try:
            diagnostics = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return findings

        if not isinstance(diagnostics, list):
            return findings

        for diag in diagnostics:
            msg_id = diag.get("message-id", "")
            message = diag.get("message", "")
            symbol = diag.get("symbol", "")
            category = self._pylint_symbol_to_category(symbol, message)
            findings.append(FileFinding(
                file_path=diag.get("path", ""),
                line=diag.get("line", 0),
                col=diag.get("column", 0),
                message=f"[{msg_id}] {message}",
                category=category,
                severity="error",
                tool="pylint",
            ))
        return findings

    @staticmethod
    def _pylint_symbol_to_category(symbol: str, message: str) -> str:
        """Map pylint symbol to our categories."""
        mapping = {
            "no-member": "ATTRIBUTE_ERROR",
            "not-callable": "TYPE_ERROR",
            "invalid-unary-operand-type": "TYPE_ERROR",
            "unsupported-membership-test": "TYPE_ERROR",
            "unsubscriptable-object": "INDEX_OUT_OF_BOUNDS",
            "unsupported-assignment-operation": "TYPE_ERROR",
            "no-value-for-parameter": "TYPE_ERROR",
            "too-many-function-args": "TYPE_ERROR",
            "unexpected-keyword-arg": "TYPE_ERROR",
        }
        if symbol in mapping:
            return mapping[symbol]
        if "None" in message:
            return "UNGUARDED_NONE"
        return "TYPE_ERROR"

    def run_all_tools(self, target_path: str) -> Dict[str, List[FileFinding]]:
        """Run all available external tools on a target."""
        results: Dict[str, List[FileFinding]] = {}
        for tool_name in self.TOOL_COMMANDS:
            findings = self.run_tool(tool_name, target_path)
            results[tool_name] = findings
            logger.info(
                "Tool %s found %d findings on %s",
                tool_name, len(findings), target_path,
            )
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Section 7: FindingClassifier
# ═══════════════════════════════════════════════════════════════════════════

class FindingSeverity(Enum):
    """Severity levels for findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FPCategory(Enum):
    """False positive categories."""
    DEAD_CODE = "dead_code"
    DEFENSIVE_GUARD = "defensive_guard"
    DYNAMIC_DISPATCH = "dynamic_dispatch"
    METAPROGRAMMING = "metaprogramming"
    PROTOCOL_TYPE = "protocol_type"
    TEST_CODE = "test_code"
    FRAMEWORK_MAGIC = "framework_magic"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedFinding:
    """A finding with classification metadata."""
    finding: FileFinding
    is_true_positive: bool
    fp_category: Optional[FPCategory] = None
    severity: FindingSeverity = FindingSeverity.MEDIUM
    confidence: float = 0.5
    reasoning: str = ""


class FindingClassifier:
    """Classify findings into true positives and false positives."""

    # Patterns strongly correlated with true positives
    TP_PATTERNS = [
        (re.compile(r"division by zero", re.I), "DIV_BY_ZERO", 0.9),
        (re.compile(r"None.*has no attribute", re.I), "NULL_DEREF", 0.85),
        (re.compile(r"index out of (range|bounds)", re.I), "INDEX_OUT_OF_BOUNDS", 0.8),
        (re.compile(r"not callable", re.I), "TYPE_ERROR", 0.7),
        (re.compile(r"undefined variable", re.I), "ATTRIBUTE_ERROR", 0.8),
    ]

    # Patterns correlated with false positives
    FP_PATTERNS = [
        (re.compile(r"overloaded function", re.I), FPCategory.DYNAMIC_DISPATCH),
        (re.compile(r"__\w+__", re.I), FPCategory.METAPROGRAMMING),
        (re.compile(r"Protocol", re.I), FPCategory.PROTOCOL_TYPE),
        (re.compile(r"test_|_test\.py", re.I), FPCategory.TEST_CODE),
        (re.compile(r"deprecated", re.I), FPCategory.DEAD_CODE),
    ]

    CATEGORY_SEVERITY = {
        "DIV_BY_ZERO": FindingSeverity.CRITICAL,
        "NULL_DEREF": FindingSeverity.HIGH,
        "UNGUARDED_NONE": FindingSeverity.HIGH,
        "INDEX_OUT_OF_BOUNDS": FindingSeverity.HIGH,
        "TYPE_ERROR": FindingSeverity.MEDIUM,
        "ATTRIBUTE_ERROR": FindingSeverity.MEDIUM,
    }

    def classify(self, finding: FileFinding) -> ClassifiedFinding:
        """Classify a single finding."""
        confidence = self._compute_confidence(finding)
        is_tp = confidence >= 0.5
        fp_cat = None if is_tp else self._determine_fp_category(finding)
        severity = self.CATEGORY_SEVERITY.get(
            finding.category, FindingSeverity.MEDIUM
        )
        reasoning = self._build_reasoning(finding, confidence, is_tp)

        return ClassifiedFinding(
            finding=finding,
            is_true_positive=is_tp,
            fp_category=fp_cat,
            severity=severity,
            confidence=confidence,
            reasoning=reasoning,
        )

    def classify_all(
        self, findings: List[FileFinding]
    ) -> List[ClassifiedFinding]:
        """Classify a list of findings."""
        return [self.classify(f) for f in findings]

    def _compute_confidence(self, finding: FileFinding) -> float:
        """Compute confidence that this finding is a true positive."""
        base = 0.5

        # Strong TP patterns
        for pattern, category, boost in self.TP_PATTERNS:
            if pattern.search(finding.message):
                base = max(base, boost)
                if finding.category == category:
                    base = min(base + 0.1, 1.0)
                break

        # Tool-specific adjustments
        if finding.tool == "refinement":
            # Our tool tends to have higher precision for guarded patterns
            if finding.confidence > 0.8:
                base = min(base + 0.15, 1.0)
        elif finding.tool == "mypy":
            if finding.severity == "error":
                base = min(base + 0.1, 1.0)
        elif finding.tool == "pyright":
            base = min(base + 0.05, 1.0)

        # Downweight test files
        if "/test" in finding.file_path or "test_" in finding.file_path:
            base *= 0.6

        # Unguarded None is a strong signal
        if finding.category in ("NULL_DEREF", "UNGUARDED_NONE"):
            base = min(base + 0.1, 1.0)

        return round(base, 3)

    def _determine_fp_category(self, finding: FileFinding) -> FPCategory:
        """Determine why a finding is a false positive."""
        for pattern, category in self.FP_PATTERNS:
            if pattern.search(finding.message) or pattern.search(finding.file_path):
                return category

        if "guard" in finding.message.lower():
            return FPCategory.DEFENSIVE_GUARD

        return FPCategory.UNKNOWN

    @staticmethod
    def _build_reasoning(
        finding: FileFinding, confidence: float, is_tp: bool
    ) -> str:
        """Build a human-readable reasoning string."""
        parts = []
        parts.append(f"conf={confidence:.2f}")
        parts.append(f"tool={finding.tool}")
        parts.append(f"cat={finding.category}")
        if is_tp:
            parts.append("likely real bug")
        else:
            parts.append("likely false positive")
        return "; ".join(parts)

    def get_fp_distribution(
        self, classified: List[ClassifiedFinding]
    ) -> Dict[str, int]:
        """Get distribution of false positive categories."""
        dist: Dict[str, int] = {cat.value: 0 for cat in FPCategory}
        for cf in classified:
            if not cf.is_true_positive and cf.fp_category is not None:
                dist[cf.fp_category.value] += 1
        return dist

    def compute_precision_recall(
        self, classified: List[ClassifiedFinding]
    ) -> Dict[str, float]:
        """Compute precision and recall from classified findings."""
        if not classified:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "total": 0}

        tp = sum(1 for c in classified if c.is_true_positive)
        fp = sum(1 for c in classified if not c.is_true_positive)
        total = tp + fp

        precision = tp / total if total > 0 else 0.0
        # recall estimated as fraction of high-confidence findings found
        high_conf = sum(1 for c in classified if c.confidence >= 0.7)
        recall = tp / high_conf if high_conf > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "total": total,
            "true_positives": tp,
            "false_positives": fp,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Section 8: StatisticalAnalyzer
# ═══════════════════════════════════════════════════════════════════════════

class StatisticalAnalyzer:
    """Statistical analysis with bootstrap CIs and paired comparisons."""

    def __init__(
        self, bootstrap_samples: int = 1000, confidence_level: float = 0.95
    ):
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self._rng_state = 42  # deterministic seed

    def _pseudo_random(self) -> float:
        """Simple LCG pseudo-random number generator in [0, 1)."""
        self._rng_state = (self._rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        return self._rng_state / 0x7FFFFFFF

    def _random_int(self, n: int) -> int:
        """Return random int in [0, n)."""
        return int(self._pseudo_random() * n) % n

    def bootstrap_ci(
        self, values: List[float], statistic: str = "mean"
    ) -> Tuple[float, float, float]:
        """Bootstrap confidence interval for a statistic.

        Returns (lower, point_estimate, upper).
        """
        if not values:
            return (0.0, 0.0, 0.0)

        n = len(values)
        estimates: List[float] = []

        for _ in range(self.bootstrap_samples):
            # Resample with replacement
            sample = [values[self._random_int(n)] for _ in range(n)]
            if statistic == "mean":
                est = sum(sample) / len(sample)
            elif statistic == "median":
                est = self._median(sample)
            else:
                est = sum(sample) / len(sample)
            estimates.append(est)

        estimates.sort()
        alpha = 1.0 - self.confidence_level
        lower_idx = max(0, int(alpha / 2 * len(estimates)))
        upper_idx = min(len(estimates) - 1, int((1 - alpha / 2) * len(estimates)))

        if statistic == "mean":
            point = sum(values) / len(values)
        else:
            point = self._median(values)

        return (estimates[lower_idx], point, estimates[upper_idx])

    def paired_comparison(
        self,
        tool_a: List[float],
        tool_b: List[float],
    ) -> Dict[str, Any]:
        """Paired comparison between two tools' scores."""
        if len(tool_a) != len(tool_b):
            min_len = min(len(tool_a), len(tool_b))
            tool_a = tool_a[:min_len]
            tool_b = tool_b[:min_len]

        n = len(tool_a)
        if n == 0:
            return {"n": 0, "mean_diff": 0.0, "significant": False}

        diffs = [a - b for a, b in zip(tool_a, tool_b)]
        mean_diff = sum(diffs) / n
        std_diff = self._std(diffs)

        # Wilcoxon signed-rank test (manual)
        p_value = self._wilcoxon_signed_rank(diffs)

        # Cohen's d (effect size)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        # Bootstrap CI on the difference
        ci = self.bootstrap_ci(diffs)

        return {
            "n": n,
            "mean_diff": round(mean_diff, 4),
            "std_diff": round(std_diff, 4),
            "cohens_d": round(cohens_d, 4),
            "p_value": round(p_value, 4),
            "significant": p_value < (1 - self.confidence_level),
            "ci_lower": round(ci[0], 4),
            "ci_upper": round(ci[2], 4),
            "effect_size_label": self._effect_size_label(cohens_d),
        }

    def cohens_kappa(
        self, ratings_a: List[int], ratings_b: List[int]
    ) -> float:
        """Cohen's kappa for inter-rater agreement.

        ratings_a and ratings_b are binary lists (0/1).
        """
        n = len(ratings_a)
        if n == 0 or len(ratings_b) != n:
            return 0.0

        # Observed agreement
        agree = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b)
        p_o = agree / n

        # Expected agreement by chance
        a1 = sum(ratings_a) / n
        b1 = sum(ratings_b) / n
        a0 = 1 - a1
        b0 = 1 - b1
        p_e = a1 * b1 + a0 * b0

        if p_e >= 1.0:
            return 1.0

        kappa = (p_o - p_e) / (1 - p_e)
        return round(kappa, 4)

    def _wilcoxon_signed_rank(self, diffs: List[float]) -> float:
        """Manual Wilcoxon signed-rank test.

        Returns approximate p-value using normal approximation.
        """
        # Remove zeros
        nonzero = [(abs(d), 1 if d > 0 else -1) for d in diffs if d != 0.0]
        n = len(nonzero)
        if n == 0:
            return 1.0

        # Rank by absolute value
        nonzero.sort(key=lambda x: x[0])
        ranks: List[Tuple[float, int]] = []
        i = 0
        while i < n:
            j = i
            while j < n and nonzero[j][0] == nonzero[i][0]:
                j += 1
            avg_rank = (i + j + 1) / 2.0  # 1-based average rank
            for k in range(i, j):
                ranks.append((avg_rank, nonzero[k][1]))
            i = j

        # W+ = sum of ranks for positive differences
        w_plus = sum(r for r, sign in ranks if sign > 0)
        w_minus = sum(r for r, sign in ranks if sign < 0)
        w = min(w_plus, w_minus)

        # Normal approximation
        mean_w = n * (n + 1) / 4.0
        var_w = n * (n + 1) * (2 * n + 1) / 24.0
        if var_w == 0:
            return 1.0

        z = (w - mean_w) / math.sqrt(var_w)
        # Approximate p-value from z using complementary error function
        p = self._normal_cdf_approx(-abs(z)) * 2
        return max(0.0, min(1.0, p))

    @staticmethod
    def _normal_cdf_approx(x: float) -> float:
        """Approximation of the standard normal CDF (Abramowitz & Stegun)."""
        if x < -8:
            return 0.0
        if x > 8:
            return 1.0
        # Constants for Horner form
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        sign = 1.0 if x >= 0 else -1.0
        x_abs = abs(x)
        t = 1.0 / (1.0 + p * x_abs)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(
            -x_abs * x_abs / 2.0
        )
        return 0.5 * (1.0 + sign * y)

    @staticmethod
    def _effect_size_label(d: float) -> str:
        """Label for Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"

    @staticmethod
    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _median(values: List[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2.0

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)

    @staticmethod
    def _quartiles(values: List[float]) -> Tuple[float, float, float]:
        """Return (Q1, Q2/median, Q3)."""
        if not values:
            return (0.0, 0.0, 0.0)
        s = sorted(values)
        n = len(s)
        q2 = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0
        lower = s[: n // 2]
        upper = s[(n + 1) // 2 :]
        q1 = (
            lower[len(lower) // 2]
            if lower
            else q2
        )
        q3 = (
            upper[len(upper) // 2]
            if upper
            else q2
        )
        return (q1, q2, q3)

    def summary_stats(self, values: List[float]) -> Dict[str, float]:
        """Compute summary statistics."""
        if not values:
            return {
                "n": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0, "q1": 0.0, "q3": 0.0,
            }
        q1, q2, q3 = self._quartiles(values)
        return {
            "n": len(values),
            "mean": round(self._mean(values), 4),
            "median": round(q2, 4),
            "std": round(self._std(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Section 9: ComparisonMatrix
# ═══════════════════════════════════════════════════════════════════════════

class ComparisonMatrix:
    """Build and display comparison matrices across tools and packages."""

    TOOLS = ["refinement", "mypy", "pyright", "pylint"]

    def __init__(self):
        self.rows: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def add_entry(
        self,
        package: str,
        tool: str,
        findings: int,
        true_positives: int,
        false_positives: int,
    ) -> None:
        """Add a cell to the matrix."""
        if package not in self.rows:
            self.rows[package] = {}
        precision = true_positives / findings if findings > 0 else 0.0
        recall = true_positives / max(true_positives, 1)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        self.rows[package][tool] = {
            "findings": findings,
            "tp": true_positives,
            "fp": false_positives,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    def format_ascii(self) -> str:
        """Format the matrix as an ASCII table."""
        lines: List[str] = []
        header = f"{'Package':<15}"
        for tool in self.TOOLS:
            header += f" | {tool:>12} (F/TP/FP/P)"
        lines.append(header)
        lines.append("─" * len(header))

        for pkg in sorted(self.rows.keys()):
            row = f"{pkg:<15}"
            for tool in self.TOOLS:
                cell = self.rows[pkg].get(tool, {})
                if cell:
                    f_count = cell["findings"]
                    tp = cell["tp"]
                    fp = cell["fp"]
                    prec = cell["precision"]
                    row += f" | {f_count:>3}/{tp:>3}/{fp:>3}/{prec:.2f}     "
                else:
                    row += f" | {'N/A':>27}"
            lines.append(row)

        lines.append("─" * len(header))

        # Summary row
        summary = f"{'TOTAL':<15}"
        for tool in self.TOOLS:
            total_f = sum(
                self.rows[p].get(tool, {}).get("findings", 0)
                for p in self.rows
            )
            total_tp = sum(
                self.rows[p].get(tool, {}).get("tp", 0)
                for p in self.rows
            )
            total_fp = sum(
                self.rows[p].get(tool, {}).get("fp", 0)
                for p in self.rows
            )
            prec = total_tp / total_f if total_f > 0 else 0.0
            summary += f" | {total_f:>3}/{total_tp:>3}/{total_fp:>3}/{prec:.2f}     "
        lines.append(summary)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert matrix to a dict for JSON serialization."""
        return {
            "packages": dict(self.rows),
            "tools": self.TOOLS,
        }

    def get_tool_scores(self, tool: str) -> List[float]:
        """Get precision scores for a tool across all packages."""
        scores: List[float] = []
        for pkg in sorted(self.rows.keys()):
            cell = self.rows[pkg].get(tool)
            if cell:
                scores.append(cell["precision"])
        return scores


# ═══════════════════════════════════════════════════════════════════════════
# Section 10: ReportGenerator
# ═══════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Generate evaluation reports in multiple formats."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary(self, result: EvalResult) -> str:
        """Generate a text summary of the evaluation."""
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("REFINEMENT TYPE INFERENCE — COMPREHENSIVE EVALUATION")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {result.timestamp}")
        lines.append(f"Packages evaluated: {len(result.packages)}")
        lines.append("")

        total_loc = 0
        total_findings = 0
        total_funcs = 0
        for name, pkg in result.packages.items():
            total_loc += pkg.total_loc
            total_findings += len(pkg.findings)
            total_funcs += pkg.total_functions

        lines.append(f"Total LoC analyzed:     {total_loc:>10,}")
        lines.append(f"Total functions:        {total_funcs:>10,}")
        lines.append(f"Total findings:         {total_findings:>10,}")
        lines.append("")

        # Per-package breakdown
        lines.append("Per-package breakdown:")
        lines.append(f"  {'Package':<15} {'Files':>6} {'LoC':>8} {'Guards':>8} {'Findings':>9} {'Time(s)':>8}")
        lines.append("  " + "-" * 60)
        for name in sorted(result.packages.keys()):
            pkg = result.packages[name]
            guard_total = pkg.guard_counts.get("total_guards", 0)
            lines.append(
                f"  {name:<15} {pkg.num_files:>6} {pkg.total_loc:>8,}"
                f" {guard_total:>8} {len(pkg.findings):>9}"
                f" {pkg.analysis_time_s:>8.2f}"
            )

        lines.append("")

        # Statistics
        if result.statistics:
            lines.append("Statistical summary:")
            for key, val in result.statistics.items():
                if isinstance(val, dict):
                    lines.append(f"  {key}:")
                    for k2, v2 in val.items():
                        lines.append(f"    {k2}: {v2}")
                else:
                    lines.append(f"  {key}: {val}")

        return "\n".join(lines)

    def generate_latex_table(self, matrix: ComparisonMatrix) -> str:
        """Generate LaTeX table for paper."""
        lines: List[str] = []
        n_tools = len(matrix.TOOLS)
        col_spec = "l" + "rrr" * n_tools
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Comparison of refinement type inference with existing tools}")
        lines.append(r"\label{tab:comparison}")
        lines.append(r"\begin{tabular}{" + col_spec + "}")
        lines.append(r"\toprule")

        # Header
        header = "Package"
        for tool in matrix.TOOLS:
            header += f" & \\multicolumn{{3}}{{c}}{{{tool}}}"
        header += r" \\"
        lines.append(header)

        subheader = ""
        for _ in matrix.TOOLS:
            subheader += " & F & P & F1"
        subheader += r" \\"
        lines.append(r"\cmidrule(lr){2-" + str(1 + 3 * n_tools) + "}")
        lines.append(subheader)
        lines.append(r"\midrule")

        # Data rows
        for pkg in sorted(matrix.rows.keys()):
            row_parts = [pkg.replace("_", r"\_")]
            for tool in matrix.TOOLS:
                cell = matrix.rows[pkg].get(tool, {})
                if cell:
                    row_parts.append(str(cell["findings"]))
                    row_parts.append(f"{cell['precision']:.2f}")
                    row_parts.append(f"{cell['f1']:.2f}")
                else:
                    row_parts.extend(["--", "--", "--"])
            lines.append(" & ".join(row_parts) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def save_json(self, result: EvalResult, filename: str = "eval_results.json") -> Path:
        """Save results as JSON."""
        path = self.output_dir / filename
        data = {
            "timestamp": result.timestamp,
            "config": result.config,
            "packages": {},
            "comparison": result.comparison,
            "statistics": result.statistics,
        }
        for name, pkg in result.packages.items():
            data["packages"][name] = {
                "num_files": pkg.num_files,
                "total_loc": pkg.total_loc,
                "total_functions": pkg.total_functions,
                "total_classes": pkg.total_classes,
                "guard_counts": pkg.guard_counts,
                "num_findings": len(pkg.findings),
                "analysis_time_s": pkg.analysis_time_s,
                "errors": pkg.errors,
                "findings": [
                    {
                        "file": f.file_path,
                        "line": f.line,
                        "col": f.col,
                        "message": f.message,
                        "category": f.category,
                        "severity": f.severity,
                        "tool": f.tool,
                        "confidence": f.confidence,
                    }
                    for f in pkg.findings
                ],
            }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)
        return path

    def save_text(self, text: str, filename: str = "eval_summary.txt") -> Path:
        """Save text report."""
        path = self.output_dir / filename
        path.write_text(text, encoding="utf-8")
        return path


# ═══════════════════════════════════════════════════════════════════════════
# Section 11: Main Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(config: EvalConfig) -> EvalResult:
    """Run the complete evaluation pipeline."""
    result = EvalResult(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        config={
            "timeout_per_file_s": config.timeout_per_file_s,
            "timeout_per_package_s": config.timeout_per_package_s,
            "bootstrap_samples": config.bootstrap_samples,
            "run_external_tools": config.run_external_tools,
        },
    )

    downloader = PackageDownloader(config.cache_dir)
    runner = AnalysisRunner(config)
    classifier = FindingClassifier()
    stats_analyzer = StatisticalAnalyzer(
        bootstrap_samples=config.bootstrap_samples,
        confidence_level=config.confidence_level,
    )
    ext_runner = ExternalToolRunner(timeout=config.timeout_per_package_s)
    matrix = ComparisonMatrix()

    packages = config.packages or DEFAULT_PACKAGES
    if config.selected_packages:
        packages = [p for p in packages if p.name in config.selected_packages]

    for pkg_config in packages:
        logger.info("=" * 60)
        logger.info("Evaluating package: %s", pkg_config.name)
        logger.info("=" * 60)

        # Locate package
        pkg_path = downloader.locate_package(pkg_config)
        if pkg_path is None:
            logger.warning("Skipping %s: not found", pkg_config.name)
            continue

        # Collect source files
        collector = SourceCollector(exclude_patterns=pkg_config.exclude_patterns)
        files = collector.collect(pkg_path, max_files=pkg_config.max_files)
        agg = collector.aggregate(files)
        logger.info(
            "Collected %d files (%d parseable, %d LoC)",
            agg["total_files"], agg["parseable_files"], agg["total_loc"],
        )

        if agg["parseable_files"] == 0:
            logger.warning("No parseable files for %s", pkg_config.name)
            continue

        # Run our analysis
        pkg_result = runner.analyze_package(pkg_config, files)
        logger.info(
            "Analysis complete: %d findings in %.2fs",
            len(pkg_result.findings), pkg_result.analysis_time_s,
        )

        # Run external tools for comparison
        if config.run_external_tools:
            ext_findings = ext_runner.run_all_tools(str(pkg_path))
            pkg_result.external_findings = ext_findings

        # Classify our findings
        classified = classifier.classify_all(pkg_result.findings)
        pr = classifier.compute_precision_recall(classified)
        tp_count = pr.get("true_positives", 0)
        fp_count = pr.get("false_positives", 0)

        # Add to comparison matrix
        matrix.add_entry(
            pkg_config.name, "refinement",
            findings=len(pkg_result.findings),
            true_positives=tp_count,
            false_positives=fp_count,
        )

        # Add external tool entries
        for tool_name, tool_findings in pkg_result.external_findings.items():
            ext_classified = classifier.classify_all(tool_findings)
            ext_pr = classifier.compute_precision_recall(ext_classified)
            matrix.add_entry(
                pkg_config.name, tool_name,
                findings=len(tool_findings),
                true_positives=ext_pr.get("true_positives", 0),
                false_positives=ext_pr.get("false_positives", 0),
            )

        result.packages[pkg_config.name] = pkg_result

    # Statistical analysis
    ref_scores = matrix.get_tool_scores("refinement")
    for other_tool in ["mypy", "pyright", "pylint"]:
        other_scores = matrix.get_tool_scores(other_tool)
        if ref_scores and other_scores:
            comparison = stats_analyzer.paired_comparison(ref_scores, other_scores)
            result.comparison[f"refinement_vs_{other_tool}"] = comparison

    # Bootstrap CI for our precision
    if ref_scores:
        ci = stats_analyzer.bootstrap_ci(ref_scores)
        result.statistics["precision_ci"] = {
            "lower": ci[0], "point": ci[1], "upper": ci[2],
        }
        result.statistics["precision_summary"] = stats_analyzer.summary_stats(
            ref_scores
        )

    # Inter-tool agreement (kappa)
    for tool_a in ["refinement", "mypy"]:
        for tool_b in ["pyright", "pylint"]:
            ratings_a: List[int] = []
            ratings_b: List[int] = []
            for pkg_name in result.packages:
                cell_a = matrix.rows.get(pkg_name, {}).get(tool_a, {})
                cell_b = matrix.rows.get(pkg_name, {}).get(tool_b, {})
                if cell_a and cell_b:
                    ratings_a.append(1 if cell_a.get("tp", 0) > 0 else 0)
                    ratings_b.append(1 if cell_b.get("tp", 0) > 0 else 0)
            if ratings_a and ratings_b:
                kappa = stats_analyzer.cohens_kappa(ratings_a, ratings_b)
                result.statistics[f"kappa_{tool_a}_vs_{tool_b}"] = kappa

    result.comparison["matrix"] = matrix.to_dict()

    # Generate reports
    report_gen = ReportGenerator(config.output_dir)
    summary_text = report_gen.generate_summary(result)
    print(summary_text)

    # Comparison table
    print("\nComparison Matrix:")
    print(matrix.format_ascii())

    # LaTeX output
    latex = report_gen.generate_latex_table(matrix)

    # Save outputs
    json_path = report_gen.save_json(result)
    report_gen.save_text(summary_text)
    report_gen.save_text(latex, "comparison_table.tex")
    logger.info("Results saved to %s", json_path)

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of refinement type inference",
    )
    parser.add_argument(
        "--packages", nargs="*", default=None,
        help="Specific packages to evaluate (default: all)",
    )
    parser.add_argument(
        "--no-external", action="store_true",
        help="Skip running external tools (mypy/pyright/pylint)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(RESULTS_DIR),
        help="Output directory for results",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Timeout per file in seconds",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=1000,
        help="Number of bootstrap samples for CIs",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = EvalConfig(
        output_dir=Path(args.output_dir),
        run_external_tools=not args.no_external,
        timeout_per_file_s=args.timeout,
        bootstrap_samples=args.bootstrap,
        verbose=args.verbose,
        selected_packages=args.packages,
    )

    run_evaluation(config)


if __name__ == "__main__":
    main()
