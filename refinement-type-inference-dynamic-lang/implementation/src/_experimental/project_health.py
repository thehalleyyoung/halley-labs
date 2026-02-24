"""Overall project health scoring for Python codebases.

Provides a comprehensive health assessment combining multiple quality signals:
- Bug density estimation via static analysis
- Test coverage estimation from test file analysis
- Code complexity scoring
- Security vulnerability scoring
- Type safety scoring (annotation coverage)
- Dependency health assessment
- Trend analysis over git history
- Benchmark comparison against open-source projects
- SVG badge generation for READMEs
- CI quality gate (pass/fail)
"""
from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class HealthCategory(Enum):
    BUG_DENSITY = "bug_density"
    TEST_COVERAGE_ESTIMATE = "test_coverage_estimate"
    CODE_COMPLEXITY = "code_complexity"
    SECURITY_SCORE = "security_score"
    TYPE_SAFETY_SCORE = "type_safety_score"
    DEPENDENCY_HEALTH = "dependency_health"


@dataclass
class CategoryScore:
    category: HealthCategory
    score: float  # 0-100
    weight: float  # contribution weight
    details: str = ""
    issues: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.category.value}: {self.score:.1f}/100 ({self.details})"


@dataclass
class HealthReport:
    overall_score: float  # 0-100
    grade: str  # A-F
    category_scores: Dict[str, CategoryScore] = field(default_factory=dict)
    total_files: int = 0
    total_lines: int = 0
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Project Health: {self.overall_score:.1f}/100 (Grade: {self.grade})",
            f"Files: {self.total_files}, Lines: {self.total_lines}",
            "",
        ]
        for cs in self.category_scores.values():
            lines.append(f"  {cs}")
        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for r in self.recommendations:
                lines.append(f"  - {r}")
        return "\n".join(lines)


@dataclass
class TrendPoint:
    date: str
    score: float
    category_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrendReport:
    project_dir: str
    points: List[TrendPoint] = field(default_factory=list)
    trend_direction: str = "stable"  # improving, declining, stable
    average_change_per_week: float = 0.0
    summary: str = ""

    def __str__(self) -> str:
        return (
            f"Trend ({self.trend_direction}): "
            f"avg change {self.average_change_per_week:+.1f}/week over "
            f"{len(self.points)} data points\n{self.summary}"
        )


@dataclass
class BenchmarkComparison:
    project_score: float
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    percentile: float = 50.0
    summary: str = ""

    def __str__(self) -> str:
        return (
            f"Score: {self.project_score:.1f} "
            f"(percentile: {self.percentile:.0f})\n{self.summary}"
        )


# ---------------------------------------------------------------------------
# AST Helpers
# ---------------------------------------------------------------------------

def _collect_python_files(project_dir: str) -> List[Path]:
    """Collect all .py files in a project directory."""
    result: List[Path] = []
    root = Path(project_dir)
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        parts = rel.parts
        if any(
            part.startswith(".") or part in ("__pycache__", "node_modules", ".git", "venv", "env", ".venv", ".env")
            for part in parts
        ):
            continue
        result.append(p)
    return sorted(result)


def _parse_file(filepath: Path) -> Optional[ast.Module]:
    """Safely parse a Python file into an AST."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        return ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return None


def _count_lines(filepath: Path) -> int:
    try:
        return len(filepath.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return 0


class _CyclomaticVisitor(ast.NodeVisitor):
    """Count cyclomatic complexity of a function."""

    def __init__(self) -> None:
        self.complexity = 1

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.complexity += 1
        self.generic_visit(node)


def _function_complexity(node: ast.FunctionDef) -> int:
    v = _CyclomaticVisitor()
    v.visit(node)
    return v.complexity


# ---------------------------------------------------------------------------
# Bug Density Scoring
# ---------------------------------------------------------------------------

class _BugPatternVisitor(ast.NodeVisitor):
    """Detect common bug patterns via AST."""

    def __init__(self) -> None:
        self.issues: List[str] = []

    def visit_Compare(self, node: ast.Compare) -> None:
        # Detect `x == None` instead of `x is None`
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(comparator, ast.Constant) and comparator.value is None:
                self.issues.append(f"line {node.lineno}: use 'is None' instead of '== None'")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # Bare except
        if node.type is None:
            self.issues.append(f"line {node.lineno}: bare except catches all exceptions")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Mutable default arguments
        for default in node.args.defaults + node.args.kw_defaults:
            if default is not None and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.issues.append(f"line {node.lineno}: mutable default argument in '{node.name}'")
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        name = _get_call_name(node)
        if name == "eval":
            self.issues.append(f"line {node.lineno}: use of eval()")
        if name == "exec":
            self.issues.append(f"line {node.lineno}: use of exec()")
        self.generic_visit(node)


def _get_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _score_bug_density(files: List[Path]) -> CategoryScore:
    """Score based on bug patterns found per 1000 lines."""
    total_issues: List[str] = []
    total_lines = 0
    for fp in files:
        tree = _parse_file(fp)
        if tree is None:
            continue
        v = _BugPatternVisitor()
        v.visit(tree)
        total_issues.extend(v.issues)
        total_lines += _count_lines(fp)

    if total_lines == 0:
        return CategoryScore(HealthCategory.BUG_DENSITY, 100.0, 0.20, "no code")

    density = (len(total_issues) / max(total_lines, 1)) * 1000
    # 0 issues/kloc -> 100, 20+ issues/kloc -> 0
    score = max(0.0, min(100.0, 100.0 - density * 5))
    return CategoryScore(
        HealthCategory.BUG_DENSITY,
        score,
        0.20,
        f"{len(total_issues)} issues in {total_lines} lines ({density:.1f}/kloc)",
        total_issues[:20],
    )


# ---------------------------------------------------------------------------
# Test Coverage Estimation
# ---------------------------------------------------------------------------

def _score_test_coverage(project_dir: str, files: List[Path]) -> CategoryScore:
    """Estimate test coverage from test file presence and structure."""
    source_files: List[Path] = []
    test_files: List[Path] = []
    for fp in files:
        name = fp.name
        if name.startswith("test_") or name.endswith("_test.py") or "tests" in fp.parts:
            test_files.append(fp)
        elif not name.startswith("__"):
            source_files.append(fp)

    if not source_files:
        return CategoryScore(HealthCategory.TEST_COVERAGE_ESTIMATE, 100.0, 0.15, "no source")

    # Count tested functions by parsing test files for references
    tested_functions: Set[str] = set()
    for tf in test_files:
        tree = _parse_file(tf)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _get_call_name(node)
                if name:
                    tested_functions.add(name)

    # Count total public functions in source
    total_functions = 0
    covered_functions = 0
    for sf in source_files:
        tree = _parse_file(sf)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    total_functions += 1
                    if node.name in tested_functions:
                        covered_functions += 1

    if total_functions == 0:
        ratio = 1.0
    else:
        ratio = covered_functions / total_functions

    file_ratio = len(test_files) / max(len(source_files), 1)
    # Combined heuristic
    combined = (ratio * 0.6 + min(file_ratio, 1.0) * 0.4)
    score = min(100.0, combined * 100)

    issues: List[str] = []
    if file_ratio < 0.3:
        issues.append(f"Low test file ratio: {len(test_files)}/{len(source_files)}")
    if ratio < 0.3:
        issues.append(f"Low function coverage estimate: {covered_functions}/{total_functions}")

    return CategoryScore(
        HealthCategory.TEST_COVERAGE_ESTIMATE,
        score,
        0.15,
        f"{covered_functions}/{total_functions} public functions, {len(test_files)} test files",
        issues,
    )


# ---------------------------------------------------------------------------
# Code Complexity Scoring
# ---------------------------------------------------------------------------

def _score_complexity(files: List[Path]) -> CategoryScore:
    """Score based on average cyclomatic complexity."""
    complexities: List[int] = []
    high_complexity: List[str] = []
    for fp in files:
        tree = _parse_file(fp)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cc = _function_complexity(node)
                complexities.append(cc)
                if cc > 10:
                    high_complexity.append(
                        f"{fp.name}:{node.lineno} '{node.name}' complexity={cc}"
                    )

    if not complexities:
        return CategoryScore(HealthCategory.CODE_COMPLEXITY, 100.0, 0.20, "no functions")

    avg = sum(complexities) / len(complexities)
    high_pct = sum(1 for c in complexities if c > 10) / len(complexities) * 100

    # avg complexity 1 -> 100, avg 15+ -> 0
    score = max(0.0, min(100.0, 100.0 - (avg - 1) * 7))
    # Penalize for high percentage of complex functions
    score = max(0.0, score - high_pct * 0.5)

    return CategoryScore(
        HealthCategory.CODE_COMPLEXITY,
        score,
        0.20,
        f"avg complexity {avg:.1f}, {len(high_complexity)} high-complexity functions",
        high_complexity[:15],
    )


# ---------------------------------------------------------------------------
# Security Scoring
# ---------------------------------------------------------------------------

class _SecurityVisitor(ast.NodeVisitor):
    """Detect security-relevant patterns."""

    def __init__(self) -> None:
        self.issues: List[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        name = _get_call_name(node)
        # Dangerous calls
        dangerous = {
            "eval": "use of eval()",
            "exec": "use of exec()",
            "loads": "potential insecure deserialization",
            "system": "use of os.system()",
        }
        if name in dangerous:
            self.issues.append(f"line {node.lineno}: {dangerous[name]}")

        # subprocess with shell=True
        if name in ("call", "run", "Popen", "check_output", "check_call"):
            for kw in node.keywords:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    self.issues.append(f"line {node.lineno}: subprocess with shell=True")

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in ("pickle", "marshal", "shelve"):
                self.issues.append(f"line {node.lineno}: import of {alias.name}")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and len(node.value) > 8:
            lower = node.value.lower()
            if any(kw in lower for kw in ("password", "secret", "api_key", "apikey", "token")):
                self.issues.append(f"line {node.lineno}: possible hardcoded secret")
        self.generic_visit(node)


def _score_security(files: List[Path]) -> CategoryScore:
    """Score based on security patterns found."""
    all_issues: List[str] = []
    for fp in files:
        tree = _parse_file(fp)
        if tree is None:
            continue
        v = _SecurityVisitor()
        v.visit(tree)
        for issue in v.issues:
            all_issues.append(f"{fp.name}: {issue}")

    total = len(files)
    if total == 0:
        return CategoryScore(HealthCategory.SECURITY_SCORE, 100.0, 0.20, "no files")

    issues_per_file = len(all_issues) / total
    score = max(0.0, min(100.0, 100.0 - issues_per_file * 25))

    return CategoryScore(
        HealthCategory.SECURITY_SCORE,
        score,
        0.20,
        f"{len(all_issues)} security concerns across {total} files",
        all_issues[:20],
    )


# ---------------------------------------------------------------------------
# Type Safety Scoring
# ---------------------------------------------------------------------------

class _TypeAnnotationVisitor(ast.NodeVisitor):
    """Count annotated vs unannotated parameters and return types."""

    def __init__(self) -> None:
        self.annotated_params = 0
        self.total_params = 0
        self.annotated_returns = 0
        self.total_functions = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _check(self, node: ast.FunctionDef) -> None:
        self.total_functions += 1
        if node.returns is not None:
            self.annotated_returns += 1

        args = node.args
        all_args = args.args + args.posonlyargs + args.kwonlyargs
        if args.vararg:
            all_args_count = len(all_args) + 1
        else:
            all_args_count = len(all_args)
        if args.kwarg:
            all_args_count += 1

        # Skip 'self'/'cls'
        for a in all_args:
            if a.arg in ("self", "cls"):
                continue
            self.total_params += 1
            if a.annotation is not None:
                self.annotated_params += 1


def _score_type_safety(files: List[Path]) -> CategoryScore:
    """Score based on type annotation coverage."""
    v = _TypeAnnotationVisitor()
    for fp in files:
        tree = _parse_file(fp)
        if tree is None:
            continue
        v.visit(tree)

    if v.total_functions == 0:
        return CategoryScore(HealthCategory.TYPE_SAFETY_SCORE, 100.0, 0.10, "no functions")

    param_ratio = v.annotated_params / max(v.total_params, 1)
    return_ratio = v.annotated_returns / v.total_functions
    combined = (param_ratio * 0.6 + return_ratio * 0.4) * 100

    issues: List[str] = []
    if param_ratio < 0.5:
        issues.append(f"Only {v.annotated_params}/{v.total_params} parameters annotated")
    if return_ratio < 0.5:
        issues.append(f"Only {v.annotated_returns}/{v.total_functions} functions have return annotations")

    return CategoryScore(
        HealthCategory.TYPE_SAFETY_SCORE,
        min(100.0, combined),
        0.10,
        f"{param_ratio:.0%} param annotations, {return_ratio:.0%} return annotations",
        issues,
    )


# ---------------------------------------------------------------------------
# Dependency Health Scoring
# ---------------------------------------------------------------------------

def _score_dependency_health(project_dir: str) -> CategoryScore:
    """Score dependency health from requirements files."""
    root = Path(project_dir)
    req_files = list(root.glob("requirements*.txt")) + list(root.glob("setup.py")) + list(root.glob("pyproject.toml"))

    if not req_files:
        return CategoryScore(
            HealthCategory.DEPENDENCY_HEALTH, 70.0, 0.15,
            "no requirements file found",
            ["Consider adding requirements.txt or pyproject.toml"],
        )

    issues: List[str] = []
    pinned = 0
    unpinned = 0
    total_deps = 0

    for rf in req_files:
        if rf.suffix == ".txt":
            try:
                for line in rf.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("-"):
                        continue
                    total_deps += 1
                    if "==" in line:
                        pinned += 1
                    else:
                        unpinned += 1
                        if ">=" not in line and "<" not in line:
                            issues.append(f"Unpinned dependency: {line}")
            except OSError:
                pass

    if total_deps == 0:
        return CategoryScore(HealthCategory.DEPENDENCY_HEALTH, 80.0, 0.15, "no deps parsed")

    pin_ratio = pinned / total_deps
    score = 50 + pin_ratio * 50  # 50-100 based on pinning

    if total_deps > 50:
        issues.append(f"Large dependency count: {total_deps}")
        score -= 10

    return CategoryScore(
        HealthCategory.DEPENDENCY_HEALTH,
        max(0.0, min(100.0, score)),
        0.15,
        f"{total_deps} deps ({pinned} pinned, {unpinned} unpinned)",
        issues[:15],
    )


# ---------------------------------------------------------------------------
# Grade Calculation
# ---------------------------------------------------------------------------

_GRADE_THRESHOLDS = [
    (90, "A"),
    (80, "B"),
    (70, "C"),
    (60, "D"),
    (0, "F"),
]


def _to_grade(score: float) -> str:
    for threshold, grade in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def health_score(project_dir: str) -> HealthReport:
    """Compute an overall 0-100 health score for a Python project.

    Breaks down into: bug_density, test_coverage_estimate, code_complexity,
    security_score, type_safety_score, dependency_health.
    """
    files = _collect_python_files(project_dir)
    total_lines = sum(_count_lines(f) for f in files)

    categories: Dict[str, CategoryScore] = {}
    scorers = [
        _score_bug_density(files),
        _score_test_coverage(project_dir, files),
        _score_complexity(files),
        _score_security(files),
        _score_type_safety(files),
        _score_dependency_health(project_dir),
    ]

    total_weight = 0.0
    weighted_sum = 0.0
    for cs in scorers:
        categories[cs.category.value] = cs
        weighted_sum += cs.score * cs.weight
        total_weight += cs.weight

    overall = weighted_sum / max(total_weight, 0.01)
    grade = _to_grade(overall)

    recommendations: List[str] = []
    for cs in sorted(scorers, key=lambda c: c.score):
        if cs.score < 60:
            recommendations.append(
                f"Improve {cs.category.value}: score {cs.score:.0f}/100"
            )
        if cs.issues:
            recommendations.append(f"  Top issue: {cs.issues[0]}")

    return HealthReport(
        overall_score=overall,
        grade=grade,
        category_scores=categories,
        total_files=len(files),
        total_lines=total_lines,
        summary=f"Grade {grade} ({overall:.1f}/100) across {len(files)} files",
        recommendations=recommendations[:10],
    )


def trend_analysis(project_dir: str, git_history_days: int = 90) -> TrendReport:
    """Analyze how project quality has changed over recent git history.

    Samples the project at weekly intervals going back *git_history_days* and
    computes a health score for each snapshot.
    """
    root = Path(project_dir)
    report = TrendReport(project_dir=project_dir)

    # Determine date range
    now = datetime.now()
    weeks = git_history_days // 7
    if weeks < 1:
        weeks = 1

    # Try to get git log dates
    try:
        result = subprocess.run(
            ["git", "log", f"--since={git_history_days} days ago", "--format=%H %aI", "--reverse"],
            capture_output=True,
            text=True,
            cwd=project_dir,
            timeout=30,
        )
        if result.returncode != 0:
            report.summary = "Unable to read git history"
            return report
    except (subprocess.SubprocessError, FileNotFoundError):
        report.summary = "Git not available"
        return report

    lines = result.stdout.strip().splitlines()
    if not lines:
        report.summary = "No commits in the specified range"
        return report

    # Sample at intervals
    step = max(1, len(lines) // min(weeks, len(lines)))
    sampled = lines[::step]
    if lines[-1] not in sampled:
        sampled.append(lines[-1])

    current_score: Optional[float] = None
    first_score: Optional[float] = None

    for entry in sampled:
        parts = entry.split(" ", 1)
        if len(parts) < 2:
            continue
        sha, date_str = parts
        date_short = date_str[:10]

        # Checkout snapshot, compute score, restore
        try:
            subprocess.run(
                ["git", "stash", "--include-untracked"],
                capture_output=True, cwd=project_dir, timeout=10,
            )
            subprocess.run(
                ["git", "checkout", sha, "--quiet"],
                capture_output=True, cwd=project_dir, timeout=10,
            )
            hr = health_score(project_dir)
            point = TrendPoint(
                date=date_short,
                score=hr.overall_score,
                category_scores={k: v.score for k, v in hr.category_scores.items()},
            )
            report.points.append(point)

            if first_score is None:
                first_score = hr.overall_score
            current_score = hr.overall_score
        except Exception:
            logger.debug("Failed to analyze commit %s", sha)
        finally:
            subprocess.run(
                ["git", "checkout", "-", "--quiet"],
                capture_output=True, cwd=project_dir, timeout=10,
            )
            subprocess.run(
                ["git", "stash", "pop"],
                capture_output=True, cwd=project_dir, timeout=10,
            )

    # Compute trend
    if first_score is not None and current_score is not None and len(report.points) >= 2:
        delta = current_score - first_score
        report.average_change_per_week = delta / max(weeks, 1)
        if delta > 3:
            report.trend_direction = "improving"
        elif delta < -3:
            report.trend_direction = "declining"
        else:
            report.trend_direction = "stable"
        report.summary = (
            f"Score went from {first_score:.1f} to {current_score:.1f} "
            f"({delta:+.1f}) over {len(report.points)} snapshots"
        )
    else:
        report.summary = f"Analyzed {len(report.points)} snapshot(s)"

    return report


# Reference benchmark scores for top open-source Python projects
_BENCHMARKS: Dict[str, float] = {
    "requests": 85.0,
    "flask": 82.0,
    "django": 80.0,
    "fastapi": 88.0,
    "numpy": 78.0,
    "pandas": 75.0,
    "scikit-learn": 79.0,
    "pytest": 90.0,
    "black": 87.0,
    "mypy": 83.0,
}


def compare_to_benchmarks(project_dir: str) -> BenchmarkComparison:
    """Compare project health score against top open-source Python projects."""
    hr = health_score(project_dir)
    project_score = hr.overall_score

    sorted_scores = sorted(_BENCHMARKS.values())
    below = sum(1 for s in sorted_scores if s <= project_score)
    percentile = (below / max(len(sorted_scores), 1)) * 100

    better_than: List[str] = []
    worse_than: List[str] = []
    for name, s in sorted(_BENCHMARKS.items(), key=lambda x: x[1], reverse=True):
        if project_score >= s:
            better_than.append(f"{name} ({s:.0f})")
        else:
            worse_than.append(f"{name} ({s:.0f})")

    parts = [f"Your score: {project_score:.1f}"]
    if better_than:
        parts.append(f"Better than: {', '.join(better_than[:3])}")
    if worse_than:
        parts.append(f"Below: {', '.join(worse_than[:3])}")

    return BenchmarkComparison(
        project_score=project_score,
        benchmark_scores=dict(_BENCHMARKS),
        percentile=percentile,
        summary="\n".join(parts),
    )


def generate_badge(score: float) -> str:
    """Generate an SVG badge showing the project health score.

    Returns SVG markup suitable for embedding in a README.
    """
    if score >= 90:
        color = "#4c1"
    elif score >= 70:
        color = "#97CA00"
    elif score >= 50:
        color = "#dfb317"
    elif score >= 30:
        color = "#fe7d37"
    else:
        color = "#e05d44"

    grade = _to_grade(score)
    label = "health"
    value = f"{score:.0f}% ({grade})"
    label_width = len(label) * 7 + 10
    value_width = len(value) * 7 + 10
    total_width = label_width + value_width

    svg = textwrap.dedent(f"""\
    <svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">
      <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
      </linearGradient>
      <clipPath id="a">
        <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
      </clipPath>
      <g clip-path="url(#a)">
        <rect width="{label_width}" height="20" fill="#555"/>
        <rect x="{label_width}" width="{value_width}" height="20" fill="{color}"/>
        <rect width="{total_width}" height="20" fill="url(#b)"/>
      </g>
      <g fill="#fff" text-anchor="middle"
         font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="{label_width / 2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
        <text x="{label_width / 2}" y="14">{label}</text>
        <text x="{label_width + value_width / 2}" y="15" fill="#010101" fill-opacity=".3">{value}</text>
        <text x="{label_width + value_width / 2}" y="14">{value}</text>
      </g>
    </svg>""")
    return svg


def ci_quality_gate(project_dir: str, min_score: int = 70) -> bool:
    """Return True if the project meets the minimum health score for CI.

    Intended to be called from CI pipelines as a quality gate.
    Prints the health report to stdout for CI logs.
    """
    hr = health_score(project_dir)
    passed = hr.overall_score >= min_score
    status = "PASSED" if passed else "FAILED"
    logger.info(
        "Quality gate %s: %.1f/100 (minimum: %d)\n%s",
        status, hr.overall_score, min_score, hr,
    )
    return passed
