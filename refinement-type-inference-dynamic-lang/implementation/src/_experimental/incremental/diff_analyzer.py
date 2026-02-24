"""
Git Diff Analyzer for Incremental Analysis.

Analyzes only code affected by git changes:
- Parse git diff output
- Map hunks to affected functions
- Track file-level and function-level changes
- Support staged, unstaged, and branch diffs
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Local type stubs
# ---------------------------------------------------------------------------

@dataclass
class RefinementType:
    base: str = "Any"
    predicates: List[str] = field(default_factory=list)


@dataclass
class FunctionSummary:
    name: str = ""
    module: str = ""
    param_types: Dict[str, RefinementType] = field(default_factory=dict)
    return_type: RefinementType = field(default_factory=RefinementType)
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Diff data structures
# ---------------------------------------------------------------------------

@dataclass
class DiffHunk:
    """A single hunk from a unified diff."""
    start_line: int
    count: int
    content_lines: List[str] = field(default_factory=list)
    file_path: str = ""
    new_start_line: int = 0
    new_count: int = 0


@dataclass
class FileDiff:
    """Represents the diff for one file."""
    old_path: str = ""
    new_path: str = ""
    hunks: List[DiffHunk] = field(default_factory=list)
    is_new: bool = False
    is_deleted: bool = False
    is_renamed: bool = False
    is_binary: bool = False


# ---------------------------------------------------------------------------
# GitDiffParser
# ---------------------------------------------------------------------------

_DIFF_HEADER = re.compile(r"^diff --git a/(.*) b/(.*)$")
_HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)
_RENAME_FROM = re.compile(r"^rename from (.+)$")
_RENAME_TO = re.compile(r"^rename to (.+)$")
_NEW_FILE = re.compile(r"^new file mode")
_DELETED_FILE = re.compile(r"^deleted file mode")
_BINARY_FILE = re.compile(r"^Binary files")


class GitDiffParser:
    """Parse unified diff output into structured :class:`FileDiff` objects."""

    def parse(self, diff_text: str) -> List[FileDiff]:
        files: List[FileDiff] = []
        current: Optional[FileDiff] = None
        current_hunk: Optional[DiffHunk] = None

        for raw_line in diff_text.splitlines():
            header_m = _DIFF_HEADER.match(raw_line)
            if header_m:
                if current is not None:
                    if current_hunk is not None:
                        current.hunks.append(current_hunk)
                        current_hunk = None
                    files.append(current)
                current = FileDiff(
                    old_path=header_m.group(1),
                    new_path=header_m.group(2),
                )
                continue

            if current is None:
                continue

            if _NEW_FILE.match(raw_line):
                current.is_new = True
                continue
            if _DELETED_FILE.match(raw_line):
                current.is_deleted = True
                continue
            if _BINARY_FILE.match(raw_line):
                current.is_binary = True
                continue

            rename_from = _RENAME_FROM.match(raw_line)
            if rename_from:
                current.old_path = rename_from.group(1)
                current.is_renamed = True
                continue
            rename_to = _RENAME_TO.match(raw_line)
            if rename_to:
                current.new_path = rename_to.group(1)
                current.is_renamed = True
                continue

            hunk_m = _HUNK_HEADER.match(raw_line)
            if hunk_m:
                if current_hunk is not None:
                    current.hunks.append(current_hunk)
                current_hunk = DiffHunk(
                    start_line=int(hunk_m.group(1)),
                    count=int(hunk_m.group(2) or "1"),
                    new_start_line=int(hunk_m.group(3)),
                    new_count=int(hunk_m.group(4) or "1"),
                    file_path=current.new_path,
                )
                continue

            if current_hunk is not None:
                if raw_line.startswith(("+", "-", " ")):
                    current_hunk.content_lines.append(raw_line)

        if current is not None:
            if current_hunk is not None:
                current.hunks.append(current_hunk)
            files.append(current)
        return files


# ---------------------------------------------------------------------------
# FunctionMapper
# ---------------------------------------------------------------------------

@dataclass
class _FuncSpan:
    name: str
    start: int
    end: int
    qualified: str = ""


class FunctionMapper:
    """Map diff hunks to the Python functions they touch."""

    def _extract_spans(self, source: str, module: str = "") -> List[_FuncSpan]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        spans: List[_FuncSpan] = []
        self._walk_spans(tree, module, spans)
        return spans

    def _walk_spans(
        self, node: ast.AST, prefix: str, out: List[_FuncSpan]
    ) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                end = child.end_lineno if child.end_lineno else child.lineno
                out.append(_FuncSpan(
                    name=child.name, start=child.lineno, end=end,
                    qualified=qualified,
                ))
                self._walk_spans(child, qualified, out)
            elif isinstance(child, ast.ClassDef):
                qualified = f"{prefix}.{child.name}" if prefix else child.name
                self._walk_spans(child, qualified, out)

    def map_hunks_to_functions(
        self,
        hunks: List[DiffHunk],
        source: str,
        module: str = "",
    ) -> List[str]:
        """Return qualified names of functions touched by any hunk."""
        spans = self._extract_spans(source, module)
        touched: Set[str] = set()
        for hunk in hunks:
            hunk_start = hunk.new_start_line
            hunk_end = hunk_start + max(hunk.new_count - 1, 0)
            for span in spans:
                if span.start <= hunk_end and span.end >= hunk_start:
                    touched.add(span.qualified)
        return sorted(touched)

    def map_file_diff(
        self,
        diff: FileDiff,
        source: str,
        module: str = "",
    ) -> List[str]:
        if diff.is_new or diff.is_deleted:
            spans = self._extract_spans(source, module)
            return [s.qualified for s in spans]
        return self.map_hunks_to_functions(diff.hunks, source, module)


# ---------------------------------------------------------------------------
# DiffAnalyzer
# ---------------------------------------------------------------------------

class DiffAnalyzer:
    """High-level interface: run git diff variants and extract affected
    functions / modules."""

    def __init__(self, repo_dir: str = ".") -> None:
        self._repo = Path(repo_dir).resolve()
        self._parser = GitDiffParser()
        self._mapper = FunctionMapper()

    def _run_git(self, *args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True, text=True,
                cwd=str(self._repo), timeout=30,
            )
            return result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            return ""

    def analyze_staged(self) -> List[FileDiff]:
        raw = self._run_git("diff", "--cached", "-U3")
        return self._parser.parse(raw)

    def analyze_unstaged(self) -> List[FileDiff]:
        raw = self._run_git("diff", "-U3")
        return self._parser.parse(raw)

    def analyze_branch(self, branch: str) -> List[FileDiff]:
        raw = self._run_git("diff", f"{branch}...HEAD", "-U3")
        return self._parser.parse(raw)

    def analyze_commits(self, from_sha: str, to_sha: str) -> List[FileDiff]:
        raw = self._run_git("diff", from_sha, to_sha, "-U3")
        return self._parser.parse(raw)

    def _read_file(self, path: str) -> str:
        full = self._repo / path
        try:
            return full.read_text(errors="replace")
        except OSError:
            return ""

    def get_affected_functions(
        self, diffs: List[FileDiff]
    ) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        for diff in diffs:
            if diff.is_binary:
                continue
            path = diff.new_path if not diff.is_deleted else diff.old_path
            if not path.endswith(".py"):
                continue
            source = self._read_file(path)
            module = path.replace("/", ".").removesuffix(".py")
            funcs = self._mapper.map_file_diff(diff, source, module)
            if funcs:
                result[path] = funcs
        return result

    def get_affected_modules(self, diffs: List[FileDiff]) -> Set[str]:
        modules: Set[str] = set()
        for diff in diffs:
            path = diff.new_path if not diff.is_deleted else diff.old_path
            if path.endswith(".py"):
                modules.add(path.replace("/", ".").removesuffix(".py"))
        return modules


# ---------------------------------------------------------------------------
# ChangeImpactAnalyzer
# ---------------------------------------------------------------------------

class ChangeImpactAnalyzer:
    """Given a set of affected functions, compute the wider impact."""

    def __init__(self) -> None:
        self._forward: Dict[str, Set[str]] = defaultdict(set)
        self._reverse: Dict[str, Set[str]] = defaultdict(set)
        self._type_cache: Dict[str, str] = {}

    def add_dependency(self, caller: str, callee: str) -> None:
        self._forward[caller].add(callee)
        self._reverse[callee].add(caller)

    def set_return_type(self, func: str, rtype: str) -> None:
        self._type_cache[func] = rtype

    def build_from_source(self, source: str, module: str = "") -> None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return
        func_names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{module}.{node.name}" if module else node.name
                func_names.add(qualified)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            caller = f"{module}.{node.name}" if module else node.name
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    name: Optional[str] = None
                    if isinstance(child.func, ast.Name):
                        name = child.func.id
                    elif isinstance(child.func, ast.Attribute):
                        name = child.func.attr
                    if name:
                        q = f"{module}.{name}" if module else name
                        if q in func_names and q != caller:
                            self.add_dependency(caller, q)

    def direct_impact(self, changed: Set[str]) -> Set[str]:
        return set(changed)

    def indirect_impact(self, changed: Set[str]) -> Set[str]:
        visited: Set[str] = set()
        queue: Deque[str] = deque(changed)
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            for caller in self._reverse.get(cur, set()):
                if caller not in visited:
                    queue.append(caller)
        return visited - changed

    def type_impact(self, changed: Set[str]) -> Set[str]:
        """Functions whose inferred type might change because a dependency's
        type changed."""
        impacted: Set[str] = set()
        for func in changed:
            for caller in self._reverse.get(func, set()):
                impacted.add(caller)
                # transitive: callers of callers whose return depends on caller
                for grandcaller in self._reverse.get(caller, set()):
                    impacted.add(grandcaller)
        return impacted - changed

    def prioritize(self, changed: Set[str]) -> List[Tuple[str, float]]:
        """Rank functions by likelihood of harbouring new bugs.

        Heuristic score = direct(3) + indirect(1) + type_impact(2),
        normalised so the top function has score 1.0.
        """
        scores: Dict[str, float] = defaultdict(float)
        for f in self.direct_impact(changed):
            scores[f] += 3.0
        for f in self.indirect_impact(changed):
            scores[f] += 1.0
        for f in self.type_impact(changed):
            scores[f] += 2.0
        if not scores:
            return []
        max_score = max(scores.values())
        if max_score == 0:
            max_score = 1.0
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(name, score / max_score) for name, score in ranked]

    def full_impact(self, changed: Set[str]) -> Dict[str, Set[str]]:
        return {
            "direct": self.direct_impact(changed),
            "indirect": self.indirect_impact(changed),
            "type": self.type_impact(changed),
        }


# ---------------------------------------------------------------------------
# DiffReporter
# ---------------------------------------------------------------------------

class DiffReporter:
    """Format diff analysis results for human consumption."""

    def __init__(self) -> None:
        self._total_functions: int = 0

    def format_affected_functions(
        self,
        affected: Dict[str, List[str]],
    ) -> str:
        lines: List[str] = ["Affected functions by file:", ""]
        total = 0
        for path in sorted(affected):
            funcs = affected[path]
            total += len(funcs)
            lines.append(f"  {path}:")
            for f in funcs:
                lines.append(f"    - {f}")
        lines.append("")
        lines.append(f"Total affected functions: {total}")
        self._total_functions = total
        return "\n".join(lines)

    def format_impact(
        self,
        impact: Dict[str, Set[str]],
    ) -> str:
        lines: List[str] = ["Impact analysis:", ""]
        for category in ("direct", "indirect", "type"):
            funcs = impact.get(category, set())
            lines.append(f"  {category.capitalize()} impact ({len(funcs)}):")
            for f in sorted(funcs):
                lines.append(f"    - {f}")
        return "\n".join(lines)

    def format_scope_estimate(
        self,
        total_project_functions: int,
        affected_count: int,
    ) -> str:
        if total_project_functions == 0:
            pct = 0.0
        else:
            pct = affected_count / total_project_functions * 100
        return (
            f"Analysis scope: {affected_count}/{total_project_functions} "
            f"functions ({pct:.1f}%)"
        )

    def format_time_savings(
        self,
        full_time: float,
        incremental_time: float,
    ) -> str:
        if full_time <= 0:
            return "No baseline available for comparison."
        saved = full_time - incremental_time
        pct = saved / full_time * 100 if full_time > 0 else 0
        return (
            f"Time savings: {saved:.2f}s saved "
            f"({pct:.1f}% faster than full analysis)"
        )

    def format_priority_list(
        self, ranked: List[Tuple[str, float]]
    ) -> str:
        lines = ["Priority ranking (likelihood of new bugs):", ""]
        for i, (name, score) in enumerate(ranked, 1):
            bar = "█" * int(score * 20)
            lines.append(f"  {i:3d}. {name:40s} {score:.2f} {bar}")
        return "\n".join(lines)

    def full_report(
        self,
        affected: Dict[str, List[str]],
        impact: Dict[str, Set[str]],
        ranked: List[Tuple[str, float]],
        total_project_functions: int = 0,
        full_time: float = 0.0,
        incremental_time: float = 0.0,
    ) -> str:
        affected_count = sum(len(v) for v in affected.values())
        sections = [
            self.format_affected_functions(affected),
            "",
            self.format_impact(impact),
            "",
            self.format_priority_list(ranked),
            "",
            self.format_scope_estimate(
                total_project_functions, affected_count
            ),
            self.format_time_savings(full_time, incremental_time),
        ]
        return "\n".join(sections)
