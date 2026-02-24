"""
Incremental Analysis Engine.

Re-analyzes only changed functions and their transitive dependents.
Integrates with the dependency tracker and cache to minimize work.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
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
    TypeVar,
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
# Change representation
# ---------------------------------------------------------------------------

class ChangeKind(Enum):
    """Kind of change detected for a function."""
    ADDED = auto()
    MODIFIED = auto()
    DELETED = auto()
    RENAMED = auto()


@dataclass
class FunctionChange:
    """Represents a change to a single function."""
    function_name: str
    change_kind: ChangeKind
    old_source: str = ""
    new_source: str = ""
    old_ast_hash: str = ""
    new_ast_hash: str = ""
    module: str = ""
    old_name: str = ""  # for renames


@dataclass
class AnalysisResult:
    """Result of an incremental analysis run."""
    analyzed_functions: Dict[str, FunctionSummary] = field(default_factory=dict)
    cached_functions: Dict[str, FunctionSummary] = field(default_factory=dict)
    invalidated: Set[str] = field(default_factory=set)
    errors: Dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    total_functions: int = 0
    reanalyzed_count: int = 0


# ---------------------------------------------------------------------------
# AST hashing
# ---------------------------------------------------------------------------

def _hash_ast_node(source: str) -> str:
    """Produce a stable hash of the AST derived from *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return hashlib.sha256(source.encode()).hexdigest()
    dumped = ast.dump(tree, annotate_fields=False)
    return hashlib.sha256(dumped.encode()).hexdigest()


def _extract_functions_from_source(source: str) -> Dict[str, str]:
    """Return mapping of function-name -> source text for every top-level
    function/method in *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    lines = source.splitlines(keepends=True)
    funcs: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno if node.end_lineno else start + 1
            funcs[node.name] = "".join(lines[start:end])
    return funcs


# ---------------------------------------------------------------------------
# ChangeDetector
# ---------------------------------------------------------------------------

class ChangeDetector:
    """Compare two versions of a module and detect per-function changes."""

    def detect_function_changes(
        self, old_source: str, new_source: str, module: str = ""
    ) -> List[FunctionChange]:
        old_funcs = _extract_functions_from_source(old_source)
        new_funcs = _extract_functions_from_source(new_source)
        changes: List[FunctionChange] = []

        all_names = set(old_funcs) | set(new_funcs)
        for name in sorted(all_names):
            if name not in old_funcs:
                changes.append(FunctionChange(
                    function_name=name, change_kind=ChangeKind.ADDED,
                    new_source=new_funcs[name],
                    new_ast_hash=_hash_ast_node(new_funcs[name]),
                    module=module,
                ))
            elif name not in new_funcs:
                changes.append(FunctionChange(
                    function_name=name, change_kind=ChangeKind.DELETED,
                    old_source=old_funcs[name],
                    old_ast_hash=_hash_ast_node(old_funcs[name]),
                    module=module,
                ))
            else:
                old_hash = _hash_ast_node(old_funcs[name])
                new_hash = _hash_ast_node(new_funcs[name])
                if old_hash != new_hash:
                    changes.append(FunctionChange(
                        function_name=name, change_kind=ChangeKind.MODIFIED,
                        old_source=old_funcs[name], new_source=new_funcs[name],
                        old_ast_hash=old_hash, new_ast_hash=new_hash,
                        module=module,
                    ))
        return changes

    def detect_class_changes(
        self, old_source: str, new_source: str
    ) -> Dict[str, ChangeKind]:
        """Return mapping class-name -> ChangeKind for changed classes."""
        def _class_hashes(src: str) -> Dict[str, str]:
            try:
                tree = ast.parse(src)
            except SyntaxError:
                return {}
            lines = src.splitlines(keepends=True)
            out: Dict[str, str] = {}
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    start = node.lineno - 1
                    end = node.end_lineno if node.end_lineno else start + 1
                    body = "".join(lines[start:end])
                    out[node.name] = _hash_ast_node(body)
            return out

        old_h = _class_hashes(old_source)
        new_h = _class_hashes(new_source)
        result: Dict[str, ChangeKind] = {}
        for name in sorted(set(old_h) | set(new_h)):
            if name not in old_h:
                result[name] = ChangeKind.ADDED
            elif name not in new_h:
                result[name] = ChangeKind.DELETED
            elif old_h[name] != new_h[name]:
                result[name] = ChangeKind.MODIFIED
        return result

    def detect_global_changes(
        self, old_source: str, new_source: str
    ) -> Dict[str, ChangeKind]:
        """Detect changes to module-level assignments."""
        def _globals(src: str) -> Dict[str, str]:
            try:
                tree = ast.parse(src)
            except SyntaxError:
                return {}
            out: Dict[str, str] = {}
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            out[target.id] = ast.dump(node.value, annotate_fields=False)
            return out

        old_g = _globals(old_source)
        new_g = _globals(new_source)
        result: Dict[str, ChangeKind] = {}
        for name in sorted(set(old_g) | set(new_g)):
            if name not in old_g:
                result[name] = ChangeKind.ADDED
            elif name not in new_g:
                result[name] = ChangeKind.DELETED
            elif old_g[name] != new_g[name]:
                result[name] = ChangeKind.MODIFIED
        return result

    def detect_renames(
        self, old_source: str, new_source: str, module: str = ""
    ) -> List[FunctionChange]:
        """Heuristic rename detection: deleted+added with same AST hash."""
        changes = self.detect_function_changes(old_source, new_source, module)
        added = {c.new_ast_hash: c for c in changes if c.change_kind == ChangeKind.ADDED}
        deleted = {c.old_ast_hash: c for c in changes if c.change_kind == ChangeKind.DELETED}
        renames: List[FunctionChange] = []
        for h, del_c in deleted.items():
            if h in added:
                add_c = added[h]
                renames.append(FunctionChange(
                    function_name=add_c.function_name,
                    change_kind=ChangeKind.RENAMED,
                    old_source=del_c.old_source, new_source=add_c.new_source,
                    old_ast_hash=h, new_ast_hash=h,
                    module=module, old_name=del_c.function_name,
                ))
        return renames


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------

class DependencyGraph:
    """Track call-level dependencies between functions and compute transitive
    closures / SCCs."""

    def __init__(self) -> None:
        self._forward: Dict[str, Set[str]] = defaultdict(set)   # caller -> callees
        self._reverse: Dict[str, Set[str]] = defaultdict(set)   # callee -> callers

    def add_edge(self, caller: str, callee: str) -> None:
        self._forward[caller].add(callee)
        self._reverse[callee].add(caller)

    def remove_node(self, name: str) -> None:
        for callee in list(self._forward.get(name, [])):
            self._reverse[callee].discard(name)
        for caller in list(self._reverse.get(name, [])):
            self._forward[caller].discard(name)
        self._forward.pop(name, None)
        self._reverse.pop(name, None)

    def callers_of(self, name: str) -> Set[str]:
        return set(self._reverse.get(name, set()))

    def callees_of(self, name: str) -> Set[str]:
        return set(self._forward.get(name, set()))

    def all_nodes(self) -> Set[str]:
        return set(self._forward) | set(self._reverse)

    # -- transitive closure ---------------------------------------------------

    def transitive_callers(self, names: Set[str]) -> Set[str]:
        """All functions that transitively depend on any function in *names*."""
        visited: Set[str] = set()
        queue: Deque[str] = deque(names)
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            for caller in self._reverse.get(cur, set()):
                if caller not in visited:
                    queue.append(caller)
        return visited

    def transitive_callees(self, names: Set[str]) -> Set[str]:
        visited: Set[str] = set()
        queue: Deque[str] = deque(names)
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            for callee in self._forward.get(cur, set()):
                if callee not in visited:
                    queue.append(callee)
        return visited

    # -- strongly connected components (Tarjan) --------------------------------

    def strongly_connected_components(self) -> List[List[str]]:
        index_counter = [0]
        stack: List[str] = []
        on_stack: Set[str] = set()
        indices: Dict[str, int] = {}
        lowlinks: Dict[str, int] = {}
        result: List[List[str]] = []

        def strongconnect(v: str) -> None:
            indices[v] = lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)
            for w in self._forward.get(v, set()):
                if w not in indices:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif w in on_stack:
                    lowlinks[v] = min(lowlinks[v], indices[w])
            if lowlinks[v] == indices[v]:
                component: List[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == v:
                        break
                result.append(component)

        for node in sorted(self.all_nodes()):
            if node not in indices:
                strongconnect(node)
        return result

    def build_from_source(self, source: str, module: str = "") -> None:
        """Populate graph by analysing calls in *source*."""
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
                    callee_name: Optional[str] = None
                    if isinstance(child.func, ast.Name):
                        callee_name = child.func.id
                    elif isinstance(child.func, ast.Attribute):
                        callee_name = child.func.attr
                    if callee_name:
                        qualified_callee = f"{module}.{callee_name}" if module else callee_name
                        if qualified_callee in func_names and qualified_callee != caller:
                            self.add_edge(caller, qualified_callee)


# ---------------------------------------------------------------------------
# IncrementalAnalyzer
# ---------------------------------------------------------------------------

class IncrementalAnalyzer:
    """Core engine: given a set of function changes, re-analyze only the
    affected subset and merge with cached results."""

    def __init__(
        self,
        cache_dir: str = ".cache/incremental",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._config = config or {}
        self._graph = DependencyGraph()
        self._summaries: Dict[str, FunctionSummary] = {}

    # -- cache persistence -----------------------------------------------------

    def _cache_path(self, func_name: str) -> Path:
        safe = func_name.replace(".", "_").replace("/", "_")
        return self._cache_dir / f"{safe}.json"

    def _load_cached(self, func_name: str) -> Optional[FunctionSummary]:
        p = self._cache_path(func_name)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            return FunctionSummary(
                name=data.get("name", func_name),
                module=data.get("module", ""),
                return_type=RefinementType(base=data.get("return_base", "Any")),
                timestamp=data.get("timestamp", 0.0),
            )
        except (json.JSONDecodeError, OSError):
            return None

    def _save_cached(self, func_name: str, summary: FunctionSummary) -> None:
        p = self._cache_path(func_name)
        data = {
            "name": summary.name,
            "module": summary.module,
            "return_base": summary.return_type.base,
            "timestamp": summary.timestamp,
        }
        try:
            p.write_text(json.dumps(data))
        except OSError:
            pass

    def _invalidate_cached(self, func_name: str) -> None:
        p = self._cache_path(func_name)
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    # -- analysis --------------------------------------------------------------

    def _analyze_function(self, source: str, name: str, module: str) -> FunctionSummary:
        """Stub analysis: parse the function and produce a summary."""
        ret = RefinementType(base="Any")
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Return) and node.value is not None:
                    if isinstance(node.value, ast.Constant):
                        ret = RefinementType(base=type(node.value.value).__name__)
                    elif isinstance(node.value, ast.Name):
                        ret = RefinementType(base=node.value.id)
        except SyntaxError:
            pass
        params: Dict[str, RefinementType] = {}
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for arg in node.args.args:
                        params[arg.arg] = RefinementType(base="Any")
                    break
        except SyntaxError:
            pass
        return FunctionSummary(
            name=name, module=module,
            param_types=params, return_type=ret,
            timestamp=time.time(),
        )

    def compute_affected_set(self, changes: List[FunctionChange]) -> Set[str]:
        """Return the full set of functions that need re-analysis."""
        roots: Set[str] = set()
        for ch in changes:
            qualified = f"{ch.module}.{ch.function_name}" if ch.module else ch.function_name
            roots.add(qualified)
        return self._graph.transitive_callers(roots)

    def analyze_changes(self, changes: List[FunctionChange]) -> AnalysisResult:
        t0 = time.time()
        affected = self.compute_affected_set(changes)
        result = AnalysisResult(total_functions=len(self._summaries) + len(affected))

        change_map: Dict[str, FunctionChange] = {}
        for ch in changes:
            key = f"{ch.module}.{ch.function_name}" if ch.module else ch.function_name
            change_map[key] = ch

        # invalidate stale entries
        for name in affected:
            self._invalidate_cached(name)
            result.invalidated.add(name)

        # re-analyze affected
        for name in sorted(affected):
            ch = change_map.get(name)
            if ch and ch.change_kind == ChangeKind.DELETED:
                self._graph.remove_node(name)
                self._summaries.pop(name, None)
                continue
            source = ch.new_source if ch else ""
            if not source:
                cached = self._load_cached(name)
                if cached:
                    result.cached_functions[name] = cached
                    continue
            try:
                summary = self._analyze_function(source, name, ch.module if ch else "")
                result.analyzed_functions[name] = summary
                self._summaries[name] = summary
                self._save_cached(name, summary)
                result.reanalyzed_count += 1
            except Exception as exc:
                result.errors[name] = str(exc)

        # cascade: if return type changed, mark callers for re-analysis
        cascaded: Set[str] = set()
        for name, summary in result.analyzed_functions.items():
            old = self._summaries.get(name)
            if old and old.return_type.base != summary.return_type.base:
                callers = self._graph.callers_of(name)
                cascaded.update(callers - affected)
        for name in cascaded:
            self._invalidate_cached(name)
            result.invalidated.add(name)

        # fill unaffected from cache
        for name, summary in self._summaries.items():
            if name not in affected and name not in result.cached_functions:
                result.cached_functions[name] = summary

        result.elapsed_seconds = time.time() - t0
        return result


# ---------------------------------------------------------------------------
# AnalysisMerger
# ---------------------------------------------------------------------------

class AnalysisMerger:
    """Merge incremental results with cached results, resolving conflicts."""

    def merge(
        self,
        incremental: Dict[str, FunctionSummary],
        cached: Dict[str, FunctionSummary],
    ) -> Dict[str, FunctionSummary]:
        merged: Dict[str, FunctionSummary] = dict(cached)
        for name, summary in incremental.items():
            merged[name] = summary
        return merged

    def detect_conflicts(
        self,
        incremental: Dict[str, FunctionSummary],
        cached: Dict[str, FunctionSummary],
    ) -> List[Tuple[str, FunctionSummary, FunctionSummary]]:
        """Return list of (name, incremental_summary, cached_summary) where
        the two disagree on return type."""
        conflicts: List[Tuple[str, FunctionSummary, FunctionSummary]] = []
        for name in incremental:
            if name in cached:
                inc = incremental[name]
                cac = cached[name]
                if inc.return_type.base != cac.return_type.base:
                    conflicts.append((name, inc, cac))
        return conflicts

    def validate_boundaries(
        self,
        merged: Dict[str, FunctionSummary],
        graph: DependencyGraph,
    ) -> List[str]:
        """Check that at every call edge caller-param types are compatible
        with callee-return types. Return list of warnings."""
        warnings: List[str] = []
        for caller in merged:
            for callee in graph.callees_of(caller):
                if callee not in merged:
                    warnings.append(
                        f"Missing summary for callee {callee} called by {caller}"
                    )
        return warnings

    def resolve_conflicts(
        self,
        conflicts: List[Tuple[str, FunctionSummary, FunctionSummary]],
        strategy: str = "prefer_new",
    ) -> Dict[str, FunctionSummary]:
        resolved: Dict[str, FunctionSummary] = {}
        for name, inc, cac in conflicts:
            if strategy == "prefer_new":
                resolved[name] = inc
            elif strategy == "prefer_cached":
                resolved[name] = cac
            else:
                resolved[name] = inc if inc.timestamp >= cac.timestamp else cac
        return resolved


# ---------------------------------------------------------------------------
# IncrementalSession
# ---------------------------------------------------------------------------

class IncrementalSession:
    """Manages an end-to-end incremental analysis session.

    Workflow: detect changes -> compute affected set -> run analysis ->
    merge with cache -> persist results.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/incremental",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._detector = ChangeDetector()
        self._analyzer = IncrementalAnalyzer(cache_dir=cache_dir, config=config)
        self._merger = AnalysisMerger()
        self._graph = self._analyzer._graph
        self._results: Optional[AnalysisResult] = None

    def load_source(self, source: str, module: str = "") -> None:
        """Register a module's source so the dependency graph can be built."""
        self._graph.build_from_source(source, module)

    def run(
        self,
        old_source: str,
        new_source: str,
        module: str = "",
    ) -> AnalysisResult:
        """Full incremental session: detect changes, analyze, merge."""
        self._graph.build_from_source(new_source, module)
        changes = self._detector.detect_function_changes(old_source, new_source, module)
        renames = self._detector.detect_renames(old_source, new_source, module)
        all_changes = changes + renames
        result = self._analyzer.analyze_changes(all_changes)

        # merge
        merged = self._merger.merge(result.analyzed_functions, result.cached_functions)
        conflicts = self._merger.detect_conflicts(
            result.analyzed_functions, result.cached_functions
        )
        if conflicts:
            resolved = self._merger.resolve_conflicts(conflicts)
            merged.update(resolved)

        warnings = self._merger.validate_boundaries(merged, self._graph)
        for w in warnings:
            result.errors.setdefault("_warnings", "")
            result.errors["_warnings"] += w + "\n"

        self._results = result
        return result

    def run_multi(
        self,
        modules: List[Tuple[str, str, str]],
    ) -> AnalysisResult:
        """Run incremental analysis over multiple modules.

        *modules* is a list of (module_name, old_source, new_source).
        """
        combined = AnalysisResult()
        t0 = time.time()
        for module_name, old_src, new_src in modules:
            r = self.run(old_src, new_src, module_name)
            combined.analyzed_functions.update(r.analyzed_functions)
            combined.cached_functions.update(r.cached_functions)
            combined.invalidated.update(r.invalidated)
            combined.errors.update(r.errors)
            combined.reanalyzed_count += r.reanalyzed_count
        combined.total_functions = (
            len(combined.analyzed_functions) + len(combined.cached_functions)
        )
        combined.elapsed_seconds = time.time() - t0
        return combined

    @property
    def last_result(self) -> Optional[AnalysisResult]:
        return self._results

    def summary_report(self) -> str:
        """Human-readable summary of the last run."""
        if self._results is None:
            return "No analysis has been run yet."
        r = self._results
        lines = [
            f"Incremental analysis complete in {r.elapsed_seconds:.3f}s",
            f"  Total functions : {r.total_functions}",
            f"  Re-analyzed     : {r.reanalyzed_count}",
            f"  From cache      : {len(r.cached_functions)}",
            f"  Invalidated     : {len(r.invalidated)}",
            f"  Errors          : {len(r.errors)}",
        ]
        return "\n".join(lines)
