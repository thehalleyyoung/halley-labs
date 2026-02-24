"""Interprocedural analysis — cross-function and cross-module reasoning.

Builds call graphs, performs interprocedural null analysis, taint tracking
across module boundaries, effect inference (IO / mutation / exceptions),
purity analysis, and escape analysis.
"""
from __future__ import annotations

import ast
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class CallSite:
    caller: str
    callee: str
    file: str
    line: int
    column: int
    is_method: bool = False
    is_dynamic: bool = False

    def __str__(self) -> str:
        return f"{self.caller} -> {self.callee} ({self.file}:{self.line})"


@dataclass
class CallGraph:
    nodes: Set[str] = field(default_factory=set)
    edges: List[CallSite] = field(default_factory=list)
    callers: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    callees: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    entry_points: Set[str] = field(default_factory=set)
    leaf_functions: Set[str] = field(default_factory=set)
    recursive_functions: Set[str] = field(default_factory=set)
    strongly_connected: List[Set[str]] = field(default_factory=list)

    def add_edge(self, site: CallSite) -> None:
        self.nodes.add(site.caller)
        self.nodes.add(site.callee)
        self.edges.append(site)
        self.callers[site.callee].add(site.caller)
        self.callees[site.caller].add(site.callee)

    def transitive_callees(self, fn: str) -> Set[str]:
        visited: Set[str] = set()
        stack = [fn]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for callee in self.callees.get(cur, set()):
                if callee not in visited:
                    stack.append(callee)
        visited.discard(fn)
        return visited

    def transitive_callers(self, fn: str) -> Set[str]:
        visited: Set[str] = set()
        stack = [fn]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for caller in self.callers.get(cur, set()):
                if caller not in visited:
                    stack.append(caller)
        visited.discard(fn)
        return visited


class BugKind(Enum):
    NULL_DEREFERENCE = "null_dereference"
    UNINITIALIZED = "uninitialized"
    TYPE_MISMATCH = "type_mismatch"
    NONE_RETURN_USED = "none_return_used"
    UNCHECKED_OPTIONAL = "unchecked_optional"


@dataclass
class Bug:
    kind: BugKind
    message: str
    file: str
    line: int
    column: int
    call_chain: List[str] = field(default_factory=list)
    severity: str = "error"
    confidence: float = 0.85

    def __str__(self) -> str:
        chain = " -> ".join(self.call_chain) if self.call_chain else ""
        prefix = f"[{chain}] " if chain else ""
        return f"{self.file}:{self.line}:{self.column} {prefix}[{self.kind.value}] {self.message}"


@dataclass
class TaintSource:
    name: str
    patterns: List[str] = field(default_factory=list)


@dataclass
class TaintSink:
    name: str
    patterns: List[str] = field(default_factory=list)
    cwe_id: Optional[str] = None


@dataclass
class TaintPath:
    source: str
    sink: str
    path: List[str] = field(default_factory=list)
    file: str = ""
    source_line: int = 0
    sink_line: int = 0
    cwe_id: Optional[str] = None

    def __str__(self) -> str:
        chain = " -> ".join(self.path)
        return f"Taint: {self.source} ~> {self.sink} via {chain}"


class EffectKind(Enum):
    IO_READ = "io_read"
    IO_WRITE = "io_write"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    MUTATION = "mutation"
    EXCEPTION = "exception"
    PRINT = "print"
    SUBPROCESS = "subprocess"
    DATABASE = "database"
    RANDOM = "random"
    TIME = "time"
    GLOBAL_WRITE = "global_write"
    ENV_READ = "env_read"


@dataclass
class Effect:
    kind: EffectKind
    description: str
    line: int = 0
    target: str = ""


@dataclass
class Effects:
    function: str
    effects: List[Effect] = field(default_factory=list)
    is_pure: bool = True
    transitive_effects: List[Effect] = field(default_factory=list)

    @property
    def has_io(self) -> bool:
        return any(e.kind in (EffectKind.IO_READ, EffectKind.IO_WRITE,
                              EffectKind.NETWORK, EffectKind.FILESYSTEM,
                              EffectKind.PRINT, EffectKind.SUBPROCESS,
                              EffectKind.DATABASE)
                   for e in self.all_effects)

    @property
    def has_mutation(self) -> bool:
        return any(e.kind == EffectKind.MUTATION for e in self.all_effects)

    @property
    def all_effects(self) -> List[Effect]:
        return self.effects + self.transitive_effects


class EscapeKind(Enum):
    NO_ESCAPE = "no_escape"
    ARG_ESCAPE = "arg_escape"
    RETURN_ESCAPE = "return_escape"
    GLOBAL_ESCAPE = "global_escape"
    HEAP_ESCAPE = "heap_escape"


@dataclass
class EscapeInfo:
    variable: str
    kind: EscapeKind
    escape_point: Optional[str] = None
    line: int = 0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts: List[str] = []
        cur = node.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _iter_python_files(directory: str) -> List[str]:
    """Yield all .py files under *directory*."""
    result: List[str] = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in (
            "__pycache__", ".git", ".venv", "venv", "node_modules",
            ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
        )]
        for f in files:
            if f.endswith(".py"):
                result.append(os.path.join(root, f))
    return result


def _parse_file(path: str) -> Optional[ast.Module]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return ast.parse(fh.read(), filename=path)
    except (SyntaxError, OSError):
        return None


def _qualified_name(module_path: str, project_dir: str, fn_name: str) -> str:
    rel = os.path.relpath(module_path, project_dir)
    mod = rel.replace(os.sep, ".").removesuffix(".py")
    return f"{mod}.{fn_name}"


# ── Call graph construction ──────────────────────────────────────────────────

class _CallGraphBuilder(ast.NodeVisitor):
    """Build a call graph from a single module."""

    def __init__(self, file: str, project_dir: str):
        self.file = file
        self.project_dir = project_dir
        self.sites: List[CallSite] = []
        self.defined_functions: Set[str] = set()
        self._current_function: Optional[str] = None
        self._class_name: Optional[str] = None
        self._imports: Dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            self._imports[name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            self._imports[name] = f"{mod}.{alias.name}"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        old_class = self._class_name
        self._class_name = node.name
        self.generic_visit(node)
        self._class_name = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prefix = f"{self._class_name}." if self._class_name else ""
        qname = _qualified_name(self.file, self.project_dir, f"{prefix}{node.name}")
        self.defined_functions.add(qname)
        old = self._current_function
        self._current_function = qname
        self.generic_visit(node)
        self._current_function = old

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if self._current_function is None:
            self.generic_visit(node)
            return
        call_name = _get_call_name(node)
        if not call_name:
            self.generic_visit(node)
            return

        is_method = isinstance(node.func, ast.Attribute)
        resolved = self._resolve_callee(call_name)
        self.sites.append(CallSite(
            caller=self._current_function,
            callee=resolved,
            file=self.file,
            line=node.lineno,
            column=node.col_offset,
            is_method=is_method,
            is_dynamic=call_name != resolved and resolved not in self.defined_functions,
        ))
        self.generic_visit(node)

    def _resolve_callee(self, name: str) -> str:
        if name in self._imports:
            return self._imports[name]
        parts = name.split(".")
        if parts[0] in self._imports:
            rest = ".".join(parts[1:])
            return f"{self._imports[parts[0]]}.{rest}"
        return _qualified_name(self.file, self.project_dir, name)


def analyze_call_graph(project_dir: str) -> CallGraph:
    """Build a complete call graph for all Python files under *project_dir*."""
    graph = CallGraph()
    all_defined: Set[str] = set()

    files = _iter_python_files(project_dir)
    builders: List[_CallGraphBuilder] = []

    for fpath in files:
        tree = _parse_file(fpath)
        if tree is None:
            continue
        builder = _CallGraphBuilder(fpath, project_dir)
        builder.visit(tree)
        builders.append(builder)
        all_defined.update(builder.defined_functions)

    for builder in builders:
        for site in builder.sites:
            graph.add_edge(site)

    graph.entry_points = graph.nodes - {callee for callees in graph.callers.values() for callee in callees if callees}
    actually_called = set()
    for callees in graph.callees.values():
        actually_called.update(callees)
    graph.entry_points = graph.nodes - actually_called

    graph.leaf_functions = {n for n in graph.nodes if not graph.callees.get(n)}
    graph.recursive_functions = {n for n in graph.nodes if n in graph.transitive_callees(n)}
    graph.strongly_connected = _tarjan_scc(graph)

    return graph


def _tarjan_scc(graph: CallGraph) -> List[Set[str]]:
    """Tarjan's algorithm for strongly connected components."""
    index_counter = [0]
    stack: List[str] = []
    lowlink: Dict[str, int] = {}
    index: Dict[str, int] = {}
    on_stack: Set[str] = set()
    result: List[Set[str]] = []

    def strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.callees.get(v, set()):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            component: Set[str] = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.add(w)
                if w == v:
                    break
            if len(component) > 1:
                result.append(component)

    for v in graph.nodes:
        if v not in index:
            strongconnect(v)

    return result


# ── Interprocedural null analysis ────────────────────────────────────────────

class _NullReturnChecker(ast.NodeVisitor):
    """Check if a function can return None."""

    def __init__(self) -> None:
        self.can_return_none = False
        self.has_return = False
        self._depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._depth > 0:
            return
        self._depth += 1
        self.generic_visit(node)
        if not self.has_return:
            self.can_return_none = True
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Return(self, node: ast.Return) -> None:
        self.has_return = True
        if node.value is None:
            self.can_return_none = True
        elif isinstance(node.value, ast.Constant) and node.value.value is None:
            self.can_return_none = True
        elif isinstance(node.value, ast.Call):
            name = _get_call_name(node.value)
            if name in _RETURNS_OPTIONAL:
                self.can_return_none = True


_RETURNS_OPTIONAL: Set[str] = {
    "dict.get", "os.environ.get", "os.getenv",
    "re.match", "re.search", "re.fullmatch",
    "getattr",
}


class _NullUseChecker(ast.NodeVisitor):
    """Detect uses of possibly-None values without guards."""

    def __init__(self, nullable_vars: Set[str], file: str):
        self.nullable_vars = set(nullable_vars)
        self.file = file
        self.bugs: List[Bug] = []
        self._guarded: Set[str] = set()
        self._depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._depth > 0:
            return
        self._depth += 1
        old_guarded = set(self._guarded)
        self.generic_visit(node)
        self._guarded = old_guarded
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        guarded = self._extract_null_guards(node.test)
        old = set(self._guarded)
        self._guarded.update(guarded)
        for stmt in node.body:
            self.visit(stmt)
        self._guarded = old
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            if node.value.id in self.nullable_vars and node.value.id not in self._guarded:
                self.bugs.append(Bug(
                    kind=BugKind.NULL_DEREFERENCE,
                    message=f"Attribute access '.{node.attr}' on possibly-None '{node.value.id}'",
                    file=self.file,
                    line=node.lineno,
                    column=node.col_offset,
                ))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in self.nullable_vars and node.func.value.id not in self._guarded:
                self.bugs.append(Bug(
                    kind=BugKind.NULL_DEREFERENCE,
                    message=f"Method call '.{node.func.attr}()' on possibly-None '{node.func.value.id}'",
                    file=self.file,
                    line=node.lineno,
                    column=node.col_offset,
                ))
        self.generic_visit(node)

    def _extract_null_guards(self, test: ast.expr) -> Set[str]:
        guarded: Set[str] = set()
        if isinstance(test, ast.Compare):
            if len(test.ops) == 1:
                comp = test.comparators[0]
                if isinstance(test.ops[0], ast.IsNot) and isinstance(comp, ast.Constant) and comp.value is None:
                    if isinstance(test.left, ast.Name):
                        guarded.add(test.left.id)
        if isinstance(test, ast.Name):
            guarded.add(test.id)
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            for val in test.values:
                guarded.update(self._extract_null_guards(val))
        return guarded


def interprocedural_null_analysis(project_dir: str) -> List[Bug]:
    """Detect cross-function null dereference bugs.

    1. Find functions that can return None.
    2. Track where their return values are used without null checks.
    """
    bugs: List[Bug] = []
    files = _iter_python_files(project_dir)

    none_returning: Dict[str, Set[str]] = {}

    for fpath in files:
        tree = _parse_file(fpath)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                checker = _NullReturnChecker()
                checker.visit(node)
                if checker.can_return_none:
                    qname = _qualified_name(fpath, project_dir, node.name)
                    none_returning.setdefault(fpath, set()).add(node.name)

    none_returning_flat: Set[str] = set()
    for names in none_returning.values():
        none_returning_flat.update(names)

    for fpath in files:
        tree = _parse_file(fpath)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nullable_from_calls: Set[str] = set()
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Assign):
                        if isinstance(stmt.value, ast.Call):
                            call_name = _get_call_name(stmt.value)
                            base_name = call_name.split(".")[-1] if "." in call_name else call_name
                            if base_name in none_returning_flat or call_name in _RETURNS_OPTIONAL:
                                for target in stmt.targets:
                                    if isinstance(target, ast.Name):
                                        nullable_from_calls.add(target.id)

                if nullable_from_calls:
                    checker = _NullUseChecker(nullable_from_calls, fpath)
                    checker.visit(node)
                    bugs.extend(checker.bugs)

    return bugs


# ── Taint analysis across modules ───────────────────────────────────────────

DEFAULT_SOURCES: List[TaintSource] = [
    TaintSource("user_input", ["input", "request.GET", "request.POST", "request.args",
                                 "request.form", "request.data", "request.json",
                                 "sys.argv", "os.environ"]),
    TaintSource("file_read", ["open", "read", "readline", "readlines"]),
    TaintSource("env_var", ["os.environ", "os.getenv"]),
]

DEFAULT_SINKS: List[TaintSink] = [
    TaintSink("sql", ["execute", "executemany", "raw", "cursor.execute"], "89"),
    TaintSink("command", ["os.system", "subprocess.run", "subprocess.call",
                           "subprocess.Popen", "os.popen"], "78"),
    TaintSink("file_path", ["open", "os.path.join", "Path"], "22"),
    TaintSink("eval", ["eval", "exec", "compile"], "95"),
    TaintSink("deserialization", ["pickle.loads", "yaml.load", "marshal.loads"], "502"),
    TaintSink("network", ["requests.get", "requests.post", "urllib.request.urlopen",
                           "httpx.get", "aiohttp.ClientSession"], "918"),
]


class _TaintTracker(ast.NodeVisitor):
    """Track taint flow within a single function."""

    def __init__(self, source_patterns: Set[str], sink_patterns: Set[str]):
        self.source_patterns = source_patterns
        self.sink_patterns = sink_patterns
        self.tainted_vars: Dict[str, List[str]] = {}
        self.flows: List[Tuple[str, str, List[str], int, int]] = []
        self._depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._depth > 0:
            return
        self._depth += 1
        # Taint parameters with suggestive names
        for arg in node.args.args:
            if arg.arg in ("user_input", "query", "data", "payload", "url",
                          "path", "command", "cmd", "sql", "body"):
                self.tainted_vars[arg.arg] = [f"param:{arg.arg}"]
        self.generic_visit(node)
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        sources_found = self._check_source(node.value)
        if sources_found:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.tainted_vars[target.id] = sources_found
        elif self._expr_is_tainted(node.value):
            taint_path = self._get_taint_path(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.tainted_vars[target.id] = taint_path + [target.id]
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _get_call_name(node)
        for pattern in self.sink_patterns:
            if call_name.endswith(pattern) or pattern.endswith(call_name):
                for arg in node.args:
                    if self._expr_is_tainted(arg):
                        path = self._get_taint_path(arg)
                        self.flows.append((
                            path[0] if path else "unknown",
                            call_name,
                            path + [call_name],
                            node.lineno,
                            node.col_offset,
                        ))
                    elif isinstance(arg, ast.JoinedStr):
                        for val in arg.values:
                            if isinstance(val, ast.FormattedValue) and self._expr_is_tainted(val.value):
                                path = self._get_taint_path(val.value)
                                self.flows.append((
                                    path[0] if path else "unknown",
                                    call_name,
                                    path + [f"f-string", call_name],
                                    node.lineno,
                                    node.col_offset,
                                ))
        self.generic_visit(node)

    def _check_source(self, node: ast.expr) -> List[str]:
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            for pattern in self.source_patterns:
                if name.endswith(pattern) or pattern.endswith(name):
                    return [f"source:{name}"]
        if isinstance(node, ast.Attribute):
            full = self._attr_chain(node)
            for pattern in self.source_patterns:
                if full.endswith(pattern) or pattern in full:
                    return [f"source:{full}"]
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            full = self._attr_chain(node.value)
            for pattern in self.source_patterns:
                if full.endswith(pattern) or pattern in full:
                    return [f"source:{full}"]
        return []

    def _expr_is_tainted(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.tainted_vars
        if isinstance(node, ast.BinOp):
            return self._expr_is_tainted(node.left) or self._expr_is_tainted(node.right)
        if isinstance(node, ast.Call):
            return any(self._expr_is_tainted(a) for a in node.args)
        if isinstance(node, ast.JoinedStr):
            for val in node.values:
                if isinstance(val, ast.FormattedValue) and self._expr_is_tainted(val.value):
                    return True
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return node.value.id in self.tainted_vars
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            return node.value.id in self.tainted_vars
        return False

    def _get_taint_path(self, node: ast.expr) -> List[str]:
        if isinstance(node, ast.Name) and node.id in self.tainted_vars:
            return list(self.tainted_vars[node.id])
        if isinstance(node, ast.BinOp):
            if self._expr_is_tainted(node.left):
                return self._get_taint_path(node.left)
            return self._get_taint_path(node.right)
        if isinstance(node, ast.Call):
            for a in node.args:
                if self._expr_is_tainted(a):
                    return self._get_taint_path(a)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in self.tainted_vars:
                return list(self.tainted_vars[node.value.id]) + [f".{node.attr}"]
        return ["unknown"]

    def _attr_chain(self, node: ast.expr) -> str:
        if isinstance(node, ast.Attribute):
            return f"{self._attr_chain(node.value)}.{node.attr}"
        if isinstance(node, ast.Name):
            return node.id
        return ""


def taint_across_modules(
    project_dir: str,
    sources: Optional[List[TaintSource]] = None,
    sinks: Optional[List[TaintSink]] = None,
) -> List[TaintPath]:
    """Run taint analysis across all modules in *project_dir*."""
    if sources is None:
        sources = DEFAULT_SOURCES
    if sinks is None:
        sinks = DEFAULT_SINKS

    source_patterns: Set[str] = set()
    for s in sources:
        source_patterns.update(s.patterns)

    sink_patterns: Set[str] = set()
    sink_cwe: Dict[str, str] = {}
    for s in sinks:
        for p in s.patterns:
            sink_patterns.add(p)
            if s.cwe_id:
                sink_cwe[p] = s.cwe_id

    results: List[TaintPath] = []
    files = _iter_python_files(project_dir)

    for fpath in files:
        tree = _parse_file(fpath)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                tracker = _TaintTracker(source_patterns, sink_patterns)
                tracker.visit(node)
                for src, snk, path, line, col in tracker.flows:
                    cwe = None
                    for p, c in sink_cwe.items():
                        if p in snk or snk.endswith(p):
                            cwe = c
                            break
                    results.append(TaintPath(
                        source=src,
                        sink=snk,
                        path=path,
                        file=fpath,
                        source_line=0,
                        sink_line=line,
                        cwe_id=cwe,
                    ))
    return results


# ── Effect inference ─────────────────────────────────────────────────────────

IO_CALLS: Dict[str, EffectKind] = {
    "print": EffectKind.PRINT,
    "open": EffectKind.FILESYSTEM,
    "os.read": EffectKind.IO_READ,
    "os.write": EffectKind.IO_WRITE,
    "os.remove": EffectKind.FILESYSTEM,
    "os.unlink": EffectKind.FILESYSTEM,
    "os.rename": EffectKind.FILESYSTEM,
    "os.mkdir": EffectKind.FILESYSTEM,
    "os.makedirs": EffectKind.FILESYSTEM,
    "os.rmdir": EffectKind.FILESYSTEM,
    "os.listdir": EffectKind.FILESYSTEM,
    "shutil.copy": EffectKind.FILESYSTEM,
    "shutil.move": EffectKind.FILESYSTEM,
    "shutil.rmtree": EffectKind.FILESYSTEM,
    "requests.get": EffectKind.NETWORK,
    "requests.post": EffectKind.NETWORK,
    "requests.put": EffectKind.NETWORK,
    "requests.delete": EffectKind.NETWORK,
    "requests.patch": EffectKind.NETWORK,
    "urllib.request.urlopen": EffectKind.NETWORK,
    "httpx.get": EffectKind.NETWORK,
    "httpx.post": EffectKind.NETWORK,
    "aiohttp.ClientSession": EffectKind.NETWORK,
    "subprocess.run": EffectKind.SUBPROCESS,
    "subprocess.call": EffectKind.SUBPROCESS,
    "subprocess.Popen": EffectKind.SUBPROCESS,
    "subprocess.check_output": EffectKind.SUBPROCESS,
    "os.system": EffectKind.SUBPROCESS,
    "os.popen": EffectKind.SUBPROCESS,
    "random.random": EffectKind.RANDOM,
    "random.randint": EffectKind.RANDOM,
    "random.choice": EffectKind.RANDOM,
    "random.shuffle": EffectKind.RANDOM,
    "time.time": EffectKind.TIME,
    "time.sleep": EffectKind.TIME,
    "datetime.datetime.now": EffectKind.TIME,
    "os.environ.get": EffectKind.ENV_READ,
    "os.getenv": EffectKind.ENV_READ,
    "cursor.execute": EffectKind.DATABASE,
    "connection.execute": EffectKind.DATABASE,
    "session.execute": EffectKind.DATABASE,
    "session.commit": EffectKind.DATABASE,
    "session.add": EffectKind.DATABASE,
    "db.session.add": EffectKind.DATABASE,
    "logging.info": EffectKind.IO_WRITE,
    "logging.debug": EffectKind.IO_WRITE,
    "logging.warning": EffectKind.IO_WRITE,
    "logging.error": EffectKind.IO_WRITE,
    "logger.info": EffectKind.IO_WRITE,
    "logger.debug": EffectKind.IO_WRITE,
    "logger.warning": EffectKind.IO_WRITE,
    "logger.error": EffectKind.IO_WRITE,
}

MUTATION_METHODS: Set[str] = {
    "append", "extend", "insert", "remove", "pop", "clear",
    "add", "discard", "update", "sort", "reverse",
    "setdefault", "__setitem__", "__delitem__",
}


class _EffectCollector(ast.NodeVisitor):
    """Collect all effects in a function body."""

    def __init__(self) -> None:
        self.effects: List[Effect] = []
        self._depth = 0
        self._globals_written: Set[str] = set()
        self._global_names: Set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._depth > 0:
            return
        self._depth += 1
        for stmt in node.body:
            if isinstance(stmt, ast.Global):
                self._global_names.update(stmt.names)
        self.generic_visit(node)
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        name = _get_call_name(node)
        for pattern, kind in IO_CALLS.items():
            if name == pattern or name.endswith(f".{pattern}"):
                self.effects.append(Effect(
                    kind=kind, description=f"Call to {name}",
                    line=node.lineno, target=name,
                ))
                break
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in MUTATION_METHODS:
                self.effects.append(Effect(
                    kind=EffectKind.MUTATION,
                    description=f"Mutation via .{node.func.attr}()",
                    line=node.lineno,
                    target=node.func.attr,
                ))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in self._global_names:
                self._globals_written.add(target.id)
                self.effects.append(Effect(
                    kind=EffectKind.GLOBAL_WRITE,
                    description=f"Write to global '{target.id}'",
                    line=node.lineno,
                    target=target.id,
                ))
            elif isinstance(target, ast.Subscript):
                self.effects.append(Effect(
                    kind=EffectKind.MUTATION,
                    description="Subscript assignment (mutation)",
                    line=node.lineno,
                ))
            elif isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name) and target.value.id != "self":
                    self.effects.append(Effect(
                        kind=EffectKind.MUTATION,
                        description=f"Attribute mutation on '{target.value.id}'",
                        line=node.lineno,
                        target=f"{target.value.id}.{target.attr}",
                    ))
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        exc_name = ""
        if node.exc:
            if isinstance(node.exc, ast.Call):
                exc_name = _get_call_name(node.exc)
            elif isinstance(node.exc, ast.Name):
                exc_name = node.exc.id
        self.effects.append(Effect(
            kind=EffectKind.EXCEPTION,
            description=f"Raises {exc_name}" if exc_name else "Raises exception",
            line=node.lineno,
            target=exc_name,
        ))


def effect_inference(project_dir: str) -> Dict[str, Effects]:
    """Infer effects (IO, mutation, exceptions) for every function in *project_dir*."""
    result: Dict[str, Effects] = {}
    files = _iter_python_files(project_dir)
    call_graph = analyze_call_graph(project_dir)

    for fpath in files:
        tree = _parse_file(fpath)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                collector = _EffectCollector()
                collector.visit(node)
                qname = _qualified_name(fpath, project_dir, node.name)
                is_pure = len(collector.effects) == 0
                result[qname] = Effects(
                    function=qname,
                    effects=collector.effects,
                    is_pure=is_pure,
                )

    # Propagate transitive effects
    for fn_name, effects_obj in result.items():
        transitive: List[Effect] = []
        for callee in call_graph.transitive_callees(fn_name):
            if callee in result:
                transitive.extend(result[callee].effects)
        effects_obj.transitive_effects = transitive
        if transitive:
            effects_obj.is_pure = False

    return result


# ── Purity analysis ──────────────────────────────────────────────────────────

def purity_analysis(project_dir: str) -> Dict[str, bool]:
    """Determine which functions in *project_dir* are pure.

    A function is pure if it has no effects (IO, mutation, exceptions,
    global writes) either directly or transitively through callees.
    """
    effects = effect_inference(project_dir)
    return {fn: eff.is_pure for fn, eff in effects.items()}


# ── Escape analysis ─────────────────────────────────────────────────────────

class _EscapeAnalyzer(ast.NodeVisitor):
    """Analyze where locally-created objects escape to."""

    def __init__(self) -> None:
        self.escapes: Dict[str, EscapeInfo] = {}
        self._locals: Set[str] = set()
        self._depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._depth > 0:
            return
        self._depth += 1
        for arg in node.args.args:
            self._locals.add(arg.arg)
        self.generic_visit(node)
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._locals.add(target.id)
                if isinstance(node.value, (ast.List, ast.Dict, ast.Set, ast.Call)):
                    self.escapes[target.id] = EscapeInfo(
                        variable=target.id,
                        kind=EscapeKind.NO_ESCAPE,
                        line=node.lineno,
                    )
            elif isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name):
                    if isinstance(node.value, ast.Name) and node.value.id in self.escapes:
                        self.escapes[node.value.id] = EscapeInfo(
                            variable=node.value.id,
                            kind=EscapeKind.HEAP_ESCAPE,
                            escape_point=f"{target.value.id}.{target.attr}",
                            line=node.lineno,
                        )
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value and isinstance(node.value, ast.Name):
            name = node.value.id
            if name in self.escapes:
                self.escapes[name] = EscapeInfo(
                    variable=name,
                    kind=EscapeKind.RETURN_ESCAPE,
                    escape_point="return",
                    line=node.lineno,
                )

    def visit_Call(self, node: ast.Call) -> None:
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id in self.escapes:
                self.escapes[arg.id] = EscapeInfo(
                    variable=arg.id,
                    kind=EscapeKind.ARG_ESCAPE,
                    escape_point=_get_call_name(node),
                    line=node.lineno,
                )
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            if name in self.escapes:
                self.escapes[name] = EscapeInfo(
                    variable=name,
                    kind=EscapeKind.GLOBAL_ESCAPE,
                    escape_point="global",
                    line=node.lineno,
                )


def escape_analysis(source: str) -> Dict[str, EscapeInfo]:
    """Analyze escape behavior of variables in *source*.

    For each locally-created object, determine whether it:
    - Stays local (NO_ESCAPE)
    - Escapes via a function argument (ARG_ESCAPE)
    - Escapes via return (RETURN_ESCAPE)
    - Escapes to the global scope (GLOBAL_ESCAPE)
    - Escapes to the heap via assignment to another object's attribute (HEAP_ESCAPE)
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    all_escapes: Dict[str, EscapeInfo] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            analyzer = _EscapeAnalyzer()
            analyzer.visit(node)
            for var, info in analyzer.escapes.items():
                key = f"{node.name}.{var}"
                all_escapes[key] = info

    return all_escapes


# ═══════════════════════════════════════════════════════════════════════════
# Liquid type contract propagation
# ═══════════════════════════════════════════════════════════════════════════

def liquid_interprocedural_analysis(project_dir: str) -> Dict[str, Any]:
    """Interprocedural liquid type contract inference across a project.

    1. Builds the call graph (reusing ``analyze_call_graph``)
    2. Processes functions in reverse topological order
    3. Infers liquid type contracts per function
    4. Propagates contracts to callers for interprocedural precision

    Args:
        project_dir: Root directory of the Python project.

    Returns:
        Dictionary mapping qualified function names to ``FunctionContract``
        objects.  Returns empty dict if Z3 is not available.
    """
    try:
        from src.liquid import LiquidTypeInferencer, FunctionContract
    except Exception:
        return {}

    graph = analyze_call_graph(project_dir)

    # Compute reverse topological order via Kahn's algorithm
    in_degree: Dict[str, int] = {n: 0 for n in graph.nodes}
    for src_node in graph.nodes:
        for callee in graph.callees.get(src_node, set()):
            if callee in in_degree:
                in_degree[callee] += 1

    # Start from leaf functions (callees first)
    queue = [n for n, d in in_degree.items() if d == 0]
    topo_order: List[str] = []
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for caller in graph.callers.get(node, set()):
            if caller in in_degree:
                in_degree[caller] -= 1
                if in_degree[caller] == 0:
                    queue.append(caller)

    # Add any remaining nodes (cycles)
    for n in graph.nodes:
        if n not in topo_order:
            topo_order.append(n)

    # Reverse: process callees before callers
    topo_order.reverse()

    # Collect per-file ASTs and sources
    files = _iter_python_files(project_dir)
    file_sources: Dict[str, str] = {}
    file_trees: Dict[str, ast.Module] = {}
    func_to_file: Dict[str, str] = {}

    for fpath in files:
        tree = _parse_file(fpath)
        if tree is None:
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                src = f.read()
        except Exception:
            continue
        file_sources[fpath] = src
        file_trees[fpath] = tree
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qname = _qualified_name(fpath, project_dir, node.name)
                func_to_file[qname] = fpath

    # Infer contracts in topological order, propagating to callers
    all_contracts: Dict[str, Any] = {}
    engine = LiquidTypeInferencer()

    for func_qname in topo_order:
        fpath = func_to_file.get(func_qname)
        if fpath is None:
            continue
        source = file_sources.get(fpath, "")
        if not source:
            continue

        # Infer contracts for the whole file (cached per file)
        if fpath not in getattr(engine, "_file_cache", {}):
            if not hasattr(engine, "_file_cache"):
                engine._file_cache = {}  # type: ignore[attr-defined]
            try:
                result = engine.infer_module(source)
                engine._file_cache[fpath] = result  # type: ignore[attr-defined]
            except Exception:
                engine._file_cache[fpath] = None  # type: ignore[attr-defined]

        cached = engine._file_cache.get(fpath)  # type: ignore[attr-defined]
        if cached is None:
            continue

        # Extract the contract for this function
        short_name = func_qname.rsplit(".", 1)[-1] if "." in func_qname else func_qname
        contract = cached.contracts.get(short_name)
        if contract is not None:
            all_contracts[func_qname] = contract

    return all_contracts


class LiquidCallGraphAnalyzer:
    """Extends call graph analysis with liquid type contract information.

    Combines structural call graph with semantic liquid type contracts
    for interprocedural reasoning.
    """

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.call_graph: Optional[CallGraph] = None
        self.contracts: Dict[str, Any] = {}

    def analyze(self) -> "LiquidCallGraphAnalyzer":
        """Build call graph and infer liquid contracts."""
        self.call_graph = analyze_call_graph(self.project_dir)
        self.contracts = liquid_interprocedural_analysis(self.project_dir)
        return self

    def get_contract(self, func_name: str) -> Optional[Any]:
        """Get liquid type contract for a function, if available."""
        return self.contracts.get(func_name)

    def callee_contracts(self, func_name: str) -> Dict[str, Any]:
        """Get contracts for all callees of *func_name*."""
        if self.call_graph is None:
            return {}
        callees = self.call_graph.callees.get(func_name, set())
        return {
            callee: self.contracts[callee]
            for callee in callees
            if callee in self.contracts
        }

    def summary(self) -> str:
        """Human-readable summary of the analysis."""
        n_nodes = len(self.call_graph.nodes) if self.call_graph else 0
        n_edges = len(self.call_graph.edges) if self.call_graph else 0
        n_contracts = len(self.contracts)
        return (
            f"LiquidCallGraphAnalyzer: {n_nodes} functions, "
            f"{n_edges} call edges, {n_contracts} liquid contracts inferred"
        )


# ── Interprocedural shape propagation ────────────────────────────────────────

@dataclass
class ShapeContract:
    """Shape contract for a function: maps parameter shapes to return shape."""
    param_shapes: Dict[str, Optional[Tuple[Union[int, str], ...]]] = field(
        default_factory=dict
    )
    return_shape: Optional[Tuple[Union[int, str], ...]] = None
    is_nn_module_forward: bool = False


class _ShapeReturnVisitor(ast.NodeVisitor):
    """Extract tensor return shapes from a function body."""

    def __init__(self):
        self.return_shapes: List[Optional[Tuple[Union[int, str], ...]]] = []
        self._shapes: Dict[str, Tuple[Union[int, str], ...]] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        shape = self._extract_shape(node.value)
        if shape is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._shapes[target.id] = shape
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None:
            self.return_shapes.append(None)
            return
        shape = self._extract_shape(node.value)
        if shape is None and isinstance(node.value, ast.Name):
            shape = self._shapes.get(node.value.id)
        self.return_shapes.append(shape)

    def _extract_shape(self, node: ast.expr) -> Optional[Tuple[Union[int, str], ...]]:
        """Try to extract a tensor shape from a constructor or matmul call."""
        if not isinstance(node, ast.Call):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                left = self._extract_shape_or_lookup(node.left)
                right = self._extract_shape_or_lookup(node.right)
                if left is not None and right is not None and len(left) >= 1 and len(right) >= 2:
                    return left[:-1] + (right[-1],)
            return None
        fname = _get_call_name(node)
        if fname is None:
            return None
        stem = fname.split(".")[-1]
        if stem in ("zeros", "ones", "randn", "rand", "empty", "full"):
            dims: List[Union[int, str]] = []
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    dims.append(arg.value)
                elif isinstance(arg, ast.Name):
                    dims.append(arg.id)
                else:
                    return None
            if dims:
                return tuple(dims)
        return None

    def _extract_shape_or_lookup(self, node: ast.expr) -> Optional[Tuple[Union[int, str], ...]]:
        if isinstance(node, ast.Name):
            return self._shapes.get(node.id)
        return self._extract_shape(node)


class InterproceduralShapeAnalyzer:
    """Propagates tensor shape constraints across function boundaries.

    Extends the call graph infrastructure to:
      1. Infer shape contracts (parameter shapes → return shape) per function
      2. Propagate return shapes to callers in topological order
      3. Annotate nn.Module.forward() chains with shape flow
    """

    def __init__(self):
        self.shape_contracts: Dict[str, ShapeContract] = {}
        self.call_graph: Optional[CallGraph] = None
        self.errors: List[str] = []

    def analyze_source(self, source: str) -> Dict[str, ShapeContract]:
        """Analyze a single source string for interprocedural shape contracts."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}
        self._infer_from_module(tree)
        self._propagate_shapes(tree)
        return dict(self.shape_contracts)

    def analyze_project(self, project_dir: str) -> Dict[str, ShapeContract]:
        """Analyze an entire project for interprocedural shape contracts."""
        self.call_graph = analyze_call_graph(project_dir)

        files = _iter_python_files(project_dir)
        func_asts: Dict[str, ast.FunctionDef] = {}
        for fpath in files:
            tree = _parse_file(fpath)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qname = _qualified_name(fpath, project_dir, node.name)
                    func_asts[qname] = node

        # Compute topological order (callees first)
        in_degree: Dict[str, int] = {n: 0 for n in self.call_graph.nodes}
        for src_node in self.call_graph.nodes:
            for callee in self.call_graph.callees.get(src_node, set()):
                if callee in in_degree:
                    in_degree[callee] += 1
        queue = [n for n, d in in_degree.items() if d == 0]
        topo: List[str] = []
        while queue:
            node = queue.pop(0)
            topo.append(node)
            for caller in self.call_graph.callers.get(node, set()):
                if caller in in_degree:
                    in_degree[caller] -= 1
                    if in_degree[caller] == 0:
                        queue.append(caller)
        for n in self.call_graph.nodes:
            if n not in topo:
                topo.append(n)
        topo.reverse()

        for func_qname in topo:
            fdef = func_asts.get(func_qname)
            if fdef is None:
                continue
            self._infer_function_contract(func_qname, fdef)

        return dict(self.shape_contracts)

    def _infer_from_module(self, tree: ast.Module) -> None:
        """Infer shape contracts for all functions in a module AST."""
        class_name: Optional[str] = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                is_module = any(
                    (isinstance(b, ast.Attribute) and b.attr == "Module")
                    or (isinstance(b, ast.Name) and b.id == "Module")
                    for b in node.bases
                )
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        qname = f"{node.name}.{item.name}"
                        self._infer_function_contract(qname, item)
                        if is_module and item.name == "forward":
                            self.shape_contracts[qname] = ShapeContract(
                                **{**self.shape_contracts.get(qname, ShapeContract()).__dict__,
                                   "is_nn_module_forward": True}
                            )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._infer_function_contract(node.name, node)

        # Propagate shapes through calls
        self._propagate_shapes(tree)

    def _infer_function_contract(
        self, func_name: str, fdef: ast.FunctionDef
    ) -> None:
        """Infer a ShapeContract for a single function."""
        visitor = _ShapeReturnVisitor()

        # Seed visitor with shapes from callee contracts
        for node in ast.walk(fdef):
            if isinstance(node, ast.Call):
                call_name = _get_call_name(node)
                if call_name and call_name in self.shape_contracts:
                    callee_contract = self.shape_contracts[call_name]
                    if callee_contract.return_shape is not None:
                        # Find assignment target
                        pass  # handled in visit_Assign

        visitor.visit(fdef)

        # Determine return shape (take the first non-None)
        ret_shape: Optional[Tuple[Union[int, str], ...]] = None
        for rs in visitor.return_shapes:
            if rs is not None:
                ret_shape = rs
                break

        contract = ShapeContract(return_shape=ret_shape)

        # Extract parameter shapes from type annotations or defaults
        for arg in fdef.args.args:
            contract.param_shapes[arg.arg] = None

        self.shape_contracts[func_name] = contract

    def _propagate_shapes(self, tree: ast.Module) -> None:
        """Propagate return shapes from callees to callers."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            call_name = _get_call_name(node.value)
            if call_name is None:
                continue

            # Look up callee contract (try both bare name and qualified)
            contract = self.shape_contracts.get(call_name)
            if contract is None:
                # Try matching against Class.method pattern for self.method() calls
                for key, c in self.shape_contracts.items():
                    if key.endswith(f".{call_name}") or key.endswith(f".{call_name.split('.')[-1]}"):
                        contract = c
                        break
            if contract is None or contract.return_shape is None:
                continue

            # Map return shape to the assignment target in _ShapeReturnVisitor
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Store for downstream visitors
                    for other in ast.walk(tree):
                        if isinstance(other, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            for sub in ast.walk(other):
                                if isinstance(sub, ast.Assign):
                                    visitor = _ShapeReturnVisitor()
                                    visitor._shapes[target.id] = contract.return_shape
                                    # The shape is now available for return analysis
