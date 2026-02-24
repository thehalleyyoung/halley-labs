"""Python API usage analyzer.

Analyzes source code to detect API usage patterns, deprecated APIs,
compatibility issues across Python versions, and constructs call graphs
with argument type information.
"""
from __future__ import annotations
import ast, copy, textwrap, re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
import numpy as np

@dataclass
class ImportRecord:
    module: str
    names: List[str] = field(default_factory=list)
    alias: Optional[str] = None
    lineno: int = 0
    is_from: bool = False

@dataclass
class ImportReport:
    all_imports: List[ImportRecord] = field(default_factory=list)
    used_names: Set[str] = field(default_factory=set)
    unused_names: Set[str] = field(default_factory=set)
    import_frequency: Dict[str, int] = field(default_factory=dict)
    import_graph: Dict[str, Set[str]] = field(default_factory=dict)
    @property
    def total_imports(self) -> int:
        return len(self.all_imports)
    @property
    def unused_ratio(self) -> float:
        t = len(self.used_names) + len(self.unused_names)
        return len(self.unused_names) / t if t else 0.0

@dataclass
class DeprecatedUsage:
    module: str
    name: str
    lineno: int
    deprecated_since: str
    replacement: str
    severity: str = "warning"

@dataclass
class CompatibilityIssue:
    feature: str
    min_version: str
    lineno: int
    description: str
    node_type: str = ""

@dataclass
class StubStatus:
    package: str
    has_stubs: bool
    stub_package: Optional[str] = None

@dataclass
class StubReport:
    statuses: List[StubStatus] = field(default_factory=list)
    @property
    def typed_count(self) -> int:
        return sum(1 for s in self.statuses if s.has_stubs)
    @property
    def untyped_count(self) -> int:
        return sum(1 for s in self.statuses if not s.has_stubs)
    @property
    def coverage(self) -> float:
        return self.typed_count / len(self.statuses) if self.statuses else 0.0

@dataclass
class CallEdge:
    caller: str
    callee: str
    lineno: int
    arg_types: List[str] = field(default_factory=list)

@dataclass
class CallGraph:
    nodes: Set[str] = field(default_factory=set)
    edges: List[CallEdge] = field(default_factory=list)
    _caller_map: Dict[str, List[CallEdge]] = field(default_factory=lambda: defaultdict(list))
    _callee_map: Dict[str, List[CallEdge]] = field(default_factory=lambda: defaultdict(list))
    def add_node(self, name: str) -> None:
        self.nodes.add(name)
    def add_edge(self, edge: CallEdge) -> None:
        self.edges.append(edge)
        self.nodes.add(edge.caller)
        self.nodes.add(edge.callee)
        self._caller_map[edge.callee].append(edge)
        self._callee_map[edge.caller].append(edge)
    def get_callers(self, func: str) -> List[str]:
        return [e.caller for e in self._caller_map.get(func, [])]
    def get_callees(self, func: str) -> List[str]:
        return [e.callee for e in self._callee_map.get(func, [])]
    def get_edges_for(self, func: str) -> List[CallEdge]:
        out = list(self._callee_map.get(func, []))
        out.extend(self._caller_map.get(func, []))
        return out
    @property
    def adjacency_matrix(self) -> np.ndarray:
        ordered = sorted(self.nodes)
        idx = {n: i for i, n in enumerate(ordered)}
        mat = np.zeros((len(ordered), len(ordered)), dtype=np.int32)
        for e in self.edges:
            mat[idx[e.caller], idx[e.callee]] += 1
        return mat

@dataclass
class FunctionInfo:
    name: str
    qualname: str
    lineno: int
    params: List[str] = field(default_factory=list)
    return_annotation: Optional[str] = None
    is_method: bool = False
    is_private: bool = False
    decorators: List[str] = field(default_factory=list)

@dataclass
class ClassInfo:
    name: str
    lineno: int
    bases: List[str] = field(default_factory=list)
    methods: List[FunctionInfo] = field(default_factory=list)
    is_private: bool = False

@dataclass
class APISurface:
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    @property
    def public_functions(self) -> List[FunctionInfo]:
        return [f for f in self.functions if not f.is_private]
    @property
    def public_classes(self) -> List[ClassInfo]:
        return [c for c in self.classes if not c.is_private]
    @property
    def total_public(self) -> int:
        return len(self.public_functions) + len(self.public_classes) + len(self.constants)

class ChangeKind(Enum):
    REMOVED_FUNCTION = auto()
    REMOVED_CLASS = auto()
    REMOVED_METHOD = auto()
    CHANGED_SIGNATURE = auto()
    REMOVED_PARAMETER = auto()
    ADDED_REQUIRED_PARAMETER = auto()
    CHANGED_DEFAULT = auto()
    CHANGED_RETURN_TYPE = auto()

@dataclass
class BreakingChange:
    kind: ChangeKind
    name: str
    detail: str
    old_lineno: int = 0
    new_lineno: int = 0

@dataclass
class DocCoverageReport:
    documented: List[str] = field(default_factory=list)
    undocumented: List[str] = field(default_factory=list)
    @property
    def total(self) -> int:
        return len(self.documented) + len(self.undocumented)
    @property
    def coverage(self) -> float:
        return len(self.documented) / self.total if self.total else 1.0

@dataclass
class APIUsageReport:
    imports: ImportReport = field(default_factory=ImportReport)
    deprecated: List[DeprecatedUsage] = field(default_factory=list)
    compatibility: List[CompatibilityIssue] = field(default_factory=list)
    stubs: StubReport = field(default_factory=StubReport)
    call_graph: CallGraph = field(default_factory=CallGraph)
    surface: APISurface = field(default_factory=APISurface)
    doc_coverage: DocCoverageReport = field(default_factory=DocCoverageReport)
    def summary_vector(self) -> np.ndarray:
        return np.array([
            self.imports.total_imports, len(self.imports.unused_names),
            len(self.deprecated), len(self.compatibility),
            self.stubs.typed_count, self.stubs.untyped_count,
            len(self.call_graph.nodes), len(self.call_graph.edges),
            self.surface.total_public, self.doc_coverage.total,
            self.doc_coverage.coverage,
        ], dtype=np.float64)

DEPRECATED_APIS: Dict[str, Tuple[str, str]] = {
    "asyncio.coroutine": ("3.8", "Use 'async def' instead"),
    "asyncio.Task.current_task": ("3.10", "Use asyncio.current_task()"),
    "asyncio.Task.all_tasks": ("3.10", "Use asyncio.all_tasks()"),
    "collections.MutableMapping": ("3.3", "Use collections.abc.MutableMapping"),
    "collections.MutableSequence": ("3.3", "Use collections.abc.MutableSequence"),
    "collections.MutableSet": ("3.3", "Use collections.abc.MutableSet"),
    "collections.Mapping": ("3.3", "Use collections.abc.Mapping"),
    "collections.Sequence": ("3.3", "Use collections.abc.Sequence"),
    "collections.Iterable": ("3.3", "Use collections.abc.Iterable"),
    "collections.Iterator": ("3.3", "Use collections.abc.Iterator"),
    "collections.Callable": ("3.3", "Use collections.abc.Callable"),
    "collections.ByteString": ("3.3", "Use collections.abc.ByteString"),
    "imp.find_module": ("3.4", "Use importlib.util.find_spec"),
    "imp.load_module": ("3.4", "Use importlib.import_module"),
    "imp.reload": ("3.4", "Use importlib.reload"),
    "optparse.OptionParser": ("3.2", "Use argparse.ArgumentParser"),
    "formatter.AbstractFormatter": ("3.4", "No direct replacement"),
    "formatter.DumbWriter": ("3.4", "No direct replacement"),
    "distutils.core.setup": ("3.10", "Use setuptools.setup"),
    "distutils.core.Extension": ("3.10", "Use setuptools.Extension"),
    "typing.Dict": ("3.9", "Use dict instead"),
    "typing.List": ("3.9", "Use list instead"),
    "typing.Set": ("3.9", "Use set instead"),
    "typing.Tuple": ("3.9", "Use tuple instead"),
    "typing.FrozenSet": ("3.9", "Use frozenset instead"),
    "typing.Type": ("3.9", "Use type instead"),
    "os.popen": ("3.0", "Use subprocess.Popen"),
    "unittest.TestCase.assertEquals": ("3.2", "Use assertEqual"),
    "unittest.TestCase.assertNotEquals": ("3.2", "Use assertNotEqual"),
    "unittest.TestCase.assertRegexpMatches": ("3.2", "Use assertRegex"),
    "cgi.escape": ("3.2", "Use html.escape"),
    "cgi.parse_qs": ("3.0", "Use urllib.parse.parse_qs"),
}

PYTHON_VERSION_CHANGES: Dict[str, List[Dict[str, str]]] = {
    "3.8": [
        {"feature": "walrus_operator", "node": "NamedExpr",
         "desc": "Assignment expressions (:= walrus operator) require 3.8+"},
        {"feature": "positional_only_params", "node": "posonlyargs",
         "desc": "Positional-only parameters (/) require 3.8+"},
    ],
    "3.9": [
        {"feature": "dict_union", "node": "BinOp_BitOr_dict",
         "desc": "Dict union operator (|) requires 3.9+"},
        {"feature": "builtin_generics", "node": "builtin_subscript",
         "desc": "Built-in generic types (list[int]) require 3.9+"},
    ],
    "3.10": [
        {"feature": "match_statement", "node": "Match",
         "desc": "Structural pattern matching (match/case) requires 3.10+"},
        {"feature": "union_type_x_or_y", "node": "BinOp_BitOr_type",
         "desc": "X | Y union type syntax requires 3.10+"},
    ],
    "3.11": [
        {"feature": "exception_groups", "node": "TryStar",
         "desc": "Exception groups (except*) require 3.11+"},
        {"feature": "tomllib", "node": "import_tomllib",
         "desc": "tomllib module requires 3.11+"},
    ],
    "3.12": [
        {"feature": "type_parameter", "node": "TypeAlias",
         "desc": "Type parameter syntax (type X = ...) requires 3.12+"},
    ],
}

TYPED_PACKAGES: Set[str] = {
    "numpy", "pandas", "requests", "flask", "django", "sqlalchemy",
    "click", "pydantic", "fastapi", "attrs", "mypy", "pytest",
    "aiohttp", "httpx", "rich", "typer", "celery", "boto3",
    "pillow", "scipy", "matplotlib", "torch", "cryptography",
}
UNTYPED_PACKAGES: Set[str] = {
    "scrapy", "paramiko", "fabric", "gevent", "eventlet",
    "twisted", "pyserial", "pygments", "pyaudio", "scapy",
}

def _resolve_attr(node: ast.Attribute) -> str:
    """Resolve dotted attribute chain to 'os.path.join' style string."""
    parts: List[str] = [node.attr]
    cur = node.value
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return ""
    parts.reverse()
    return ".".join(parts)

def _version_tuple(v: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))

def _infer_arg_type(node: ast.expr) -> str:
    """Best-effort type inference from a call-site argument node."""
    if isinstance(node, ast.Constant):
        return type(node.value).__name__
    mapping = {
        ast.List: "list", ast.Dict: "dict", ast.Set: "set",
        ast.Tuple: "tuple", ast.JoinedStr: "str",
        ast.ListComp: "list", ast.SetComp: "set",
        ast.DictComp: "dict", ast.GeneratorExp: "generator",
    }
    for cls, name in mapping.items():
        if isinstance(node, cls):
            return name
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        return "call"
    return "unknown"

def _param_names(args: ast.arguments) -> List[str]:
    names: List[str] = []
    for a in getattr(args, "posonlyargs", []):
        names.append(a.arg)
    for a in args.args:
        names.append(a.arg)
    if args.vararg:
        names.append(f"*{args.vararg.arg}")
    for a in args.kwonlyargs:
        names.append(a.arg)
    if args.kwarg:
        names.append(f"**{args.kwarg.arg}")
    return names

def _return_annotation_str(node: ast.FunctionDef) -> Optional[str]:
    return ast.dump(node.returns) if node.returns else None

def _extract_decorators(node: ast.FunctionDef) -> List[str]:
    result: List[str] = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            result.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            result.append(_resolve_attr(dec) or dec.attr)
        elif isinstance(dec, ast.Call):
            fn = dec.func
            if isinstance(fn, ast.Name):
                result.append(fn.id)
            elif isinstance(fn, ast.Attribute):
                result.append(_resolve_attr(fn) or fn.attr)
    return result

class ImportAnalyzer(ast.NodeVisitor):
    """Walk AST to find all imports and track usage."""

    def __init__(self) -> None:
        self._imports: List[ImportRecord] = []
        self._imported_names: Dict[str, ImportRecord] = {}
        self._name_refs: Set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            rec = ImportRecord(module=alias.name, names=[alias.name],
                               alias=alias.asname, lineno=node.lineno, is_from=False)
            self._imports.append(rec)
            self._imported_names[alias.asname or alias.name] = rec
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            rec = ImportRecord(module=mod, names=[alias.name],
                               alias=alias.asname, lineno=node.lineno, is_from=True)
            self._imports.append(rec)
            self._imported_names[alias.asname or alias.name] = rec
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self._name_refs.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        full = _resolve_attr(node)
        if full:
            self._name_refs.add(full)
        self.generic_visit(node)

    def analyze_imports(self, tree: ast.AST) -> ImportReport:
        self.visit(tree)
        used: Set[str] = set()
        unused: Set[str] = set()
        for bound, _rec in self._imported_names.items():
            if bound.split(".")[0] in self._name_refs:
                used.add(bound)
            else:
                unused.add(bound)
        freq = dict(Counter(r.module for r in self._imports))
        graph: Dict[str, Set[str]] = defaultdict(set)
        for rec in self._imports:
            graph[rec.module].update(rec.names)
        return ImportReport(all_imports=self._imports, used_names=used,
                            unused_names=unused, import_frequency=freq,
                            import_graph={k: v for k, v in graph.items()})

class DeprecatedAPIChecker(ast.NodeVisitor):
    """Detect usage of deprecated standard-library APIs."""

    def __init__(self) -> None:
        self._results: List[DeprecatedUsage] = []
        self._from_imports: Dict[str, str] = {}

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            self._from_imports[alias.asname or alias.name] = f"{mod}.{alias.name}"
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        full = _resolve_attr(node)
        if full in DEPRECATED_APIS:
            since, repl = DEPRECATED_APIS[full]
            self._results.append(DeprecatedUsage(
                module=full.rsplit(".", 1)[0], name=full.rsplit(".", 1)[1],
                lineno=node.lineno, deprecated_since=since, replacement=repl))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        qual = self._from_imports.get(node.id, "")
        if qual in DEPRECATED_APIS:
            since, repl = DEPRECATED_APIS[qual]
            self._results.append(DeprecatedUsage(
                module=qual.rsplit(".", 1)[0], name=qual.rsplit(".", 1)[1],
                lineno=node.lineno, deprecated_since=since, replacement=repl))
        self.generic_visit(node)

    def check_deprecated(self, tree: ast.AST) -> List[DeprecatedUsage]:
        self.visit(tree)
        return self._results

class CompatibilityChecker(ast.NodeVisitor):
    """Detect Python-version-specific syntax in an AST."""

    def __init__(self, min_version: str = "3.7", max_version: str = "3.12") -> None:
        self.min_ver = _version_tuple(min_version)
        self.max_ver = _version_tuple(max_version)
        self._issues: List[CompatibilityIssue] = []

    def _record(self, feature: str, ver: str, lineno: int, desc: str,
                node_type: str = "") -> None:
        if _version_tuple(ver) > self.min_ver:
            self._issues.append(CompatibilityIssue(
                feature=feature, min_version=ver, lineno=lineno,
                description=desc, node_type=node_type))

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._record("walrus_operator", "3.8", node.lineno,
                      "Assignment expression (:=) requires 3.8+", "NamedExpr")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if getattr(node.args, "posonlyargs", None):
            self._record("positional_only_params", "3.8", node.lineno,
                          "Positional-only params require 3.8+", "posonlyargs")
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Match(self, node: ast.AST) -> None:
        self._record("match_statement", "3.10", getattr(node, "lineno", 0),
                      "match/case requires 3.10+", "Match")
        self.generic_visit(node)

    def visit_TryStar(self, node: ast.AST) -> None:
        self._record("exception_groups", "3.11", getattr(node, "lineno", 0),
                      "except* requires 3.11+", "TryStar")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.BitOr):
            left_ty = isinstance(node.left, (ast.Name, ast.Subscript, ast.Constant))
            right_ty = isinstance(node.right, (ast.Name, ast.Subscript, ast.Constant))
            if left_ty and right_ty:
                self._record("union_type_x_or_y", "3.10", node.lineno,
                              "X | Y union type syntax requires 3.10+", "BinOp_BitOr")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in (
            "list", "dict", "set", "tuple", "frozenset", "type"):
            self._record("builtin_generics", "3.9", node.lineno,
                          f"Built-in generic {node.value.id}[...] requires 3.9+",
                          "builtin_subscript")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "tomllib":
            self._record("tomllib", "3.11", node.lineno,
                          "tomllib requires 3.11+", "import_tomllib")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "tomllib":
                self._record("tomllib", "3.11", node.lineno,
                              "tomllib requires 3.11+", "import_tomllib")
        self.generic_visit(node)

    def check_compatibility(self, tree: ast.AST, min_version: str = "",
                            max_version: str = "") -> List[CompatibilityIssue]:
        if min_version:
            self.min_ver = _version_tuple(min_version)
        if max_version:
            self.max_ver = _version_tuple(max_version)
        self.visit(tree)
        return self._issues

class StubAnalyzer:
    """Check whether imported packages ship type stubs."""

    def check_stubs(self, imports: List[ImportRecord]) -> StubReport:
        seen: Set[str] = set()
        statuses: List[StubStatus] = []
        for rec in imports:
            top = rec.module.split(".")[0]
            if top in seen or not top:
                continue
            seen.add(top)
            if top in TYPED_PACKAGES:
                statuses.append(StubStatus(package=top, has_stubs=True,
                                           stub_package=f"types-{top}"))
            else:
                statuses.append(StubStatus(package=top, has_stubs=False))
        return StubReport(statuses=statuses)

class CallGraphBuilder(ast.NodeVisitor):
    """Build a call graph from function definitions and call expressions."""

    def __init__(self) -> None:
        self._graph = CallGraph()
        self._scope_stack: List[str] = ["<module>"]

    @property
    def _scope(self) -> str:
        return self._scope_stack[-1]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        qn = f"{self._scope}.{node.name}" if self._scope != "<module>" else node.name
        self._graph.add_node(qn)
        self._scope_stack.append(qn)
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qn = f"{self._scope}.{node.name}" if self._scope != "<module>" else node.name
        self._scope_stack.append(qn)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        callee = ""
        if isinstance(node.func, ast.Name):
            callee = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee = _resolve_attr(node.func) or node.func.attr
        if callee:
            arg_types = [_infer_arg_type(a) for a in node.args]
            self._graph.add_edge(CallEdge(
                caller=self._scope, callee=callee,
                lineno=node.lineno, arg_types=arg_types))
        self.generic_visit(node)

    def build(self, tree: ast.AST) -> CallGraph:
        self.visit(tree)
        return self._graph

class APISurfaceExtractor(ast.NodeVisitor):
    """Compute the public API surface of a module."""

    def __init__(self) -> None:
        self._functions: List[FunctionInfo] = []
        self._classes: List[ClassInfo] = []
        self._constants: List[str] = []
        self._class_stack: List[ClassInfo] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        info = FunctionInfo(
            name=node.name, qualname=node.name, lineno=node.lineno,
            params=_param_names(node.args),
            return_annotation=_return_annotation_str(node),
            is_method=bool(self._class_stack),
            is_private=node.name.startswith("_"),
            decorators=_extract_decorators(node))
        if self._class_stack:
            info.qualname = f"{self._class_stack[-1].name}.{node.name}"
            self._class_stack[-1].methods.append(info)
        else:
            self._functions.append(info)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases: List[str] = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(_resolve_attr(b) or b.attr)
        ci = ClassInfo(name=node.name, lineno=node.lineno,
                       bases=bases, is_private=node.name.startswith("_"))
        self._class_stack.append(ci)
        self.generic_visit(node)
        self._class_stack.pop()
        self._classes.append(ci)

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self._class_stack:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    self._constants.append(target.id)
        self.generic_visit(node)

    def compute_surface(self, tree: ast.AST) -> APISurface:
        self.visit(tree)
        return APISurface(functions=self._functions,
                          classes=self._classes, constants=self._constants)

class BreakingChangeDetector:
    """Compare two ASTs and report breaking API changes."""

    def detect_breaking_changes(self, old_tree: ast.AST,
                                new_tree: ast.AST) -> List[BreakingChange]:
        old_s = APISurfaceExtractor().compute_surface(old_tree)
        new_s = APISurfaceExtractor().compute_surface(new_tree)
        changes: List[BreakingChange] = []
        self._cmp_functions(old_s.functions, new_s.functions, changes)
        self._cmp_classes(old_s.classes, new_s.classes, changes)
        return changes

    def _cmp_functions(self, old_fns: List[FunctionInfo],
                       new_fns: List[FunctionInfo],
                       out: List[BreakingChange]) -> None:
        old_map = {f.qualname: f for f in old_fns if not f.is_private}
        new_map = {f.qualname: f for f in new_fns if not f.is_private}
        for name, of in old_map.items():
            if name not in new_map:
                out.append(BreakingChange(ChangeKind.REMOVED_FUNCTION, name,
                                          f"Public function '{name}' removed",
                                          old_lineno=of.lineno))
            else:
                self._cmp_sig(of, new_map[name], out)

    def _cmp_sig(self, old_f: FunctionInfo, new_f: FunctionInfo,
                 out: List[BreakingChange]) -> None:
        old_p = {p for p in old_f.params if not p.startswith("self")}
        new_p = {p for p in new_f.params if not p.startswith("self")}
        for p in old_p - new_p:
            if not p.startswith("*"):
                out.append(BreakingChange(ChangeKind.REMOVED_PARAMETER,
                    old_f.qualname, f"Parameter '{p}' removed from '{old_f.qualname}'",
                    old_lineno=old_f.lineno, new_lineno=new_f.lineno))
        for p in new_p - old_p:
            if not p.startswith("*"):
                out.append(BreakingChange(ChangeKind.ADDED_REQUIRED_PARAMETER,
                    new_f.qualname, f"New parameter '{p}' in '{new_f.qualname}'",
                    old_lineno=old_f.lineno, new_lineno=new_f.lineno))
        if (old_f.return_annotation and new_f.return_annotation
                and old_f.return_annotation != new_f.return_annotation):
            out.append(BreakingChange(ChangeKind.CHANGED_RETURN_TYPE,
                old_f.qualname, f"Return type of '{old_f.qualname}' changed",
                old_lineno=old_f.lineno, new_lineno=new_f.lineno))

    def _cmp_classes(self, old_cls: List[ClassInfo],
                     new_cls: List[ClassInfo],
                     out: List[BreakingChange]) -> None:
        old_map = {c.name: c for c in old_cls if not c.is_private}
        new_map = {c.name: c for c in new_cls if not c.is_private}
        for name, oc in old_map.items():
            if name not in new_map:
                out.append(BreakingChange(ChangeKind.REMOVED_CLASS, name,
                    f"Public class '{name}' removed", old_lineno=oc.lineno))
                continue
            nc = new_map[name]
            old_meths = {m.name for m in oc.methods if not m.is_private}
            new_meths = {m.name for m in nc.methods if not m.is_private}
            for rm in old_meths - new_meths:
                out.append(BreakingChange(ChangeKind.REMOVED_METHOD,
                    f"{name}.{rm}", f"Method '{rm}' removed from '{name}'",
                    old_lineno=oc.lineno))
            old_m_map = {m.name: m for m in oc.methods if not m.is_private}
            new_m_map = {m.name: m for m in nc.methods if not m.is_private}
            for shared in old_meths & new_meths:
                self._cmp_sig(old_m_map[shared], new_m_map[shared], out)

class DocCoverageChecker(ast.NodeVisitor):
    """Check which public APIs have docstrings."""

    def __init__(self) -> None:
        self._documented: List[str] = []
        self._undocumented: List[str] = []
        self._class_stack: List[str] = []

    def _qn(self, name: str) -> str:
        return f"{self._class_stack[-1]}.{name}" if self._class_stack else name

    @staticmethod
    def _has_docstring(node: Union[ast.FunctionDef, ast.ClassDef]) -> bool:
        if not node.body:
            return False
        first = node.body[0]
        return (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not node.name.startswith("_") or node.name == "__init__":
            qn = self._qn(node.name)
            (self._documented if self._has_docstring(node)
             else self._undocumented).append(qn)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if not node.name.startswith("_"):
            (self._documented if self._has_docstring(node)
             else self._undocumented).append(node.name)
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def check_docs(self, tree: ast.AST) -> DocCoverageReport:
        self.visit(tree)
        return DocCoverageReport(documented=self._documented,
                                 undocumented=self._undocumented)
class APIAnalyzer:
    """Unified entry point running all sub-analyzers on source code."""
    def __init__(self, min_version: str = "3.7", max_version: str = "3.12") -> None:
        self.min_version = min_version
        self.max_version = max_version
    def analyze(self, source_code: str) -> APIUsageReport:
        tree = ast.parse(textwrap.dedent(source_code))
        imports = ImportAnalyzer().analyze_imports(copy.deepcopy(tree))
        deprecated = DeprecatedAPIChecker().check_deprecated(copy.deepcopy(tree))
        compat = CompatibilityChecker(
            self.min_version, self.max_version,
        ).check_compatibility(copy.deepcopy(tree))
        stubs = StubAnalyzer().check_stubs(imports.all_imports)
        cg = CallGraphBuilder().build(copy.deepcopy(tree))
        surface = APISurfaceExtractor().compute_surface(copy.deepcopy(tree))
        docs = DocCoverageChecker().check_docs(copy.deepcopy(tree))
        return APIUsageReport(imports=imports, deprecated=deprecated,
                              compatibility=compat, stubs=stubs,
                              call_graph=cg, surface=surface,
                              doc_coverage=docs)
    def analyze_breaking_changes(self, old_source: str,
                                 new_source: str) -> List[BreakingChange]:
        old_tree = ast.parse(textwrap.dedent(old_source))
        new_tree = ast.parse(textwrap.dedent(new_source))
        return BreakingChangeDetector().detect_breaking_changes(old_tree, new_tree)
