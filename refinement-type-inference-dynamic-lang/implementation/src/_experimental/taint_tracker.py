"""
Taint analysis for Python: track data from untrusted sources to
dangerous sinks, with propagation through string operations and
interprocedural flow.
"""

import ast
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class VulnerabilityType(Enum):
    SQL_INJECTION = "sql-injection"
    COMMAND_INJECTION = "command-injection"
    PATH_TRAVERSAL = "path-traversal"
    XSS = "xss"
    SSRF = "ssrf"
    CODE_INJECTION = "code-injection"
    LOG_INJECTION = "log-injection"
    OPEN_REDIRECT = "open-redirect"


class TaintSeverity(Enum):
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()


@dataclass
class CodeLocation:
    file: str = "<string>"
    line: int = 0
    col: int = 0
    snippet: str = ""

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


@dataclass
class TaintFlow:
    source: CodeLocation
    sink: CodeLocation
    path: List[CodeLocation]
    vulnerability_type: VulnerabilityType
    severity: TaintSeverity
    description: str = ""
    source_kind: str = ""
    sink_kind: str = ""

    def __str__(self) -> str:
        return (f"[{self.severity.name}] {self.vulnerability_type.value}: "
                f"{self.source} -> {self.sink} ({self.description})")


# ---------------------------------------------------------------------------
# Default source / sink / sanitizer definitions
# ---------------------------------------------------------------------------

# (module_or_object, attribute_or_func) -> description
DEFAULT_SOURCES: Dict[Tuple[str, str], str] = {
    ("", "input"): "user-input",
    ("sys", "argv"): "command-line-args",
    ("os", "environ"): "environment-variables",
    ("os.environ", "get"): "environment-variables",
    ("request", "args"): "http-query-params",
    ("request", "form"): "http-form-data",
    ("request", "data"): "http-body",
    ("request", "json"): "http-json-body",
    ("request", "headers"): "http-headers",
    ("request", "cookies"): "http-cookies",
    ("request", "files"): "http-files",
    ("request", "values"): "http-values",
    ("request", "url"): "http-url",
    ("request", "path"): "http-path",
    ("", "raw_input"): "user-input",
}

DEFAULT_SINKS: Dict[Tuple[str, str], VulnerabilityType] = {
    # SQL injection
    ("cursor", "execute"): VulnerabilityType.SQL_INJECTION,
    ("db", "execute"): VulnerabilityType.SQL_INJECTION,
    ("connection", "execute"): VulnerabilityType.SQL_INJECTION,
    ("conn", "execute"): VulnerabilityType.SQL_INJECTION,
    ("session", "execute"): VulnerabilityType.SQL_INJECTION,
    # Command injection
    ("os", "system"): VulnerabilityType.COMMAND_INJECTION,
    ("os", "popen"): VulnerabilityType.COMMAND_INJECTION,
    ("subprocess", "call"): VulnerabilityType.COMMAND_INJECTION,
    ("subprocess", "run"): VulnerabilityType.COMMAND_INJECTION,
    ("subprocess", "Popen"): VulnerabilityType.COMMAND_INJECTION,
    ("", "exec"): VulnerabilityType.CODE_INJECTION,
    ("", "eval"): VulnerabilityType.CODE_INJECTION,
    ("", "compile"): VulnerabilityType.CODE_INJECTION,
    # Path traversal
    ("", "open"): VulnerabilityType.PATH_TRAVERSAL,
    ("os", "path"): VulnerabilityType.PATH_TRAVERSAL,
    ("os", "remove"): VulnerabilityType.PATH_TRAVERSAL,
    ("os", "unlink"): VulnerabilityType.PATH_TRAVERSAL,
    ("shutil", "copy"): VulnerabilityType.PATH_TRAVERSAL,
    ("shutil", "move"): VulnerabilityType.PATH_TRAVERSAL,
    # XSS
    ("", "render_template_string"): VulnerabilityType.XSS,
    ("", "Markup"): VulnerabilityType.XSS,
    ("response", "write"): VulnerabilityType.XSS,
    # SSRF
    ("requests", "get"): VulnerabilityType.SSRF,
    ("requests", "post"): VulnerabilityType.SSRF,
    ("requests", "put"): VulnerabilityType.SSRF,
    ("requests", "delete"): VulnerabilityType.SSRF,
    ("urllib", "urlopen"): VulnerabilityType.SSRF,
    ("urllib.request", "urlopen"): VulnerabilityType.SSRF,
    ("http.client", "HTTPConnection"): VulnerabilityType.SSRF,
}

SANITIZERS: Set[str] = {
    "escape", "html_escape", "quote", "urlencode", "bleach.clean",
    "markupsafe.escape", "cgi.escape", "html.escape",
    "shlex.quote", "pipes.quote",
    "sanitize", "validate", "clean", "safe",
    "parameterize", "quote_identifier",
    "int", "float", "bool",
}

SEVERITY_MAP: Dict[VulnerabilityType, TaintSeverity] = {
    VulnerabilityType.SQL_INJECTION: TaintSeverity.CRITICAL,
    VulnerabilityType.COMMAND_INJECTION: TaintSeverity.CRITICAL,
    VulnerabilityType.CODE_INJECTION: TaintSeverity.CRITICAL,
    VulnerabilityType.PATH_TRAVERSAL: TaintSeverity.HIGH,
    VulnerabilityType.XSS: TaintSeverity.HIGH,
    VulnerabilityType.SSRF: TaintSeverity.HIGH,
    VulnerabilityType.LOG_INJECTION: TaintSeverity.MEDIUM,
    VulnerabilityType.OPEN_REDIRECT: TaintSeverity.MEDIUM,
}


# ---------------------------------------------------------------------------
# Taint state
# ---------------------------------------------------------------------------

@dataclass
class TaintInfo:
    is_tainted: bool = False
    source_location: Optional[CodeLocation] = None
    source_kind: str = ""
    path: List[CodeLocation] = field(default_factory=list)
    sanitized: bool = False


class TaintState:
    """Tracks taint status for all variables in scope."""

    def __init__(self) -> None:
        self.vars: Dict[str, TaintInfo] = {}
        self.attrs: Dict[str, TaintInfo] = {}  # "obj.attr" -> taint

    def set_tainted(self, name: str, loc: CodeLocation, kind: str = "") -> None:
        info = TaintInfo(
            is_tainted=True,
            source_location=loc,
            source_kind=kind,
            path=[loc],
        )
        self.vars[name] = info

    def set_clean(self, name: str) -> None:
        self.vars[name] = TaintInfo(is_tainted=False)

    def is_tainted(self, name: str) -> bool:
        info = self.vars.get(name)
        return info is not None and info.is_tainted and not info.sanitized

    def get_info(self, name: str) -> Optional[TaintInfo]:
        return self.vars.get(name)

    def propagate(self, from_name: str, to_name: str, loc: CodeLocation) -> None:
        src = self.vars.get(from_name)
        if src and src.is_tainted and not src.sanitized:
            new_info = TaintInfo(
                is_tainted=True,
                source_location=src.source_location,
                source_kind=src.source_kind,
                path=src.path + [loc],
            )
            self.vars[to_name] = new_info

    def copy(self) -> "TaintState":
        s = TaintState()
        for k, v in self.vars.items():
            s.vars[k] = TaintInfo(
                is_tainted=v.is_tainted,
                source_location=v.source_location,
                source_kind=v.source_kind,
                path=list(v.path),
                sanitized=v.sanitized,
            )
        for k, v in self.attrs.items():
            s.attrs[k] = TaintInfo(
                is_tainted=v.is_tainted,
                source_location=v.source_location,
                source_kind=v.source_kind,
                path=list(v.path),
                sanitized=v.sanitized,
            )
        return s


# ---------------------------------------------------------------------------
# Function summaries (interprocedural)
# ---------------------------------------------------------------------------

@dataclass
class FunctionSummary:
    name: str
    tainted_params: Set[int] = field(default_factory=set)  # indices
    returns_tainted: bool = False
    taints_propagated_from: Set[int] = field(default_factory=set)  # param idx -> return


# ---------------------------------------------------------------------------
# Main taint analyser
# ---------------------------------------------------------------------------

class TaintAnalyzer(ast.NodeVisitor):
    """AST-based intraprocedural taint analyser with simple interprocedural summaries."""

    def __init__(self, filename: str = "<string>",
                 sources: Optional[Dict[Tuple[str, str], str]] = None,
                 sinks: Optional[Dict[Tuple[str, str], VulnerabilityType]] = None) -> None:
        self.filename = filename
        self.sources = sources if sources is not None else dict(DEFAULT_SOURCES)
        self.sinks = sinks if sinks is not None else dict(DEFAULT_SINKS)
        self.state = TaintState()
        self.flows: List[TaintFlow] = []
        self.func_summaries: Dict[str, FunctionSummary] = {}
        self._current_func: Optional[str] = None
        self._return_tainted = False

    def _loc(self, node: ast.AST) -> CodeLocation:
        return CodeLocation(
            file=self.filename,
            line=getattr(node, "lineno", 0),
            col=getattr(node, "col_offset", 0),
        )

    # -- expression taint checking ----------------------------------------

    def _expr_tainted(self, node: ast.expr) -> Optional[TaintInfo]:
        """Return TaintInfo if the expression is tainted, else None."""
        if isinstance(node, ast.Name):
            info = self.state.get_info(node.id)
            if info and info.is_tainted and not info.sanitized:
                return info
        elif isinstance(node, ast.BinOp):
            # String concatenation propagates taint
            left = self._expr_tainted(node.left)
            if left:
                return left
            right = self._expr_tainted(node.right)
            if right:
                return right
        elif isinstance(node, ast.JoinedStr):
            # f-string
            for val in node.values:
                if isinstance(val, ast.FormattedValue):
                    info = self._expr_tainted(val.value)
                    if info:
                        return info
        elif isinstance(node, ast.Call):
            return self._call_tainted(node)
        elif isinstance(node, ast.Subscript):
            return self._expr_tainted(node.value)
        elif isinstance(node, ast.Attribute):
            # Check if the attribute access itself is a source
            obj_name = self._resolve_name(node.value)
            key = (obj_name, node.attr)
            if key in self.sources:
                return TaintInfo(
                    is_tainted=True,
                    source_location=self._loc(node),
                    source_kind=self.sources[key],
                    path=[self._loc(node)],
                )
            return self._expr_tainted(node.value)
        elif isinstance(node, ast.IfExp):
            body = self._expr_tainted(node.body)
            if body:
                return body
            return self._expr_tainted(node.orelse)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.elts:
                info = self._expr_tainted(elt)
                if info:
                    return info
        elif isinstance(node, ast.Dict):
            for v in node.values:
                info = self._expr_tainted(v)
                if info:
                    return info
        return None

    def _call_tainted(self, node: ast.Call) -> Optional[TaintInfo]:
        """Check if a call returns tainted data (source) or propagates taint."""
        func_name, obj_name = self._resolve_call(node)

        # Check if this is a sanitizer
        if func_name in SANITIZERS or f"{obj_name}.{func_name}" in SANITIZERS:
            return None

        # Check if it is a source
        key = (obj_name, func_name)
        if key in self.sources:
            return TaintInfo(
                is_tainted=True,
                source_location=self._loc(node),
                source_kind=self.sources[key],
                path=[self._loc(node)],
            )
        # Bare function source
        bare_key = ("", func_name)
        if bare_key in self.sources:
            return TaintInfo(
                is_tainted=True,
                source_location=self._loc(node),
                source_kind=self.sources[bare_key],
                path=[self._loc(node)],
            )

        # String methods propagate taint
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            obj_info = self._expr_tainted(node.func.value)
            if obj_info and attr in (
                "format", "replace", "join", "strip", "lstrip", "rstrip",
                "lower", "upper", "title", "capitalize", "encode", "decode",
                "split", "rsplit",
            ):
                return obj_info

        # str.format with tainted args
        if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
            for arg in node.args:
                info = self._expr_tainted(arg)
                if info:
                    return info

        # Check interprocedural summary
        summary = self.func_summaries.get(func_name)
        if summary and summary.returns_tainted:
            for idx in summary.taints_propagated_from:
                if idx < len(node.args):
                    info = self._expr_tainted(node.args[idx])
                    if info:
                        return info

        # If any argument is tainted and func is not a known sanitizer, propagate
        for arg in node.args:
            info = self._expr_tainted(arg)
            if info:
                # Only propagate for string operations or unknown functions
                return info

        return None

    def _check_sink(self, node: ast.Call) -> None:
        """Check if a call is a sink and any argument is tainted."""
        func_name, obj_name = self._resolve_call(node)

        key = (obj_name, func_name)
        bare_key = ("", func_name)

        vuln_type: Optional[VulnerabilityType] = None
        if key in self.sinks:
            vuln_type = self.sinks[key]
        elif bare_key in self.sinks:
            vuln_type = self.sinks[bare_key]

        if vuln_type is None:
            return

        # Check each argument
        for arg in node.args:
            info = self._expr_tainted(arg)
            if info:
                self.flows.append(TaintFlow(
                    source=info.source_location or self._loc(node),
                    sink=self._loc(node),
                    path=info.path + [self._loc(node)],
                    vulnerability_type=vuln_type,
                    severity=SEVERITY_MAP.get(vuln_type, TaintSeverity.MEDIUM),
                    description=f"Tainted data from {info.source_kind} flows to {func_name}",
                    source_kind=info.source_kind,
                    sink_kind=func_name,
                ))
                return  # one finding per call

        # Check keyword arguments
        for kw in node.keywords:
            if kw.value:
                info = self._expr_tainted(kw.value)
                if info:
                    self.flows.append(TaintFlow(
                        source=info.source_location or self._loc(node),
                        sink=self._loc(node),
                        path=info.path + [self._loc(node)],
                        vulnerability_type=vuln_type,
                        severity=SEVERITY_MAP.get(vuln_type, TaintSeverity.MEDIUM),
                        description=f"Tainted data from {info.source_kind} flows to {func_name} via kwarg '{kw.arg}'",
                        source_kind=info.source_kind,
                        sink_kind=func_name,
                    ))
                    return

    # -- statement visitors ------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        info = self._expr_tainted(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if info:
                    self.state.set_tainted(target.id, self._loc(node), info.source_kind)
                    # Extend path
                    t = self.state.get_info(target.id)
                    if t:
                        t.path = info.path + [self._loc(node)]
                else:
                    self.state.set_clean(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        if info:
                            self.state.set_tainted(elt.id, self._loc(node), info.source_kind)
                        else:
                            self.state.set_clean(elt.id)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        info = self._expr_tainted(node.value)
        if isinstance(node.target, ast.Name):
            existing = self.state.get_info(node.target.id)
            if info or (existing and existing.is_tainted):
                src_info = info or existing
                self.state.set_tainted(node.target.id, self._loc(node),
                                       src_info.source_kind if src_info else "")
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Call):
            self._check_sink(node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._check_sink(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value:
            info = self._expr_tainted(node.value)
            if info:
                self._return_tainted = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        old_func = self._current_func
        old_return = self._return_tainted
        self._current_func = node.name
        self._return_tainted = False

        old_state = self.state.copy()
        # Mark parameters as potentially tainted based on their names
        param_tainted: Set[int] = set()
        for i, arg in enumerate(node.args.args):
            # Don't auto-taint, but track which params get used in sinks
            pass

        self.generic_visit(node)

        # Build summary
        summary = FunctionSummary(
            name=node.name,
            returns_tainted=self._return_tainted,
        )
        self.func_summaries[node.name] = summary

        self._current_func = old_func
        self._return_tainted = old_return
        self.state = old_state

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        # If iterating over tainted data, loop var is tainted
        info = self._expr_tainted(node.iter)
        if info and isinstance(node.target, ast.Name):
            self.state.set_tainted(node.target.id, self._loc(node), info.source_kind)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                info = self._expr_tainted(item.context_expr)
                if info:
                    self.state.set_tainted(item.optional_vars.id, self._loc(node), info.source_kind)
        self.generic_visit(node)

    # -- resolution helpers -----------------------------------------------

    def _resolve_name(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = self._resolve_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    def _resolve_call(self, node: ast.Call) -> Tuple[str, str]:
        """Return (func_name, obj_name)."""
        if isinstance(node.func, ast.Name):
            return (node.func.id, "")
        if isinstance(node.func, ast.Attribute):
            obj_name = self._resolve_name(node.func.value)
            return (node.func.attr, obj_name)
        return ("", "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TaintTracker:
    """Track tainted data from sources to sinks in Python source code."""

    def __init__(self) -> None:
        self.custom_sources: Dict[Tuple[str, str], str] = {}
        self.custom_sinks: Dict[Tuple[str, str], VulnerabilityType] = {}

    def add_source(self, obj: str, attr: str, kind: str) -> None:
        self.custom_sources[(obj, attr)] = kind

    def add_sink(self, obj: str, attr: str, vuln: VulnerabilityType) -> None:
        self.custom_sinks[(obj, attr)] = vuln

    def analyze(self, source_code: str,
                sources: Optional[Dict[Tuple[str, str], str]] = None,
                sinks: Optional[Dict[Tuple[str, str], VulnerabilityType]] = None,
                filename: str = "<string>") -> List[TaintFlow]:
        try:
            tree = ast.parse(source_code, filename=filename)
        except SyntaxError:
            return []

        all_sources = dict(DEFAULT_SOURCES)
        all_sources.update(self.custom_sources)
        if sources:
            all_sources.update(sources)

        all_sinks = dict(DEFAULT_SINKS)
        all_sinks.update(self.custom_sinks)
        if sinks:
            all_sinks.update(sinks)

        analyzer = TaintAnalyzer(
            filename=filename,
            sources=all_sources,
            sinks=all_sinks,
        )
        analyzer.visit(tree)
        return analyzer.flows


# ---------------------------------------------------------------------------
# Convenience: analyse a file
# ---------------------------------------------------------------------------

def analyze_file(path: str) -> List[TaintFlow]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()
    tracker = TaintTracker()
    return tracker.analyze(source, filename=path)
