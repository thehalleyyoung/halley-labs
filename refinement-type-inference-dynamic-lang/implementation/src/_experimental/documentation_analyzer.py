"""Documentation analysis and generation for Python projects.

Provides AST-based documentation quality analysis:
- Documentation coverage measurement (% of functions/classes documented)
- Stale documentation detection (docs that don't match code)
- Automatic docstring generation from code analysis
- Public API surface extraction
- Markdown API documentation generation
"""
from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class DocStyle(Enum):
    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"
    NONE = "none"


@dataclass
class UndocumentedItem:
    kind: str  # "function", "class", "method"
    name: str
    file: str
    line: int
    params: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.file}:{self.line} {self.kind} '{self.name}'"


@dataclass
class DocCoverage:
    total_functions: int = 0
    documented_functions: int = 0
    total_classes: int = 0
    documented_classes: int = 0
    total_modules: int = 0
    documented_modules: int = 0
    coverage_pct: float = 0.0
    undocumented: List[UndocumentedItem] = field(default_factory=list)
    summary: str = ""

    def __str__(self) -> str:
        return (
            f"Doc coverage: {self.coverage_pct:.1f}%\n"
            f"  Functions: {self.documented_functions}/{self.total_functions}\n"
            f"  Classes: {self.documented_classes}/{self.total_classes}\n"
            f"  Modules: {self.documented_modules}/{self.total_modules}\n"
            f"  Undocumented: {len(self.undocumented)}"
        )


@dataclass
class StaleDoc:
    file: str
    line: int
    name: str
    reason: str
    details: str = ""

    def __str__(self) -> str:
        return f"{self.file}:{self.line} '{self.name}': {self.reason}"


@dataclass
class ParamInfo:
    name: str
    annotation: Optional[str] = None
    default: Optional[str] = None
    doc: str = ""

    def __str__(self) -> str:
        ann = f": {self.annotation}" if self.annotation else ""
        default = f" = {self.default}" if self.default else ""
        return f"{self.name}{ann}{default}"


@dataclass
class FunctionInfo:
    name: str
    file: str
    line: int
    params: List[ParamInfo] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    is_public: bool = True

    def __str__(self) -> str:
        async_prefix = "async " if self.is_async else ""
        params = ", ".join(str(p) for p in self.params)
        ret = f" -> {self.return_type}" if self.return_type else ""
        return f"{async_prefix}def {self.name}({params}){ret}"


@dataclass
class ClassInfo:
    name: str
    file: str
    line: int
    methods: List[FunctionInfo] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_public: bool = True

    def __str__(self) -> str:
        bases = f"({', '.join(self.bases)})" if self.bases else ""
        return f"class {self.name}{bases}"


@dataclass
class APISurface:
    modules: List[str] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    summary: str = ""

    def __str__(self) -> str:
        return (
            f"API Surface: {len(self.modules)} modules, "
            f"{len(self.classes)} classes, "
            f"{len(self.functions)} functions"
        )


# ---------------------------------------------------------------------------
# AST Helpers
# ---------------------------------------------------------------------------

def _collect_python_files(project_dir: str) -> List[Path]:
    root = Path(project_dir)
    skip = {"__pycache__", "node_modules", ".git", "venv", "env", ".venv", ".env", ".tox"}
    result: List[Path] = []
    for p in root.rglob("*.py"):
        if not any(part in skip or part.startswith(".") for part in p.relative_to(root).parts):
            result.append(p)
    return sorted(result)


def _parse_file(filepath: Path) -> Optional[ast.Module]:
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        return ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return None


def _get_docstring(node: ast.AST) -> Optional[str]:
    """Extract docstring from a function/class/module node."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            return node.body[0].value.value
    return None


def _get_annotation_str(node: Optional[ast.expr]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _extract_params(func_node: ast.FunctionDef) -> List[ParamInfo]:
    """Extract parameter info from a function definition."""
    params: List[ParamInfo] = []
    args = func_node.args

    all_args = args.posonlyargs + args.args + args.kwonlyargs
    defaults = args.defaults
    kw_defaults = args.kw_defaults

    # Calculate default offsets for positional args
    n_positional = len(args.posonlyargs) + len(args.args)
    default_offset = n_positional - len(defaults)

    for i, arg in enumerate(args.posonlyargs + args.args):
        if arg.arg in ("self", "cls"):
            continue
        default_idx = i - default_offset
        default_val = None
        if default_idx >= 0 and default_idx < len(defaults):
            try:
                default_val = ast.unparse(defaults[default_idx])
            except Exception:
                pass
        params.append(ParamInfo(
            name=arg.arg,
            annotation=_get_annotation_str(arg.annotation),
            default=default_val,
        ))

    for i, arg in enumerate(args.kwonlyargs):
        default_val = None
        if i < len(kw_defaults) and kw_defaults[i] is not None:
            try:
                default_val = ast.unparse(kw_defaults[i])
            except Exception:
                pass
        params.append(ParamInfo(
            name=arg.arg,
            annotation=_get_annotation_str(arg.annotation),
            default=default_val,
        ))

    if args.vararg:
        params.append(ParamInfo(
            name=f"*{args.vararg.arg}",
            annotation=_get_annotation_str(args.vararg.annotation),
        ))
    if args.kwarg:
        params.append(ParamInfo(
            name=f"**{args.kwarg.arg}",
            annotation=_get_annotation_str(args.kwarg.annotation),
        ))

    return params


def _extract_function_info(node: ast.FunctionDef, filepath: str) -> FunctionInfo:
    decorators: List[str] = []
    for dec in node.decorator_list:
        try:
            decorators.append(ast.unparse(dec))
        except Exception:
            pass

    return FunctionInfo(
        name=node.name,
        file=filepath,
        line=node.lineno,
        params=_extract_params(node),
        return_type=_get_annotation_str(node.returns),
        docstring=_get_docstring(node),
        is_async=isinstance(node, ast.AsyncFunctionDef),
        decorators=decorators,
        is_public=not node.name.startswith("_"),
    )


def _extract_class_info(node: ast.ClassDef, filepath: str) -> ClassInfo:
    bases: List[str] = []
    for base in node.bases:
        try:
            bases.append(ast.unparse(base))
        except Exception:
            pass

    methods: List[FunctionInfo] = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(_extract_function_info(item, filepath))

    return ClassInfo(
        name=node.name,
        file=filepath,
        line=node.lineno,
        methods=methods,
        bases=bases,
        docstring=_get_docstring(node),
        is_public=not node.name.startswith("_"),
    )


# ---------------------------------------------------------------------------
# Docstring parameter parsing
# ---------------------------------------------------------------------------

_GOOGLE_PARAM_RE = re.compile(r"^\s+(\w+)\s*(?:\(.*?\))?:\s*(.*)$")
_SPHINX_PARAM_RE = re.compile(r":param\s+(\w+):\s*(.*)")
_NUMPY_PARAM_RE = re.compile(r"^(\w+)\s*:\s*(.*)$")


def _parse_docstring_params(docstring: str) -> Set[str]:
    """Extract parameter names documented in a docstring."""
    params: Set[str] = set()
    for pattern in (_GOOGLE_PARAM_RE, _SPHINX_PARAM_RE, _NUMPY_PARAM_RE):
        for m in pattern.finditer(docstring):
            params.add(m.group(1))
    return params


# ---------------------------------------------------------------------------
# Public API: doc_coverage
# ---------------------------------------------------------------------------

def doc_coverage(project_dir: str) -> DocCoverage:
    """Measure documentation coverage of a Python project.

    Counts functions, classes, and modules that have docstrings.
    Returns coverage percentage and list of undocumented items.
    """
    files = _collect_python_files(project_dir)
    cov = DocCoverage()

    for fp in files:
        tree = _parse_file(fp)
        if tree is None:
            continue
        rel = str(fp.relative_to(Path(project_dir)))

        # Module docstring
        cov.total_modules += 1
        if _get_docstring(tree):
            cov.documented_modules += 1

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_") and node.name != "__init__":
                    continue
                cov.total_functions += 1
                if _get_docstring(node):
                    cov.documented_functions += 1
                else:
                    param_names = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
                    cov.undocumented.append(UndocumentedItem(
                        kind="function",
                        name=node.name,
                        file=rel,
                        line=node.lineno,
                        params=param_names,
                    ))

            elif isinstance(node, ast.ClassDef):
                if node.name.startswith("_"):
                    continue
                cov.total_classes += 1
                if _get_docstring(node):
                    cov.documented_classes += 1
                else:
                    cov.undocumented.append(UndocumentedItem(
                        kind="class",
                        name=node.name,
                        file=rel,
                        line=node.lineno,
                    ))

    total = cov.total_functions + cov.total_classes + cov.total_modules
    documented = cov.documented_functions + cov.documented_classes + cov.documented_modules
    cov.coverage_pct = (documented / max(total, 1)) * 100
    cov.summary = f"{documented}/{total} items documented ({cov.coverage_pct:.1f}%)"
    return cov


# ---------------------------------------------------------------------------
# Public API: find_stale_docs
# ---------------------------------------------------------------------------

def find_stale_docs(project_dir: str) -> List[StaleDoc]:
    """Find documentation that doesn't match the actual code.

    Detects:
    - Docstrings mentioning parameters that don't exist
    - Missing parameter documentation for existing parameters
    - Return type mismatches between docstring and annotation
    """
    files = _collect_python_files(project_dir)
    stale: List[StaleDoc] = []

    for fp in files:
        tree = _parse_file(fp)
        if tree is None:
            continue
        rel = str(fp.relative_to(Path(project_dir)))

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            docstring = _get_docstring(node)
            if not docstring:
                continue

            # Get actual params
            actual_params = {
                a.arg for a in node.args.args + node.args.posonlyargs + node.args.kwonlyargs
                if a.arg not in ("self", "cls")
            }

            # Get documented params
            doc_params = _parse_docstring_params(docstring)

            # Check for documented params that don't exist
            phantom = doc_params - actual_params
            if phantom:
                stale.append(StaleDoc(
                    file=rel,
                    line=node.lineno,
                    name=node.name,
                    reason=f"Documents nonexistent params: {', '.join(sorted(phantom))}",
                ))

            # Check for undocumented params (if some are documented, all should be)
            if doc_params:
                missing = actual_params - doc_params
                if missing:
                    stale.append(StaleDoc(
                        file=rel,
                        line=node.lineno,
                        name=node.name,
                        reason=f"Missing docs for params: {', '.join(sorted(missing))}",
                    ))

            # Check return type consistency
            if node.returns is not None:
                ret_ann = _get_annotation_str(node.returns)
                if ret_ann and "return" in docstring.lower():
                    # Simple check: if annotation says None but doc describes a return
                    if ret_ann == "None" and re.search(r"[Rr]eturns?\s*:", docstring):
                        doc_return = re.search(r"[Rr]eturns?\s*:\s*(.+)", docstring)
                        if doc_return and "none" not in doc_return.group(1).lower():
                            stale.append(StaleDoc(
                                file=rel,
                                line=node.lineno,
                                name=node.name,
                                reason=f"Return annotation is None but docstring describes return value",
                            ))

    return stale


# ---------------------------------------------------------------------------
# Public API: generate_docstrings
# ---------------------------------------------------------------------------

def _infer_param_description(name: str, annotation: Optional[str]) -> str:
    """Generate a placeholder description based on parameter name and type."""
    if annotation:
        return f"The {name.replace('_', ' ')} ({annotation})."
    return f"The {name.replace('_', ' ')}."


def _infer_return_description(node: ast.FunctionDef) -> str:
    """Infer what a function returns from its body."""
    ret_ann = _get_annotation_str(node.returns)
    if ret_ann:
        if ret_ann == "None":
            return ""
        if ret_ann == "bool":
            return "True if the operation succeeds, False otherwise."
        if ret_ann == "str":
            return "The resulting string."
        return f"The {ret_ann} result."

    # Check return statements
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            if isinstance(child.value, ast.Constant):
                if isinstance(child.value.value, bool):
                    return "True if the condition is met, False otherwise."
                if isinstance(child.value.value, str):
                    return "The resulting string."
            return "The computed result."
    return ""


def _generate_function_docstring(node: ast.FunctionDef) -> str:
    """Generate a Google-style docstring for a function."""
    parts: List[str] = []

    # Summary line from function name
    name = node.name.replace("_", " ").strip()
    if name.startswith("get "):
        summary = f"Get {name[4:]}."
    elif name.startswith("set "):
        summary = f"Set {name[4:]}."
    elif name.startswith("is ") or name.startswith("has "):
        summary = f"Check if {name[3:]}."
    elif name.startswith("find "):
        summary = f"Find {name[5:]}."
    elif name.startswith("create "):
        summary = f"Create {name[7:]}."
    elif name.startswith("delete ") or name.startswith("remove "):
        summary = f"Remove {name[7:]}."
    elif name.startswith("update "):
        summary = f"Update {name[7:]}."
    elif name.startswith("validate "):
        summary = f"Validate {name[9:]}."
    elif name.startswith("parse "):
        summary = f"Parse {name[6:]}."
    elif name.startswith("convert "):
        summary = f"Convert {name[8:]}."
    else:
        summary = f"{name.capitalize()}."

    parts.append(summary)

    # Parameters
    params = _extract_params(node)
    if params:
        parts.append("")
        parts.append("Args:")
        for p in params:
            if p.name.startswith("*"):
                continue
            desc = _infer_param_description(p.name, p.annotation)
            default = f" Defaults to {p.default}." if p.default else ""
            parts.append(f"    {p.name}: {desc}{default}")

    # Returns
    ret_desc = _infer_return_description(node)
    if ret_desc:
        parts.append("")
        parts.append("Returns:")
        parts.append(f"    {ret_desc}")

    # Raises
    raises: List[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and child.exc is not None:
            exc_name = ""
            if isinstance(child.exc, ast.Call):
                exc_name = _get_call_name_simple(child.exc)
            elif isinstance(child.exc, ast.Name):
                exc_name = child.exc.id
            if exc_name and exc_name not in raises:
                raises.append(exc_name)

    if raises:
        parts.append("")
        parts.append("Raises:")
        for r in raises:
            parts.append(f"    {r}: If an error occurs.")

    return "\n".join(parts)


def _get_call_name_simple(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def generate_docstrings(source: str) -> str:
    """Auto-generate docstrings for undocumented functions and classes.

    Uses Google-style docstring format. Only adds docstrings where none exist.
    """
    tree = ast.parse(source)

    class _DocstringInserter(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            self.generic_visit(node)
            if _get_docstring(node) is not None:
                return node
            if node.name.startswith("_") and node.name != "__init__":
                return node

            doc = _generate_function_docstring(node)
            doc_node = ast.Expr(value=ast.Constant(value=doc))
            ast.copy_location(doc_node, node)
            ast.fix_missing_locations(doc_node)
            node.body.insert(0, doc_node)
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
            self.generic_visit(node)
            if _get_docstring(node) is not None:
                return node
            if node.name.startswith("_"):
                return node

            doc = f"{node.name} class."
            doc_node = ast.Expr(value=ast.Constant(value=doc))
            ast.copy_location(doc_node, node)
            ast.fix_missing_locations(doc_node)
            node.body.insert(0, doc_node)
            return node

    inserter = _DocstringInserter()
    new_tree = inserter.visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        return ast.unparse(new_tree)
    except Exception:
        return source


# ---------------------------------------------------------------------------
# Public API: api_surface
# ---------------------------------------------------------------------------

def api_surface(project_dir: str) -> APISurface:
    """Extract the public API surface of a Python project.

    Returns all public modules, classes, functions, and module-level constants.
    """
    files = _collect_python_files(project_dir)
    surface = APISurface()

    for fp in files:
        tree = _parse_file(fp)
        if tree is None:
            continue
        rel = str(fp.relative_to(Path(project_dir)))

        # Check __all__ for explicit exports
        all_names: Optional[Set[str]] = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            all_names = set()
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    all_names.add(elt.value)

        surface.modules.append(rel)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if all_names is not None:
                    if node.name not in all_names:
                        continue
                elif node.name.startswith("_"):
                    continue
                surface.functions.append(_extract_function_info(node, rel))

            elif isinstance(node, ast.ClassDef):
                if all_names is not None:
                    if node.name not in all_names:
                        continue
                elif node.name.startswith("_"):
                    continue
                surface.classes.append(_extract_class_info(node, rel))

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        surface.constants.append(f"{rel}:{target.id}")

    surface.summary = (
        f"{len(surface.modules)} modules, {len(surface.classes)} classes, "
        f"{len(surface.functions)} functions, {len(surface.constants)} constants"
    )
    return surface


# ---------------------------------------------------------------------------
# Public API: generate_api_docs
# ---------------------------------------------------------------------------

def generate_api_docs(project_dir: str) -> str:
    """Generate markdown API documentation for a Python project.

    Produces a complete API reference from the public API surface.
    """
    surface = api_surface(project_dir)
    lines: List[str] = [
        "# API Reference",
        "",
        f"> {surface.summary}",
        "",
    ]

    # Table of contents
    lines.append("## Table of Contents")
    lines.append("")
    for cls in surface.classes:
        lines.append(f"- [{cls.name}](#{cls.name.lower()})")
    for func in surface.functions:
        lines.append(f"- [{func.name}](#{func.name.lower()})")
    lines.append("")

    # Classes
    if surface.classes:
        lines.append("## Classes")
        lines.append("")
        for cls in surface.classes:
            lines.append(f"### {cls.name}")
            lines.append("")
            if cls.bases:
                lines.append(f"*Bases: {', '.join(cls.bases)}*")
                lines.append("")
            if cls.docstring:
                lines.append(cls.docstring.strip())
                lines.append("")
            lines.append(f"*Defined in `{cls.file}:{cls.line}`*")
            lines.append("")

            # Methods
            public_methods = [m for m in cls.methods if not m.name.startswith("_") or m.name == "__init__"]
            if public_methods:
                lines.append("#### Methods")
                lines.append("")
                for method in public_methods:
                    async_prefix = "async " if method.is_async else ""
                    params_str = ", ".join(str(p) for p in method.params)
                    ret = f" -> {method.return_type}" if method.return_type else ""
                    lines.append(f"##### `{async_prefix}{method.name}({params_str}){ret}`")
                    lines.append("")
                    if method.docstring:
                        lines.append(method.docstring.strip())
                        lines.append("")
                    lines.append("---")
                    lines.append("")

    # Functions
    if surface.functions:
        lines.append("## Functions")
        lines.append("")
        for func in surface.functions:
            async_prefix = "async " if func.is_async else ""
            params_str = ", ".join(str(p) for p in func.params)
            ret = f" -> {func.return_type}" if func.return_type else ""
            lines.append(f"### `{async_prefix}{func.name}({params_str}){ret}`")
            lines.append("")
            if func.decorators:
                lines.append(f"*Decorators: {', '.join(func.decorators)}*")
                lines.append("")
            if func.docstring:
                lines.append(func.docstring.strip())
                lines.append("")
            lines.append(f"*Defined in `{func.file}:{func.line}`*")
            lines.append("")
            lines.append("---")
            lines.append("")

    # Constants
    if surface.constants:
        lines.append("## Constants")
        lines.append("")
        for const in surface.constants:
            lines.append(f"- `{const}`")
        lines.append("")

    return "\n".join(lines)
