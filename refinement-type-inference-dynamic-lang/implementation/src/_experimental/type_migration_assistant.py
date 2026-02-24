"""Type migration assistant for Python codebases.
Helps migrate untyped Python code to fully typed code by inferring types,
inserting annotations, generating stub files, prioritizing files for
migration, and tracking migration progress.
"""
from __future__ import annotations
import ast
import copy
import re
import textwrap
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
import numpy as np

class TypeKind(Enum):
    SIMPLE = auto()
    OPTIONAL = auto()
    UNION = auto()
    LIST = auto()
    DICT = auto()
    SET = auto()
    TUPLE = auto()
    NONE = auto()
    UNKNOWN = auto()
@dataclass(frozen=True)
class InferredType:
    """A type inferred from source-code usage patterns."""
    kind: TypeKind
    name: str = "Unknown"
    children: Tuple[InferredType, ...] = ()

    def to_annotation(self) -> str:
        if self.kind == TypeKind.NONE:
            return "None"
        if self.kind == TypeKind.UNKNOWN:
            return "Any"
        if self.kind == TypeKind.SIMPLE:
            return self.name
        if self.kind == TypeKind.OPTIONAL:
            inner = self.children[0].to_annotation() if self.children else "Any"
            return f"Optional[{inner}]"
        if self.kind == TypeKind.UNION:
            return f"Union[{', '.join(c.to_annotation() for c in self.children)}]"
        if self.kind == TypeKind.LIST:
            inner = self.children[0].to_annotation() if self.children else "Any"
            return f"List[{inner}]"
        if self.kind == TypeKind.DICT:
            if len(self.children) >= 2:
                return f"Dict[{self.children[0].to_annotation()}, {self.children[1].to_annotation()}]"
            return "Dict[Any, Any]"
        if self.kind == TypeKind.SET:
            inner = self.children[0].to_annotation() if self.children else "Any"
            return f"Set[{inner}]"
        if self.kind == TypeKind.TUPLE:
            parts = [c.to_annotation() for c in self.children] if self.children else ["Any"]
            return f"Tuple[{', '.join(parts)}]"
        return "Any"

    def is_numeric(self) -> bool:
        return self.kind == TypeKind.SIMPLE and self.name in ("int", "float", "bool")
SIMPLE_INT = InferredType(TypeKind.SIMPLE, "int")
SIMPLE_FLOAT = InferredType(TypeKind.SIMPLE, "float")
SIMPLE_STR = InferredType(TypeKind.SIMPLE, "str")
SIMPLE_BOOL = InferredType(TypeKind.SIMPLE, "bool")
SIMPLE_NONE = InferredType(TypeKind.NONE, "None")
SIMPLE_UNKNOWN = InferredType(TypeKind.UNKNOWN)
@dataclass
class PredictedError:
    """An error predicted to occur when strict typing is applied."""
    lineno: int
    col_offset: int
    code: str
    message: str
    severity: str = "error"

    def __str__(self) -> str:
        return f"{self.lineno}:{self.col_offset}: {self.code} - {self.message}"

class FileStatus(Enum):
    UNTYPED = "untyped"
    PARTIAL = "partial"
    FULLY_TYPED = "fully_typed"
@dataclass
class FileState:
    """Tracks the annotation state of a single file."""
    path: str
    status: FileStatus = FileStatus.UNTYPED
    total_functions: int = 0
    annotated_functions: int = 0
    total_variables: int = 0
    annotated_variables: int = 0
    lines_of_code: int = 0
@dataclass
class MigrationPlan:
    """The plan produced by the migration assistant."""
    files_in_order: List[str] = field(default_factory=list)
    estimated_effort: float = 0.0
    phases: List[Dict[str, Any]] = field(default_factory=list)
    strict_mode: bool = False
    py_typed_content: str = ""
@dataclass
class ProgressReport:
    """Current migration progress."""
    percent_annotated: float = 0.0
    functions_done: int = 0
    functions_total: int = 0
    files_done: int = 0
    files_total: int = 0
    estimated_remaining: float = 0.0
    velocity: float = 0.0
@dataclass
class TypeConflict:
    """A conflict detected between inferred types at different sites."""
    variable: str
    location_a: int
    location_b: int
    type_a: InferredType
    type_b: InferredType
    message: str = ""
_METHOD_TYPE_MAP: Dict[str, InferredType] = {
    "append": InferredType(TypeKind.LIST), "extend": InferredType(TypeKind.LIST),
    "insert": InferredType(TypeKind.LIST), "pop": InferredType(TypeKind.LIST),
    "remove": InferredType(TypeKind.LIST), "sort": InferredType(TypeKind.LIST),
    "reverse": InferredType(TypeKind.LIST),
    "add": InferredType(TypeKind.SET), "discard": InferredType(TypeKind.SET),
    "intersection": InferredType(TypeKind.SET), "difference": InferredType(TypeKind.SET),
    "symmetric_difference": InferredType(TypeKind.SET),
    "keys": InferredType(TypeKind.DICT), "values": InferredType(TypeKind.DICT),
    "items": InferredType(TypeKind.DICT), "update": InferredType(TypeKind.DICT),
    "get": InferredType(TypeKind.DICT), "setdefault": InferredType(TypeKind.DICT),
    "upper": SIMPLE_STR, "lower": SIMPLE_STR, "strip": SIMPLE_STR,
    "lstrip": SIMPLE_STR, "rstrip": SIMPLE_STR,
    "split": SIMPLE_STR, "join": SIMPLE_STR, "replace": SIMPLE_STR,
    "startswith": SIMPLE_STR, "endswith": SIMPLE_STR, "encode": SIMPLE_STR,
    "format": SIMPLE_STR, "find": SIMPLE_STR, "count": SIMPLE_STR,
    "title": SIMPLE_STR, "capitalize": SIMPLE_STR, "zfill": SIMPLE_STR,
}

def _merge_types(types: List[InferredType]) -> InferredType:
    """Merge a list of inferred types into a single type."""
    unique = list(dict.fromkeys(t for t in types if t.kind != TypeKind.UNKNOWN))
    if not unique:
        return SIMPLE_UNKNOWN
    if len(unique) == 1:
        return unique[0]
    has_none = any(t.kind == TypeKind.NONE for t in unique)
    non_none = [t for t in unique if t.kind != TypeKind.NONE]
    if has_none and len(non_none) == 1:
        return InferredType(TypeKind.OPTIONAL, children=(non_none[0],))
    if has_none and len(non_none) > 1:
        inner = InferredType(TypeKind.UNION, children=tuple(non_none))
        return InferredType(TypeKind.OPTIONAL, children=(inner,))
    return InferredType(TypeKind.UNION, children=tuple(unique))

class TypeInferrer(ast.NodeVisitor):
    """Infers types from assignment patterns, operations, and method calls."""

    def __init__(self) -> None:
        self._var_types: Dict[str, List[InferredType]] = defaultdict(list)
        self._return_types: Dict[str, List[InferredType]] = defaultdict(list)
        self._current_func: Optional[str] = None

    def infer(self, tree: ast.Module) -> Dict[str, InferredType]:
        """Walk *tree* and return a mapping of name -> InferredType."""
        self._var_types.clear()
        self._return_types.clear()
        self._current_func = None
        self.visit(tree)
        result: Dict[str, InferredType] = {}
        for name, types in self._var_types.items():
            result[name] = _merge_types(types)
        for fname, types in self._return_types.items():
            result[f"{fname}.__return__"] = _merge_types(types)
        return result

    def visit_Assign(self, node: ast.Assign) -> None:
        inferred = self._infer_expr(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._var_types[target.id].append(inferred)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self._var_types[elt.id].append(SIMPLE_UNKNOWN)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value is not None:
            self._var_types[node.target.id].append(self._infer_expr(node.value))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._current_func
        self._current_func = node.name
        for arg in node.args.args:
            self._var_types[arg.arg].append(SIMPLE_UNKNOWN)
        self.generic_visit(node)
        self._current_func = prev
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Return(self, node: ast.Return) -> None:
        if self._current_func is not None:
            if node.value is not None:
                self._return_types[self._current_func].append(self._infer_expr(node.value))
            else:
                self._return_types[self._current_func].append(SIMPLE_NONE)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            method = node.func.attr
            if method in _METHOD_TYPE_MAP:
                self._var_types[node.func.value.id].append(_METHOD_TYPE_MAP[method])
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if isinstance(node.target, ast.Name):
            self._var_types[node.target.id].append(SIMPLE_UNKNOWN)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if isinstance(node.left, ast.Name):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Is, ast.IsNot)):
                    if isinstance(comp, ast.Constant) and comp.value is None:
                        if self._var_types.get(node.left.id):
                            self._var_types[node.left.id].append(SIMPLE_NONE)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            inferred = self._infer_expr(node.value)
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                if inferred.kind == TypeKind.SIMPLE and inferred.name in ("int", "float"):
                    self._var_types[node.target.id].append(inferred)
        self.generic_visit(node)

    def _infer_expr(self, node: ast.expr) -> InferredType:
        if isinstance(node, ast.Constant):
            return self._infer_constant(node.value)
        if isinstance(node, ast.List):
            return self._infer_container(TypeKind.LIST, node.elts)
        if isinstance(node, ast.Set):
            return self._infer_container(TypeKind.SET, node.elts)
        if isinstance(node, ast.Dict):
            kt = [self._infer_expr(k) for k in node.keys if k is not None]
            vt = [self._infer_expr(v) for v in node.values]
            key_type = _merge_types(kt) if kt else SIMPLE_UNKNOWN
            val_type = _merge_types(vt) if vt else SIMPLE_UNKNOWN
            return InferredType(TypeKind.DICT, children=(key_type, val_type))
        if isinstance(node, ast.Tuple):
            return InferredType(TypeKind.TUPLE, children=tuple(self._infer_expr(e) for e in node.elts))
        if isinstance(node, ast.BinOp):
            return self._infer_binop(node.op, self._infer_expr(node.left), self._infer_expr(node.right))
        if isinstance(node, ast.BoolOp):
            return SIMPLE_BOOL
        if isinstance(node, ast.UnaryOp):
            return SIMPLE_BOOL if isinstance(node.op, ast.Not) else self._infer_expr(node.operand)
        if isinstance(node, ast.Compare):
            return SIMPLE_BOOL
        if isinstance(node, ast.Call):
            return self._infer_call(node)
        if isinstance(node, ast.IfExp):
            return _merge_types([self._infer_expr(node.body), self._infer_expr(node.orelse)])
        if isinstance(node, ast.Name):
            stored = self._var_types.get(node.id, [])
            return _merge_types(stored) if stored else SIMPLE_UNKNOWN
        if isinstance(node, ast.ListComp):
            return InferredType(TypeKind.LIST)
        if isinstance(node, ast.SetComp):
            return InferredType(TypeKind.SET)
        if isinstance(node, ast.DictComp):
            return InferredType(TypeKind.DICT)
        if isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
            return SIMPLE_STR
        if isinstance(node, ast.Subscript):
            return SIMPLE_UNKNOWN
        return SIMPLE_UNKNOWN
    @staticmethod
    def _infer_constant(value: object) -> InferredType:
        if value is None: return SIMPLE_NONE
        _CONST = {bool: SIMPLE_BOOL, int: SIMPLE_INT, float: SIMPLE_FLOAT,
                  str: SIMPLE_STR, bytes: InferredType(TypeKind.SIMPLE, "bytes")}
        return _CONST.get(type(value), SIMPLE_UNKNOWN)

    def _infer_container(self, kind: TypeKind, elts: list) -> InferredType:
        if not elts:
            return InferredType(kind)
        child_types = [self._infer_expr(e) for e in elts]
        return InferredType(kind, children=(_merge_types(child_types),))
    @staticmethod
    def _infer_binop(op: ast.operator, left: InferredType, right: InferredType) -> InferredType:
        if isinstance(op, ast.Add):
            if left == SIMPLE_STR or right == SIMPLE_STR: return SIMPLE_STR
            if left == SIMPLE_FLOAT or right == SIMPLE_FLOAT: return SIMPLE_FLOAT
            if left == SIMPLE_INT and right == SIMPLE_INT: return SIMPLE_INT
        if isinstance(op, (ast.Sub, ast.Mult)):
            if left == SIMPLE_FLOAT or right == SIMPLE_FLOAT: return SIMPLE_FLOAT
            if left == SIMPLE_INT and right == SIMPLE_INT: return SIMPLE_INT
        if isinstance(op, ast.Div): return SIMPLE_FLOAT
        if isinstance(op, ast.FloorDiv): return SIMPLE_INT
        if isinstance(op, ast.Mod): return SIMPLE_STR if left == SIMPLE_STR else SIMPLE_INT
        if isinstance(op, ast.Pow):
            return SIMPLE_FLOAT if (left == SIMPLE_FLOAT or right == SIMPLE_FLOAT) else SIMPLE_INT
        return SIMPLE_UNKNOWN
    @staticmethod
    def _infer_call(node: ast.Call) -> InferredType:
        if not isinstance(node.func, ast.Name):
            return SIMPLE_UNKNOWN
        _BUILTIN_MAP: Dict[str, InferredType] = {
            "int": SIMPLE_INT, "float": SIMPLE_FLOAT, "str": SIMPLE_STR, "bool": SIMPLE_BOOL,
            "list": InferredType(TypeKind.LIST), "dict": InferredType(TypeKind.DICT),
            "set": InferredType(TypeKind.SET), "tuple": InferredType(TypeKind.TUPLE),
            "len": SIMPLE_INT, "abs": SIMPLE_INT, "round": SIMPLE_INT,
            "sorted": InferredType(TypeKind.LIST), "range": InferredType(TypeKind.LIST),
            "enumerate": InferredType(TypeKind.LIST), "zip": InferredType(TypeKind.LIST),
            "map": InferredType(TypeKind.LIST), "filter": InferredType(TypeKind.LIST),
            "isinstance": SIMPLE_BOOL, "hasattr": SIMPLE_BOOL, "callable": SIMPLE_BOOL,
            "repr": SIMPLE_STR, "hex": SIMPLE_STR, "oct": SIMPLE_STR, "bin": SIMPLE_STR,
            "ord": SIMPLE_INT, "chr": SIMPLE_STR, "sum": SIMPLE_INT,
            "any": SIMPLE_BOOL, "all": SIMPLE_BOOL, "print": SIMPLE_NONE, "input": SIMPLE_STR,
            "open": SIMPLE_UNKNOWN, "min": SIMPLE_UNKNOWN, "max": SIMPLE_UNKNOWN,
        }
        return _BUILTIN_MAP.get(node.func.id, SIMPLE_UNKNOWN)

class AnnotationInserter:
    """Inserts PEP 484 type annotations into source code."""

    def insert(self, source_code: str, type_map: Dict[str, InferredType]) -> str:
        tree = ast.parse(source_code)
        lines = source_code.splitlines(keepends=True)
        edits: List[Tuple[int, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                edit = self._annotate_function(node, lines, type_map)
                if edit is not None:
                    edits.append(edit)
        edits.sort(key=lambda e: e[0], reverse=True)
        for lineno, new_line in edits:
            if 0 <= lineno < len(lines):
                lines[lineno] = new_line
        return "".join(lines)

    def _annotate_function(
        self, node: ast.FunctionDef, lines: list, type_map: Dict[str, InferredType]
    ) -> Optional[Tuple[int, str]]:
        line_idx = node.lineno - 1
        if line_idx >= len(lines): return None
        original = lines[line_idx]
        if "->" in original: return None
        ret_type = type_map.get(f"{node.name}.__return__", SIMPLE_UNKNOWN)
        indent = re.match(r"^(\s*)", original).group(1)
        is_async = "async " in original.split("def")[0] if "def" in original else False
        parts: List[str] = []
        for arg in node.args.args:
            if arg.arg == "self": parts.append("self")
            elif arg.annotation is not None: parts.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
            else: parts.append(f"{arg.arg}: {type_map.get(arg.arg, SIMPLE_UNKNOWN).to_annotation()}")
        prefix = "async def" if is_async else "def"
        return (line_idx, f"{indent}{prefix} {node.name}({', '.join(parts)}) -> {ret_type.to_annotation()}:\n")

class StubGenerator:
    """Generates .pyi stub files from inferred types."""

    def generate(self, tree: ast.Module, type_map: Dict[str, InferredType]) -> str:
        lines: List[str] = [
            "# Auto-generated stub file",
            "from typing import Any, Dict, List, Optional, Set, Tuple, Union",
            "",
        ]
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines.append(self._stub_function(node, type_map, ""))
            elif isinstance(node, ast.ClassDef):
                lines.extend(self._stub_class(node, type_map))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        lines.append(f"{target.id}: {type_map.get(target.id, SIMPLE_UNKNOWN).to_annotation()}")
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                lines.append(f"{node.target.id}: {type_map.get(node.target.id, SIMPLE_UNKNOWN).to_annotation()}")
        lines.append("")
        return "\n".join(lines)

    def _stub_function(self, node: ast.FunctionDef, type_map: Dict[str, InferredType], indent: str) -> str:
        parts: List[str] = []
        for arg in node.args.args:
            if arg.arg == "self": parts.append("self")
            elif arg.annotation is not None: parts.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
            else: parts.append(f"{arg.arg}: {type_map.get(arg.arg, SIMPLE_UNKNOWN).to_annotation()}")
        ret = type_map.get(f"{node.name}.__return__", SIMPLE_UNKNOWN).to_annotation()
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{indent}{prefix} {node.name}({', '.join(parts)}) -> {ret}: ..."

    def _stub_class(self, node: ast.ClassDef, type_map: Dict[str, InferredType]) -> List[str]:
        bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
        header = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
        lines: List[str] = [header]
        has_body = False
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines.append(self._stub_function(child, type_map, "    "))
                has_body = True
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                t = type_map.get(child.target.id, SIMPLE_UNKNOWN)
                lines.append(f"    {child.target.id}: {t.to_annotation()}")
                has_body = True
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        t = type_map.get(target.id, SIMPLE_UNKNOWN)
                        lines.append(f"    {target.id}: {t.to_annotation()}")
                        has_body = True
        if not has_body:
            lines.append("    ...")
        lines.append("")
        return lines

class PriorityScorer:
    """Scores files by how much they'd benefit from annotations.
    Factors: number of functions, cyclomatic complexity estimate,
    estimated callers, and current annotation coverage gap.
    """

    def __init__(
        self,
        weight_functions: float = 0.30,
        weight_complexity: float = 0.25,
        weight_callers: float = 0.20,
        weight_coverage: float = 0.25,
    ) -> None:
        self._weights = np.array(
            [weight_functions, weight_complexity, weight_callers, weight_coverage],
            dtype=np.float64,
        )

    def score(self, file_info: FileState) -> float:
        func_score = min(file_info.total_functions / 20.0, 1.0)
        complexity = self._estimate_complexity(file_info)
        caller_score = min(file_info.total_functions * 0.15, 1.0)
        if file_info.total_functions > 0:
            coverage_gap = 1.0 - file_info.annotated_functions / file_info.total_functions
        else:
            coverage_gap = 0.0
        features = np.array([func_score, complexity, caller_score, coverage_gap], dtype=np.float64)
        return float(np.dot(self._weights, features))

    def prioritize(self, files: List[FileState]) -> List[FileState]:
        scored = [(self.score(f), f) for f in files]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored]
    @staticmethod
    def _estimate_complexity(file_info: FileState) -> float:
        if file_info.lines_of_code == 0:
            return 0.0
        ratio = file_info.lines_of_code / max(file_info.total_functions, 1)
        return min(ratio / 100.0, 1.0)

class IncrementalMigrator:
    """Plans incremental migration from untyped to fully-typed code.
    Supports strict mode (fully typed, empty py.typed) and permissive
    mode (partial py.typed marker).
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def plan_incremental(self, file_states: List[FileState]) -> MigrationPlan:
        scorer = PriorityScorer()
        ordered = scorer.prioritize(file_states)
        plan = MigrationPlan(files_in_order=[fs.path for fs in ordered], strict_mode=self.strict)
        phase_size = max(len(ordered) // 3, 1)
        phases: List[Dict[str, Any]] = []
        for i in range(0, len(ordered), phase_size):
            batch = ordered[i : i + phase_size]
            effort = sum(self._file_effort(fs) for fs in batch)
            phases.append({
                "phase": len(phases) + 1,
                "files": [fs.path for fs in batch],
                "estimated_hours": round(effort, 2),
            })
        plan.phases = phases
        plan.estimated_effort = sum(p["estimated_hours"] for p in phases)
        plan.py_typed_content = "" if self.strict else "partial\n"
        return plan
    @staticmethod
    def _file_effort(fs: FileState) -> float:
        unannotated_funcs = fs.total_functions - fs.annotated_functions
        unannotated_vars = fs.total_variables - fs.annotated_variables
        return unannotated_funcs * 0.15 + unannotated_vars * 0.05

class ErrorPredictor(ast.NodeVisitor):
    """Predicts mypy-like errors from adding strict types.
    Common predicted errors: missing-return-type, missing-param-type,
    incompatible-assignment, incompatible-return, missing-return-statement.
    """

    def __init__(self) -> None:
        self._errors: List[PredictedError] = []
        self._type_map: Dict[str, InferredType] = {}
        self._current_func_ret: Optional[InferredType] = None

    def predict(self, tree: ast.Module, type_map: Dict[str, InferredType]) -> List[PredictedError]:
        self._errors = []
        self._type_map = type_map
        self.visit(tree)
        return list(self._errors)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        ret_key = f"{node.name}.__return__"
        self._current_func_ret = self._type_map.get(ret_key)
        if ret_key not in self._type_map:
            self._errors.append(PredictedError(
                node.lineno, node.col_offset,
                "missing-return-type",
                f"Function '{node.name}' has no return type annotation",
            ))
        for arg in node.args.args:
            if arg.arg != "self" and arg.annotation is None and arg.arg not in self._type_map:
                ln = arg.lineno if hasattr(arg, "lineno") else node.lineno
                co = arg.col_offset if hasattr(arg, "col_offset") else 0
                self._errors.append(PredictedError(
                    ln, co, "missing-param-type",
                    f"Parameter '{arg.arg}' has no type annotation",
                ))
        if not self._has_return(node) and ret_key in self._type_map:
            rt = self._type_map[ret_key]
            if rt.kind not in (TypeKind.NONE, TypeKind.UNKNOWN):
                self._errors.append(PredictedError(
                    node.lineno, node.col_offset,
                    "missing-return-statement",
                    f"Function '{node.name}' declared to return {rt.to_annotation()} but may not return",
                ))
        self.generic_visit(node)
        self._current_func_ret = None
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                existing = self._type_map.get(target.id)
                if existing and existing.kind == TypeKind.UNION:
                    self._errors.append(PredictedError(
                        node.lineno, node.col_offset,
                        "incompatible-assignment",
                        f"Variable '{target.id}' has conflicting types: {existing.to_annotation()}",
                    ))
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if self._current_func_ret is not None and node.value is not None:
            val_type = TypeInferrer()._infer_expr(node.value)
            if (
                val_type.kind != TypeKind.UNKNOWN
                and self._current_func_ret.kind != TypeKind.UNKNOWN
                and val_type != self._current_func_ret
                and self._current_func_ret.kind not in (TypeKind.UNION, TypeKind.OPTIONAL)
            ):
                self._errors.append(PredictedError(
                    node.lineno, node.col_offset,
                    "incompatible-return",
                    f"Returning {val_type.to_annotation()} but expected {self._current_func_ret.to_annotation()}",
                ))
        self.generic_visit(node)
    @staticmethod
    def _has_return(node: ast.FunctionDef) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                return True
        return False

class ProgressTracker:
    """Tracks migration progress across a project.
    Computes percentage of code annotated, functions annotated, files
    fully typed, and estimates remaining effort based on velocity.
    """

    def __init__(self) -> None:
        self._history: List[Tuple[float, float]] = []

    def track(self, project_state: List[FileState]) -> ProgressReport:
        total_funcs = sum(fs.total_functions for fs in project_state)
        ann_funcs = sum(fs.annotated_functions for fs in project_state)
        total_vars = sum(fs.total_variables for fs in project_state)
        ann_vars = sum(fs.annotated_variables for fs in project_state)
        total_items = total_funcs + total_vars
        done_items = ann_funcs + ann_vars
        pct = (done_items / total_items * 100.0) if total_items > 0 else 0.0
        files_done = sum(1 for fs in project_state if fs.status == FileStatus.FULLY_TYPED)
        now = time.time()
        self._history.append((now, pct))
        velocity = self._compute_velocity()
        remaining = self._estimate_remaining(pct, velocity)
        return ProgressReport(
            percent_annotated=round(pct, 2),
            functions_done=ann_funcs,
            functions_total=total_funcs,
            files_done=files_done,
            files_total=len(project_state),
            estimated_remaining=round(remaining, 2),
            velocity=round(velocity, 4),
        )

    def _compute_velocity(self) -> float:
        if len(self._history) < 2:
            return 0.0
        timestamps = np.array([h[0] for h in self._history], dtype=np.float64)
        percents = np.array([h[1] for h in self._history], dtype=np.float64)
        dt = timestamps[-1] - timestamps[0]
        if dt <= 0:
            return 0.0
        return float((percents[-1] - percents[0]) / dt)
    @staticmethod
    def _estimate_remaining(pct: float, velocity: float) -> float:
        if velocity <= 0 or pct >= 100.0:
            return 0.0
        return (100.0 - pct) / velocity / 3600.0

class TypeChecker:
    """Checks consistency of inferred types across call sites."""

    def check_consistency(self, type_map: Dict[str, InferredType]) -> List[TypeConflict]:
        conflicts: List[TypeConflict] = []
        for var, inferred in type_map.items():
            if inferred.kind != TypeKind.UNION or not inferred.children: continue
            for i, child_a in enumerate(inferred.children):
                for j in range(i + 1, len(inferred.children)):
                    child_b = inferred.children[j]
                    if not self._compatible(child_a, child_b):
                        conflicts.append(TypeConflict(
                            variable=var, location_a=0, location_b=0,
                            type_a=child_a, type_b=child_b,
                            message=f"Variable '{var}' used as both {child_a.to_annotation()} and {child_b.to_annotation()}",
                        ))
        return conflicts
    @staticmethod
    def _compatible(a: InferredType, b: InferredType) -> bool:
        if a == b:
            return True
        if a.kind == TypeKind.SIMPLE and b.kind == TypeKind.SIMPLE:
            numeric_names = {"int", "float", "bool"}
            if a.name in numeric_names and b.name in numeric_names:
                return True
        if a.kind == TypeKind.NONE or b.kind == TypeKind.NONE:
            return True
        return False

def _extract_file_state(path: str, source: str) -> FileState:
    """Build a FileState by analysing source code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return FileState(path=path, lines_of_code=source.count("\n"))
    total_funcs = 0
    annotated_funcs = 0
    total_vars = 0
    annotated_vars = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            total_funcs += 1
            has_ret = node.returns is not None
            all_args = all(
                a.annotation is not None for a in node.args.args if a.arg != "self"
            )
            if has_ret and all_args:
                annotated_funcs += 1
        elif isinstance(node, ast.AnnAssign):
            total_vars += 1
            annotated_vars += 1
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    total_vars += 1
    if total_funcs == 0 or annotated_funcs == total_funcs:
        status = FileStatus.FULLY_TYPED
    elif annotated_funcs > 0:
        status = FileStatus.PARTIAL
    else:
        status = FileStatus.UNTYPED
    return FileState(
        path=path,
        status=status,
        total_functions=total_funcs,
        annotated_functions=annotated_funcs,
        total_variables=total_vars,
        annotated_variables=annotated_vars,
        lines_of_code=source.count("\n") + 1,
    )

class TypeMigrationAssistant:
    """High-level assistant for migrating a Python project to typed code.
    Combines type inference, annotation insertion, stub generation,
    error prediction, type checking, and progress tracking.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self._inferrer = TypeInferrer()
        self._inserter = AnnotationInserter()
        self._stub_gen = StubGenerator()
        self._migrator = IncrementalMigrator(strict=strict)
        self._predictor = ErrorPredictor()
        self._checker = TypeChecker()
        self._tracker = ProgressTracker()

    def plan(self, source_files: Dict[str, str]) -> MigrationPlan:
        """Produce an ordered migration plan for the given source files."""
        file_states = [_extract_file_state(p, s) for p, s in source_files.items()]
        plan = self._migrator.plan_incremental(file_states)
        error_count = 0
        for source in source_files.values():
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            tmap = self._inferrer.infer(tree)
            error_count += len(self._predictor.predict(tree, tmap))
        plan.estimated_effort += error_count * 0.02
        return plan

    def annotate(self, source_code: str) -> str:
        """Return an annotated version of the source code."""
        tree = ast.parse(source_code)
        type_map = self._inferrer.infer(tree)
        return self._inserter.insert(source_code, type_map)

    def generate_stubs(self, source_code: str) -> str:
        """Return a .pyi stub for the given source."""
        tree = ast.parse(source_code)
        type_map = self._inferrer.infer(tree)
        return self._stub_gen.generate(tree, type_map)

    def predict_errors(self, source_code: str) -> List[PredictedError]:
        """Predict typing errors for the given source."""
        tree = ast.parse(source_code)
        type_map = self._inferrer.infer(tree)
        return self._predictor.predict(tree, type_map)

    def check_types(self, source_code: str) -> List[TypeConflict]:
        """Check type consistency for the given source."""
        tree = ast.parse(source_code)
        type_map = self._inferrer.infer(tree)
        return self._checker.check_consistency(type_map)
    def get_progress(self, source_files: Dict[str, str]) -> ProgressReport:
        """Report migration progress across the project."""
        states = [_extract_file_state(p, s) for p, s in source_files.items()]
        return self._tracker.track(states)
