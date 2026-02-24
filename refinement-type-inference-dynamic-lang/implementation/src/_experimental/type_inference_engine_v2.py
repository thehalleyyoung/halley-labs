"""
Python type inference engine: infer types from unannotated Python source
using constraint generation and unification, with flow-sensitive narrowing.
"""

import ast
import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Abstract type system
# ---------------------------------------------------------------------------

class TypeKind(Enum):
    TOP = auto()
    BOTTOM = auto()
    INT = auto()
    FLOAT = auto()
    STR = auto()
    BOOL = auto()
    NONE = auto()
    BYTES = auto()
    LIST = auto()
    DICT = auto()
    SET = auto()
    TUPLE = auto()
    OPTIONAL = auto()
    UNION = auto()
    CALLABLE = auto()
    CLASS = auto()
    TYPEVAR = auto()
    INSTANCE = auto()
    MODULE = auto()


@dataclass(frozen=True)
class AbstractType:
    kind: TypeKind
    params: Tuple["AbstractType", ...] = ()
    name: str = ""

    def __str__(self) -> str:
        if self.kind == TypeKind.TOP:
            return "Any"
        if self.kind == TypeKind.BOTTOM:
            return "Never"
        if self.kind in (TypeKind.INT, TypeKind.FLOAT, TypeKind.STR,
                         TypeKind.BOOL, TypeKind.NONE, TypeKind.BYTES):
            return self.kind.name.capitalize()
        if self.kind == TypeKind.LIST:
            inner = str(self.params[0]) if self.params else "Any"
            return f"List[{inner}]"
        if self.kind == TypeKind.DICT:
            k = str(self.params[0]) if len(self.params) > 0 else "Any"
            v = str(self.params[1]) if len(self.params) > 1 else "Any"
            return f"Dict[{k}, {v}]"
        if self.kind == TypeKind.SET:
            inner = str(self.params[0]) if self.params else "Any"
            return f"Set[{inner}]"
        if self.kind == TypeKind.TUPLE:
            elts = ", ".join(str(p) for p in self.params)
            return f"Tuple[{elts}]"
        if self.kind == TypeKind.OPTIONAL:
            inner = str(self.params[0]) if self.params else "Any"
            return f"Optional[{inner}]"
        if self.kind == TypeKind.UNION:
            parts = ", ".join(str(p) for p in self.params)
            return f"Union[{parts}]"
        if self.kind == TypeKind.CALLABLE:
            args = ", ".join(str(p) for p in self.params[:-1]) if len(self.params) > 1 else ""
            ret = str(self.params[-1]) if self.params else "Any"
            return f"Callable[[{args}], {ret}]"
        if self.kind in (TypeKind.CLASS, TypeKind.INSTANCE):
            return self.name or self.kind.name
        if self.kind == TypeKind.TYPEVAR:
            return self.name or "T"
        if self.kind == TypeKind.MODULE:
            return f"Module[{self.name}]"
        return self.kind.name


# Singletons
TOP = AbstractType(TypeKind.TOP)
BOTTOM = AbstractType(TypeKind.BOTTOM)
INT = AbstractType(TypeKind.INT)
FLOAT = AbstractType(TypeKind.FLOAT)
STR = AbstractType(TypeKind.STR)
BOOL = AbstractType(TypeKind.BOOL)
NONE_TYPE = AbstractType(TypeKind.NONE)
BYTES = AbstractType(TypeKind.BYTES)


def List_(elem: AbstractType = TOP) -> AbstractType:
    return AbstractType(TypeKind.LIST, (elem,))


def Dict_(key: AbstractType = TOP, val: AbstractType = TOP) -> AbstractType:
    return AbstractType(TypeKind.DICT, (key, val))


def Set_(elem: AbstractType = TOP) -> AbstractType:
    return AbstractType(TypeKind.SET, (elem,))


def Tuple_(*elems: AbstractType) -> AbstractType:
    return AbstractType(TypeKind.TUPLE, elems)


def Optional_(inner: AbstractType) -> AbstractType:
    return AbstractType(TypeKind.OPTIONAL, (inner,))


def Union_(*parts: AbstractType) -> AbstractType:
    flat: List[AbstractType] = []
    for p in parts:
        if p.kind == TypeKind.UNION:
            flat.extend(p.params)
        elif p not in flat:
            flat.append(p)
    if len(flat) == 1:
        return flat[0]
    return AbstractType(TypeKind.UNION, tuple(flat))


def Callable_(args: List[AbstractType], ret: AbstractType) -> AbstractType:
    return AbstractType(TypeKind.CALLABLE, tuple(args) + (ret,))


# ---------------------------------------------------------------------------
# Type constraints
# ---------------------------------------------------------------------------

class ConstraintKind(Enum):
    EQUAL = auto()
    SUBTYPE = auto()


@dataclass
class Constraint:
    kind: ConstraintKind
    left: AbstractType
    right: AbstractType
    source_line: int = 0


# ---------------------------------------------------------------------------
# Unification / constraint solver
# ---------------------------------------------------------------------------

_tv_counter = itertools.count(0)


def fresh_typevar(prefix: str = "T") -> AbstractType:
    return AbstractType(TypeKind.TYPEVAR, name=f"{prefix}{next(_tv_counter)}")


class Substitution:
    """Maps type variables to concrete types."""

    def __init__(self) -> None:
        self.mapping: Dict[str, AbstractType] = {}

    def apply(self, t: AbstractType) -> AbstractType:
        if t.kind == TypeKind.TYPEVAR:
            if t.name in self.mapping:
                resolved = self.mapping[t.name]
                if resolved != t:
                    return self.apply(resolved)
            return t
        if t.params:
            new_params = tuple(self.apply(p) for p in t.params)
            return AbstractType(t.kind, new_params, t.name)
        return t

    def unify(self, a: AbstractType, b: AbstractType) -> bool:
        a = self.apply(a)
        b = self.apply(b)

        if a == b:
            return True
        if a.kind == TypeKind.TYPEVAR:
            self.mapping[a.name] = b
            return True
        if b.kind == TypeKind.TYPEVAR:
            self.mapping[b.name] = a
            return True
        if a.kind == TypeKind.TOP or b.kind == TypeKind.TOP:
            return True
        if a.kind == TypeKind.BOTTOM:
            return True
        if b.kind == TypeKind.BOTTOM:
            return True
        # Int <: Float
        if a.kind == TypeKind.INT and b.kind == TypeKind.FLOAT:
            return True
        if a.kind == TypeKind.BOOL and b.kind == TypeKind.INT:
            return True
        # Optional[T] unifies with T or None
        if a.kind == TypeKind.OPTIONAL:
            return self.unify(a.params[0], b) or b.kind == TypeKind.NONE
        if b.kind == TypeKind.OPTIONAL:
            return self.unify(a, b.params[0]) or a.kind == TypeKind.NONE
        if a.kind == TypeKind.UNION:
            return any(self.unify(p, b) for p in a.params)
        if b.kind == TypeKind.UNION:
            return any(self.unify(a, p) for p in b.params)
        if a.kind != b.kind:
            return False
        if len(a.params) != len(b.params):
            return False
        return all(self.unify(ap, bp) for ap, bp in zip(a.params, b.params))


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

@dataclass
class Signature:
    name: str
    param_types: Dict[str, AbstractType] = field(default_factory=dict)
    return_type: AbstractType = field(default_factory=lambda: TOP)
    is_method: bool = False


# ---------------------------------------------------------------------------
# TypeEnv
# ---------------------------------------------------------------------------

@dataclass
class TypeEnv:
    variables: Dict[str, AbstractType] = field(default_factory=dict)
    functions: Dict[str, Signature] = field(default_factory=dict)
    classes: Dict[str, Dict[str, AbstractType]] = field(default_factory=dict)

    def get(self, name: str) -> AbstractType:
        return self.variables.get(name, TOP)

    def set(self, name: str, ty: AbstractType) -> None:
        self.variables[name] = ty

    def copy(self) -> "TypeEnv":
        return TypeEnv(
            variables=dict(self.variables),
            functions=dict(self.functions),
            classes={k: dict(v) for k, v in self.classes.items()},
        )


# ---------------------------------------------------------------------------
# Constraint generator
# ---------------------------------------------------------------------------

_BUILTIN_TYPES: Dict[str, AbstractType] = {
    "int": INT,
    "float": FLOAT,
    "str": STR,
    "bool": BOOL,
    "bytes": BYTES,
    "list": List_(),
    "dict": Dict_(),
    "set": Set_(),
    "tuple": Tuple_(),
    "None": NONE_TYPE,
    "True": BOOL,
    "False": BOOL,
}


class ConstraintGenerator(ast.NodeVisitor):
    """Walk the AST and generate type constraints."""

    def __init__(self) -> None:
        self.env = TypeEnv()
        self.constraints: List[Constraint] = []
        self._return_types: List[List[AbstractType]] = []
        self._current_class: Optional[str] = None
        self._sub = Substitution()

    def infer_expr(self, node: ast.expr) -> AbstractType:
        if isinstance(node, ast.Constant):
            return self._constant_type(node)
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                return self.env.get(node.id)
            return TOP
        if isinstance(node, ast.BinOp):
            return self._infer_binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._infer_unaryop(node)
        if isinstance(node, ast.BoolOp):
            return BOOL
        if isinstance(node, ast.Compare):
            return BOOL
        if isinstance(node, ast.List):
            return self._infer_list(node)
        if isinstance(node, ast.Dict):
            return self._infer_dict(node)
        if isinstance(node, ast.Set):
            return self._infer_set(node)
        if isinstance(node, ast.Tuple):
            return self._infer_tuple(node)
        if isinstance(node, ast.Call):
            return self._infer_call(node)
        if isinstance(node, ast.Attribute):
            return self._infer_attribute(node)
        if isinstance(node, ast.Subscript):
            return self._infer_subscript(node)
        if isinstance(node, ast.IfExp):
            return self._infer_ifexp(node)
        if isinstance(node, ast.ListComp):
            return List_()
        if isinstance(node, ast.SetComp):
            return Set_()
        if isinstance(node, ast.DictComp):
            return Dict_()
        if isinstance(node, ast.GeneratorExp):
            return TOP
        if isinstance(node, ast.JoinedStr):
            return STR
        if isinstance(node, ast.FormattedValue):
            return STR
        if isinstance(node, ast.Lambda):
            return self._infer_lambda(node)
        if isinstance(node, ast.Starred):
            return List_()
        return TOP

    def _constant_type(self, node: ast.Constant) -> AbstractType:
        v = node.value
        if isinstance(v, bool):
            return BOOL
        if isinstance(v, int):
            return INT
        if isinstance(v, float):
            return FLOAT
        if isinstance(v, str):
            return STR
        if isinstance(v, bytes):
            return BYTES
        if v is None:
            return NONE_TYPE
        return TOP

    def _infer_binop(self, node: ast.BinOp) -> AbstractType:
        lt = self.infer_expr(node.left)
        rt = self.infer_expr(node.right)
        if isinstance(node.op, ast.Add):
            if lt == STR and rt == STR:
                return STR
            if lt == INT and rt == INT:
                return INT
            if lt == FLOAT or rt == FLOAT:
                return FLOAT
            if lt.kind == TypeKind.LIST:
                return lt
            return TOP
        if isinstance(node.op, (ast.Sub, ast.Mult)):
            if lt == INT and rt == INT:
                return INT
            if lt == FLOAT or rt == FLOAT:
                return FLOAT
            if isinstance(node.op, ast.Mult) and lt == STR and rt == INT:
                return STR
            return TOP
        if isinstance(node.op, ast.Div):
            return FLOAT
        if isinstance(node.op, ast.FloorDiv):
            if lt == INT and rt == INT:
                return INT
            return FLOAT
        if isinstance(node.op, ast.Mod):
            if lt == STR:
                return STR
            if lt == INT and rt == INT:
                return INT
            return TOP
        if isinstance(node.op, ast.Pow):
            if lt == INT and rt == INT:
                return INT
            return FLOAT
        if isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor,
                                 ast.LShift, ast.RShift)):
            return INT
        return TOP

    def _infer_unaryop(self, node: ast.UnaryOp) -> AbstractType:
        t = self.infer_expr(node.operand)
        if isinstance(node.op, ast.Not):
            return BOOL
        if isinstance(node.op, ast.USub):
            return t
        if isinstance(node.op, ast.Invert):
            return INT
        return t

    def _infer_list(self, node: ast.List) -> AbstractType:
        if not node.elts:
            return List_()
        elem_types = [self.infer_expr(e) for e in node.elts]
        unified = self._join_types(elem_types)
        return List_(unified)

    def _infer_dict(self, node: ast.Dict) -> AbstractType:
        if not node.keys:
            return Dict_()
        key_types = [self.infer_expr(k) for k in node.keys if k is not None]
        val_types = [self.infer_expr(v) for v in node.values]
        kt = self._join_types(key_types) if key_types else TOP
        vt = self._join_types(val_types) if val_types else TOP
        return Dict_(kt, vt)

    def _infer_set(self, node: ast.Set) -> AbstractType:
        if not node.elts:
            return Set_()
        elem_types = [self.infer_expr(e) for e in node.elts]
        return Set_(self._join_types(elem_types))

    def _infer_tuple(self, node: ast.Tuple) -> AbstractType:
        return Tuple_(*(self.infer_expr(e) for e in node.elts))

    def _infer_call(self, node: ast.Call) -> AbstractType:
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in _BUILTIN_TYPES:
                return _BUILTIN_TYPES[name]
            if name == "len":
                return INT
            if name == "range":
                return List_(INT)
            if name == "sorted":
                return List_()
            if name == "isinstance":
                return BOOL
            if name == "type":
                return TOP
            if name == "input":
                return STR
            if name == "open":
                return TOP
            sig = self.env.functions.get(name)
            if sig:
                return sig.return_type
            return TOP
        if isinstance(node.func, ast.Attribute):
            obj_type = self.infer_expr(node.func.value)
            attr = node.func.attr
            if obj_type == STR:
                if attr in ("join", "format", "replace", "strip", "lstrip",
                            "rstrip", "upper", "lower", "title", "capitalize",
                            "encode", "decode"):
                    return STR if attr != "encode" else BYTES
                if attr in ("split", "splitlines", "rsplit"):
                    return List_(STR)
                if attr in ("find", "index", "rfind", "rindex", "count"):
                    return INT
                if attr in ("startswith", "endswith", "isdigit", "isalpha"):
                    return BOOL
            if obj_type.kind == TypeKind.LIST:
                if attr == "append":
                    return NONE_TYPE
                if attr == "pop":
                    return obj_type.params[0] if obj_type.params else TOP
                if attr == "copy":
                    return obj_type
                if attr in ("index", "count"):
                    return INT
                if attr == "sort":
                    return NONE_TYPE
                if attr == "reverse":
                    return NONE_TYPE
            if obj_type.kind == TypeKind.DICT:
                if attr == "get":
                    vt = obj_type.params[1] if len(obj_type.params) > 1 else TOP
                    return Optional_(vt)
                if attr == "keys":
                    return List_(obj_type.params[0] if obj_type.params else TOP)
                if attr == "values":
                    return List_(obj_type.params[1] if len(obj_type.params) > 1 else TOP)
                if attr == "items":
                    kt = obj_type.params[0] if obj_type.params else TOP
                    vt = obj_type.params[1] if len(obj_type.params) > 1 else TOP
                    return List_(Tuple_(kt, vt))
            return TOP
        return TOP

    def _infer_attribute(self, node: ast.Attribute) -> AbstractType:
        obj_type = self.infer_expr(node.value)
        if self._current_class and isinstance(node.value, ast.Name) and node.value.id == "self":
            cls_attrs = self.env.classes.get(self._current_class, {})
            return cls_attrs.get(node.attr, TOP)
        return TOP

    def _infer_subscript(self, node: ast.Subscript) -> AbstractType:
        val_type = self.infer_expr(node.value)
        if val_type.kind == TypeKind.LIST:
            return val_type.params[0] if val_type.params else TOP
        if val_type.kind == TypeKind.DICT:
            return val_type.params[1] if len(val_type.params) > 1 else TOP
        if val_type.kind == TypeKind.TUPLE:
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                idx = node.slice.value
                if 0 <= idx < len(val_type.params):
                    return val_type.params[idx]
            return TOP
        if val_type == STR:
            return STR
        return TOP

    def _infer_ifexp(self, node: ast.IfExp) -> AbstractType:
        bt = self.infer_expr(node.body)
        ot = self.infer_expr(node.orelse)
        return self._join_types([bt, ot])

    def _infer_lambda(self, node: ast.Lambda) -> AbstractType:
        ret = self.infer_expr(node.body)
        arg_types = [TOP] * len(node.args.args)
        return Callable_(arg_types, ret)

    def _join_types(self, types: List[AbstractType]) -> AbstractType:
        unique = list(dict.fromkeys(types))
        unique = [t for t in unique if t.kind != TypeKind.TOP]
        if not unique:
            return TOP
        if len(unique) == 1:
            return unique[0]
        return Union_(*unique)

    # -- statement visitors -----------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        val_type = self.infer_expr(node.value)
        for target in node.targets:
            self._assign_target(target, val_type)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        val_type = self.infer_expr(node.value)
        if isinstance(node.target, ast.Name):
            existing = self.env.get(node.target.id)
            result = self._infer_binop(ast.BinOp(
                left=ast.Name(id=node.target.id, ctx=ast.Load()),
                op=node.op,
                right=node.value,
            ))
            self.env.set(node.target.id, result)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value:
            val_type = self.infer_expr(node.value)
        else:
            val_type = TOP
        ann_type = self._annotation_to_type(node.annotation)
        if isinstance(node.target, ast.Name):
            self.env.set(node.target.id, ann_type if ann_type != TOP else val_type)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        sig = self._infer_function_sig(node)
        self.env.functions[node.name] = sig
        self.env.set(node.name, Callable_(
            list(sig.param_types.values()), sig.return_type
        ))

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        old_class = self._current_class
        self._current_class = node.name
        self.env.classes.setdefault(node.name, {})

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = self._infer_function_sig(item)
                self.env.functions[f"{node.name}.{item.name}"] = sig
                if item.name == "__init__":
                    # Infer instance attributes from __init__
                    for stmt in item.body:
                        if isinstance(stmt, ast.Assign):
                            for t in stmt.targets:
                                if (isinstance(t, ast.Attribute)
                                        and isinstance(t.value, ast.Name)
                                        and t.value.id == "self"):
                                    attr_type = self.infer_expr(stmt.value)
                                    self.env.classes[node.name][t.attr] = attr_type
            elif isinstance(item, ast.Assign):
                val_type = self.infer_expr(item.value)
                for t in item.targets:
                    if isinstance(t, ast.Name):
                        self.env.classes[node.name][t.id] = val_type

        self.env.set(node.name, AbstractType(TypeKind.CLASS, name=node.name))
        self._current_class = old_class

    def visit_For(self, node: ast.For) -> None:
        iter_type = self.infer_expr(node.iter)
        elem_type = TOP
        if iter_type.kind == TypeKind.LIST and iter_type.params:
            elem_type = iter_type.params[0]
        elif iter_type == STR:
            elem_type = STR
        elif iter_type.kind == TypeKind.DICT and iter_type.params:
            elem_type = iter_type.params[0]
        self._assign_target(node.target, elem_type)
        for stmt in node.body:
            self.visit(stmt)

    def visit_If(self, node: ast.If) -> None:
        # Flow-sensitive: narrow types after isinstance/None checks
        narrowed_env = self.env.copy()
        self._apply_narrowing(node.test, narrowed_env)
        old_env = self.env
        self.env = narrowed_env
        for stmt in node.body:
            self.visit(stmt)
        self.env = old_env
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_Return(self, node: ast.Return) -> None:
        if self._return_types:
            if node.value:
                rt = self.infer_expr(node.value)
            else:
                rt = NONE_TYPE
            self._return_types[-1].append(rt)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self.env.set(name, AbstractType(TypeKind.MODULE, name=alias.name))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            self.env.set(name, TOP)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars:
                ctx_type = self.infer_expr(item.context_expr)
                self._assign_target(item.optional_vars, ctx_type)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Try(self, node: ast.Try) -> None:
        for stmt in node.body:
            self.visit(stmt)
        for handler in node.handlers:
            if handler.name:
                self.env.set(handler.name, TOP)
            for stmt in handler.body:
                self.visit(stmt)
        for stmt in node.finalbody:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_Expr(self, node: ast.Expr) -> None:
        self.infer_expr(node.value)

    def visit_While(self, node: ast.While) -> None:
        # Type widening for loops: visit body twice
        snapshot = self.env.copy()
        for stmt in node.body:
            self.visit(stmt)
        # Widen: merge with pre-loop types
        for var in self.env.variables:
            if var in snapshot.variables and self.env.variables[var] != snapshot.variables[var]:
                self.env.variables[var] = Union_(snapshot.variables[var], self.env.variables[var])
        for stmt in node.body:
            self.visit(stmt)

    # -- helpers -----------------------------------------------------------

    def _assign_target(self, target: ast.expr, val_type: AbstractType) -> None:
        if isinstance(target, ast.Name):
            self.env.set(target.id, val_type)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, elt in enumerate(target.elts):
                if val_type.kind == TypeKind.TUPLE and i < len(val_type.params):
                    self._assign_target(elt, val_type.params[i])
                elif val_type.kind == TypeKind.LIST and val_type.params:
                    self._assign_target(elt, val_type.params[0])
                else:
                    self._assign_target(elt, TOP)
        elif isinstance(target, ast.Starred):
            self._assign_target(target.value, List_())
        elif isinstance(target, ast.Attribute):
            if (self._current_class
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                self.env.classes.setdefault(self._current_class, {})[target.attr] = val_type

    def _infer_function_sig(self, node: ast.FunctionDef) -> Signature:
        sig = Signature(name=node.name)
        is_method = self._current_class is not None

        args = node.args
        for i, arg in enumerate(args.args):
            if i == 0 and is_method and arg.arg == "self":
                sig.is_method = True
                continue
            if arg.annotation:
                sig.param_types[arg.arg] = self._annotation_to_type(arg.annotation)
            else:
                # Try to infer from defaults
                default_idx = i - (len(args.args) - len(args.defaults))
                if default_idx >= 0 and default_idx < len(args.defaults):
                    sig.param_types[arg.arg] = self.infer_expr(args.defaults[default_idx])
                else:
                    sig.param_types[arg.arg] = TOP

        for arg in args.kwonlyargs:
            if arg.annotation:
                sig.param_types[arg.arg] = self._annotation_to_type(arg.annotation)
            else:
                sig.param_types[arg.arg] = TOP

        # Infer return type from body
        self._return_types.append([])
        old_env = self.env.copy()
        for pname, ptype in sig.param_types.items():
            self.env.set(pname, ptype)
        if is_method:
            self.env.set("self", AbstractType(TypeKind.INSTANCE, name=self._current_class or ""))

        for stmt in node.body:
            self.visit(stmt)

        returns = self._return_types.pop()
        if node.returns:
            sig.return_type = self._annotation_to_type(node.returns)
        elif returns:
            sig.return_type = self._join_types(returns)
        else:
            sig.return_type = NONE_TYPE

        self.env = old_env
        return sig

    def _annotation_to_type(self, node: ast.expr) -> AbstractType:
        if isinstance(node, ast.Constant):
            if node.value is None:
                return NONE_TYPE
            if isinstance(node.value, str):
                return _BUILTIN_TYPES.get(node.value, TOP)
        if isinstance(node, ast.Name):
            name = node.id
            if name in _BUILTIN_TYPES:
                return _BUILTIN_TYPES[name]
            if name in self.env.classes:
                return AbstractType(TypeKind.INSTANCE, name=name)
            return TOP
        if isinstance(node, ast.Subscript):
            base = node.value
            if isinstance(base, ast.Name):
                if base.id == "Optional":
                    inner = self._annotation_to_type(node.slice)
                    return Optional_(inner)
                if base.id == "List":
                    inner = self._annotation_to_type(node.slice)
                    return List_(inner)
                if base.id == "Dict":
                    if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                        k = self._annotation_to_type(node.slice.elts[0])
                        v = self._annotation_to_type(node.slice.elts[1])
                        return Dict_(k, v)
                if base.id == "Set":
                    inner = self._annotation_to_type(node.slice)
                    return Set_(inner)
                if base.id == "Tuple":
                    if isinstance(node.slice, ast.Tuple):
                        elts = tuple(self._annotation_to_type(e) for e in node.slice.elts)
                        return Tuple_(*elts)
                if base.id == "Union":
                    if isinstance(node.slice, ast.Tuple):
                        parts = tuple(self._annotation_to_type(e) for e in node.slice.elts)
                        return Union_(*parts)
        return TOP

    def _apply_narrowing(self, test: ast.expr, env: TypeEnv) -> None:
        """Narrow types based on isinstance / None checks."""
        if isinstance(test, ast.Call):
            if isinstance(test.func, ast.Name) and test.func.id == "isinstance":
                if len(test.args) == 2 and isinstance(test.args[0], ast.Name):
                    var = test.args[0].id
                    narrowed = self._annotation_to_type(test.args[1])
                    if narrowed != TOP:
                        env.set(var, narrowed)
        elif isinstance(test, ast.Compare) and len(test.ops) == 1:
            if isinstance(test.ops[0], ast.IsNot):
                if (isinstance(test.comparators[0], ast.Constant)
                        and test.comparators[0].value is None
                        and isinstance(test.left, ast.Name)):
                    cur = env.get(test.left.id)
                    if cur.kind == TypeKind.OPTIONAL and cur.params:
                        env.set(test.left.id, cur.params[0])
            elif isinstance(test.ops[0], ast.Is):
                if (isinstance(test.comparators[0], ast.Constant)
                        and test.comparators[0].value is None
                        and isinstance(test.left, ast.Name)):
                    env.set(test.left.id, NONE_TYPE)
        elif isinstance(test, ast.Name):
            cur = env.get(test.id)
            if cur.kind == TypeKind.OPTIONAL and cur.params:
                env.set(test.id, cur.params[0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TypeInferencer:
    """Infer types for Python source code."""

    def infer(self, source_code: str) -> TypeEnv:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return TypeEnv()

        gen = ConstraintGenerator()
        for node in tree.body:
            gen.visit(node)
        return gen.env
