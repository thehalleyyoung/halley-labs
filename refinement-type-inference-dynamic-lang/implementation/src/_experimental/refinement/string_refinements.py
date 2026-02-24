"""
Refined string type analysis for Python programs.

Models string refinements ({s:str | s.startswith('http')}, {s:str | s.isdigit()},
{s:str | len(s) <= 255}, regex-matched patterns) and tracks taint propagation.
"""

from __future__ import annotations

import ast
import re
import string
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from src.refinement_lattice import (
    ANY_TYPE, BOOL_TYPE, INT_TYPE, NEVER_TYPE, NONE_TYPE, STR_TYPE,
    BaseTypeKind, BaseTypeR, Pred, PredOp, RefType,
)

# Supporting data types

class TaintSource(Enum):
    USER_INPUT = auto()
    DATABASE = auto()
    FILE = auto()
    NETWORK = auto()
    ENVIRONMENT = auto()

class StringProperty(Enum):
    STARTSWITH = auto()
    ENDSWITH = auto()
    ISDIGIT = auto()
    ISALPHA = auto()
    ISALNUM = auto()
    ISUPPER = auto()
    ISLOWER = auto()
    ISSPACE = auto()
    ISIDENTIFIER = auto()
    MATCHES_REGEX = auto()

@dataclass(frozen=True)
class StringPredicate:
    """A specific string property (startswith, isdigit, matches_regex, etc.)."""
    prop: StringProperty
    argument: Optional[str] = None
    negated: bool = False

    def to_pred(self, var: str) -> Pred:
        attr_map = {
            StringProperty.STARTSWITH: f"__startswith_{self.argument}",
            StringProperty.ENDSWITH: f"__endswith_{self.argument}",
            StringProperty.ISDIGIT: "__isdigit__",
            StringProperty.ISALPHA: "__isalpha__",
            StringProperty.ISALNUM: "__isalnum__",
            StringProperty.ISUPPER: "__isupper__",
            StringProperty.ISLOWER: "__islower__",
            StringProperty.ISSPACE: "__isspace__",
            StringProperty.ISIDENTIFIER: "__isidentifier__",
            StringProperty.MATCHES_REGEX: f"__matches_{self.argument}",
        }
        p = Pred.hasattr_(var, attr_map.get(self.prop, "__unknown__"))
        return p.not_() if self.negated else p

@dataclass
class TaintStatus:
    source: Optional[TaintSource] = None
    sanitizers_applied: List[str] = field(default_factory=list)
    is_safe: bool = True

    def tainted(self) -> bool:
        return self.source is not None and not self.is_safe

    def apply_sanitizer(self, sanitizer: str) -> TaintStatus:
        return TaintStatus(self.source, self.sanitizers_applied + [sanitizer], True)

    def propagate(self) -> TaintStatus:
        return TaintStatus(self.source, list(self.sanitizers_applied), self.is_safe)

@dataclass
class RegexInfo:
    pattern: str
    num_groups: int = 0
    named_groups: Dict[str, int] = field(default_factory=dict)
    anchored_start: bool = False
    anchored_end: bool = False
    is_fullmatch: bool = False

    @staticmethod
    def from_pattern(pattern: str) -> RegexInfo:
        anchored_s = pattern.startswith("^")
        anchored_e = pattern.endswith("$") and not pattern.endswith("\\$")
        try:
            c = re.compile(pattern)
            ng, named = c.groups, dict(c.groupindex)
        except re.error:
            ng, named = 0, {}
        return RegexInfo(pattern, ng, named, anchored_s, anchored_e,
                         anchored_s and anchored_e)

@dataclass
class AnalysisState:
    var_types: Dict[str, RefType] = field(default_factory=dict)
    predicates: Dict[str, List[Pred]] = field(default_factory=dict)
    string_props: Dict[str, List[StringPredicate]] = field(default_factory=dict)
    taint: Dict[str, TaintStatus] = field(default_factory=dict)
    regex_matches: Dict[str, RegexInfo] = field(default_factory=dict)

    def copy(self) -> AnalysisState:
        return AnalysisState(
            var_types=dict(self.var_types),
            predicates={k: list(v) for k, v in self.predicates.items()},
            string_props={k: list(v) for k, v in self.string_props.items()},
            taint={k: TaintStatus(v.source, list(v.sanitizers_applied), v.is_safe)
                   for k, v in self.taint.items()},
            regex_matches=dict(self.regex_matches),
        )

    def add_pred(self, var: str, pred: Pred) -> AnalysisState:
        new = self.copy()
        new.predicates.setdefault(var, []).append(pred)
        return new

    def add_string_prop(self, var: str, prop: StringPredicate) -> AnalysisState:
        new = self.copy()
        new.string_props.setdefault(var, []).append(prop)
        return new

    def set_type(self, var: str, typ: RefType) -> AnalysisState:
        new = self.copy()
        new.var_types[var] = typ
        return new

    def get_type(self, var: str) -> Optional[RefType]:
        return self.var_types.get(var)

    def combined_pred(self, var: str) -> Pred:
        preds = self.predicates.get(var, [])
        if not preds:
            return Pred.true_()
        result = preds[0]
        for p in preds[1:]:
            result = result.and_(p)
        return result

    def merge(self, other: AnalysisState) -> AnalysisState:
        merged = self.copy()
        for var, typ in other.var_types.items():
            if var not in merged.var_types:
                merged.var_types[var] = typ
        for var, preds in other.predicates.items():
            merged.predicates.setdefault(var, []).extend(
                p for p in preds if p not in merged.predicates.get(var, []))
        for var, ts in other.taint.items():
            if var in merged.taint:
                cur = merged.taint[var]
                merged.taint[var] = TaintStatus(
                    source=ts.source or cur.source, sanitizers_applied=[],
                    is_safe=not (ts.tainted() or cur.tainted()))
            else:
                merged.taint[var] = ts
        return merged

# String method database

LIST_OF_STR = BaseTypeR(BaseTypeKind.LIST, type_args=(STR_TYPE,))
BYTES_TYPE = BaseTypeR(BaseTypeKind.OBJECT)

_PRESERVES_LEN = "preserves_len"
_REDUCES_LEN = "reduces_len"
_RETURNS_BOOL = "returns_bool"
_RETURNS_INT = "returns_int"
_RETURNS_LIST = "returns_list"
_RETURNS_STR = "returns_str"
_RETURNS_BYTES = "returns_bytes"

STRING_METHOD_DB: Dict[str, Dict[str, Any]] = {
    "startswith":  {"return": _RETURNS_BOOL, "guard": True, "prop": StringProperty.STARTSWITH},
    "endswith":    {"return": _RETURNS_BOOL, "guard": True, "prop": StringProperty.ENDSWITH},
    "isdigit":     {"return": _RETURNS_BOOL, "guard": True, "prop": StringProperty.ISDIGIT},
    "isalpha":     {"return": _RETURNS_BOOL, "guard": True, "prop": StringProperty.ISALPHA},
    "isalnum":     {"return": _RETURNS_BOOL, "guard": True, "prop": StringProperty.ISALNUM},
    "isupper":     {"return": _RETURNS_BOOL, "guard": True, "prop": StringProperty.ISUPPER},
    "islower":     {"return": _RETURNS_BOOL, "guard": True, "prop": StringProperty.ISLOWER},
    "isspace":     {"return": _RETURNS_BOOL, "guard": True, "prop": StringProperty.ISSPACE},
    "isidentifier": {"return": _RETURNS_BOOL, "guard": True, "prop": StringProperty.ISIDENTIFIER},
    "find": {"return": _RETURNS_INT}, "rfind": {"return": _RETURNS_INT},
    "index": {"return": _RETURNS_INT}, "rindex": {"return": _RETURNS_INT},
    "count": {"return": _RETURNS_INT},
    "upper": {"return": _RETURNS_STR, "effect": _PRESERVES_LEN},
    "lower": {"return": _RETURNS_STR, "effect": _PRESERVES_LEN},
    "title": {"return": _RETURNS_STR, "effect": _PRESERVES_LEN},
    "capitalize": {"return": _RETURNS_STR, "effect": _PRESERVES_LEN},
    "swapcase": {"return": _RETURNS_STR, "effect": _PRESERVES_LEN},
    "casefold": {"return": _RETURNS_STR, "effect": _PRESERVES_LEN},
    "strip": {"return": _RETURNS_STR, "effect": _REDUCES_LEN},
    "lstrip": {"return": _RETURNS_STR, "effect": _REDUCES_LEN},
    "rstrip": {"return": _RETURNS_STR, "effect": _REDUCES_LEN},
    "replace": {"return": _RETURNS_STR}, "join": {"return": _RETURNS_STR},
    "center": {"return": _RETURNS_STR}, "ljust": {"return": _RETURNS_STR},
    "rjust": {"return": _RETURNS_STR}, "zfill": {"return": _RETURNS_STR},
    "format": {"return": _RETURNS_STR}, "format_map": {"return": _RETURNS_STR},
    "split": {"return": _RETURNS_LIST}, "rsplit": {"return": _RETURNS_LIST},
    "splitlines": {"return": _RETURNS_LIST},
    "partition": {"return": _RETURNS_LIST}, "rpartition": {"return": _RETURNS_LIST},
    "encode": {"return": _RETURNS_BYTES},
}

# Main analyser

class StringRefinementAnalyzer:
    """Infers refined string types from Python AST nodes."""

    def __init__(self) -> None:
        self._format_parser = string.Formatter()

    def analyze_string_method(
        self, obj_var: str, method: str, args: List[ast.expr],
        state: AnalysisState,
    ) -> Tuple[RefType, AnalysisState]:
        """Model the return type of ``obj_var.method(*args)``."""
        info = STRING_METHOD_DB.get(method)
        if info is None:
            return RefType.trivial(ANY_TYPE), state

        new_state = state.copy()
        self._propagate_taint(method, [obj_var], new_state)
        result_var = f"_ret_{obj_var}_{method}"

        ret_kind = info["return"]
        if ret_kind == _RETURNS_BOOL:
            return RefType.trivial(BOOL_TYPE), new_state
        if ret_kind == _RETURNS_INT:
            return RefType.trivial(INT_TYPE), new_state
        if ret_kind == _RETURNS_BYTES:
            encoding = self._extract_const_str(args[0]) if args else "utf-8"
            return self._model_encode_decode(obj_var, method, encoding), new_state
        if ret_kind == _RETURNS_LIST:
            if method in ("split", "rsplit"):
                sep = self._extract_const_str(args[0]) if args else None
                maxsplit_val: Optional[int] = None
                if len(args) >= 2:
                    maxsplit_val = self._extract_const_int(args[1])
                return self._model_split(obj_var, sep, maxsplit_val), new_state
            return RefType.trivial(LIST_OF_STR), new_state

        if method == "strip" or method in ("lstrip", "rstrip"):
            chars = self._extract_const_str(args[0]) if args else None
            return self._model_strip(obj_var, chars), new_state
        if method == "replace":
            old_s = self._extract_const_str(args[0]) if len(args) > 0 else ""
            new_s = self._extract_const_str(args[1]) if len(args) > 1 else ""
            return self._model_replace(obj_var, old_s or "", new_s or ""), new_state
        if method == "join":
            items_var = self._name_of(args[0]) if args else "_items"
            return self._model_join(obj_var, items_var, new_state), new_state
        if method == "format":
            fmt_str = self._try_resolve_const(obj_var, state)
            if fmt_str is not None:
                kwargs: Dict[str, ast.expr] = {}
                return self.analyze_format_string(fmt_str, args, kwargs, state), new_state

        effect = info.get("effect")
        if effect == _PRESERVES_LEN:
            existing = state.get_type(obj_var)
            if existing and existing.base.kind == BaseTypeKind.STR:
                return RefType(result_var, STR_TYPE, existing.pred), new_state
        return RefType.trivial(STR_TYPE), new_state

    def analyze_fstring(self, node: ast.JoinedStr, state: AnalysisState) -> RefType:
        """Infer a refinement type for an f-string expression."""
        parts_info: List[Pred] = []
        total_min_len = 0
        taint_sources: List[str] = []

        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                total_min_len += len(value.value)
            elif isinstance(value, ast.FormattedValue):
                inner_var = self._name_of(value.value)
                taint_sources.append(inner_var)
                inner_type = state.get_type(inner_var)
                if inner_type and inner_type.base.kind == BaseTypeKind.STR:
                    for sp in state.string_props.get(inner_var, []):
                        parts_info.append(sp.to_pred(inner_var))

        binder = "_fstr"
        pred = Pred.len_ge(binder, total_min_len)
        for extra in parts_info:
            pred = pred.and_(extra)
        return RefType(binder, STR_TYPE, pred)

    def model_regex_match(
        self, pattern: str, string_var: str, state: AnalysisState,
    ) -> AnalysisState:
        """Update state after ``re.match(pattern, string_var)``."""
        info = RegexInfo.from_pattern(pattern)
        match_var = f"_match_{string_var}"
        new_state = state.copy()
        new_state.regex_matches[match_var] = info

        match_type = RefType(match_var, BaseTypeR(BaseTypeKind.OBJECT), Pred.true_())
        new_state.var_types[match_var] = match_type
        regex_pred = self._regex_to_refinement(pattern)
        if regex_pred is not None:
            new_state = new_state.add_pred(string_var, regex_pred)
            new_state = new_state.add_string_prop(
                string_var,
                StringPredicate(StringProperty.MATCHES_REGEX, pattern),
            )
        return new_state

    def model_regex_groups(
        self, match_var: str, pattern: str, state: AnalysisState,
    ) -> AnalysisState:
        """Refine state with types for ``match_var.group(i)``."""
        info = state.regex_matches.get(match_var)
        if info is None:
            info = RegexInfo.from_pattern(pattern)

        new_state = state.copy()
        new_state.var_types[f"{match_var}_g0"] = RefType.trivial(STR_TYPE)
        for i in range(1, info.num_groups + 1):
            gvar = f"{match_var}_g{i}"
            new_state.var_types[gvar] = RefType(
                gvar, ANY_TYPE,
                Pred.isinstance_(gvar, "str").or_(Pred.is_none(gvar)))

        for name, idx in info.named_groups.items():
            alias = f"{match_var}_{name}"
            source = f"{match_var}_g{idx}"
            if source in new_state.var_types:
                new_state.var_types[alias] = new_state.var_types[source]
        return new_state

    def track_taint(self, var: str, source: str, state: AnalysisState) -> AnalysisState:
        """Mark *var* as tainted from the given *source* label."""
        source_map: Dict[str, TaintSource] = {
            "input": TaintSource.USER_INPUT, "request": TaintSource.USER_INPUT,
            "stdin": TaintSource.USER_INPUT, "form": TaintSource.USER_INPUT,
            "query": TaintSource.USER_INPUT,
            "database": TaintSource.DATABASE, "db": TaintSource.DATABASE,
            "sql": TaintSource.DATABASE,
            "file": TaintSource.FILE, "open": TaintSource.FILE,
            "read": TaintSource.FILE,
            "network": TaintSource.NETWORK, "socket": TaintSource.NETWORK,
            "http": TaintSource.NETWORK,
            "env": TaintSource.ENVIRONMENT, "environ": TaintSource.ENVIRONMENT,
            "getenv": TaintSource.ENVIRONMENT,
        }
        taint_src = source_map.get(source.lower(), TaintSource.USER_INPUT)
        new_state = state.copy()
        new_state.taint[var] = TaintStatus(source=taint_src, is_safe=False)
        return new_state

    def check_sanitized(self, var: str, state: AnalysisState) -> bool:
        """Return ``True`` if *var*'s taint has been neutralised."""
        ts = state.taint.get(var)
        if ts is None:
            return True
        return ts.is_safe

    def analyze_format_string(
        self, format_str: str, args: List[ast.expr],
        kwargs: Dict[str, ast.expr], state: AnalysisState,
    ) -> RefType:
        """Infer type for ``format_str.format(*args, **kwargs)``."""
        keys = self._extract_format_keys(format_str)
        binder = "_fmt"

        literal_len = 0
        last_end = 0
        for m in re.finditer(r"\{[^}]*\}", format_str):
            literal_len += m.start() - last_end
            last_end = m.end()
        literal_len += len(format_str) - last_end

        pred = Pred.len_ge(binder, literal_len)

        taint_sources: List[str] = []
        for a in args:
            taint_sources.append(self._name_of(a))
        for v in kwargs.values():
            taint_sources.append(self._name_of(v))

        return RefType(binder, STR_TYPE, pred)

    def analyze_string_guard(
        self, guard: ast.expr, var: str, state: AnalysisState,
    ) -> Tuple[AnalysisState, AnalysisState]:
        """Split abstract state on a string-predicate guard."""
        true_state = state.copy()
        false_state = state.copy()

        if isinstance(guard, ast.Call) and isinstance(guard.func, ast.Attribute):
            method = guard.func.attr
            info = STRING_METHOD_DB.get(method)
            if info and info.get("guard"):
                prop_kind = info["prop"]
                arg_str: Optional[str] = None
                if guard.args:
                    arg_str = self._extract_const_str(guard.args[0])

                pos_prop = StringPredicate(prop_kind, argument=arg_str, negated=False)
                neg_prop = StringPredicate(prop_kind, argument=arg_str, negated=True)

                true_state = true_state.add_string_prop(var, pos_prop)
                true_state = true_state.add_pred(var, pos_prop.to_pred(var))

                false_state = false_state.add_string_prop(var, neg_prop)
                false_state = false_state.add_pred(var, neg_prop.to_pred(var))

                # Refine the variable type in the true branch
                existing = state.get_type(var)
                if existing and existing.base.kind == BaseTypeKind.STR:
                    refined_pred = existing.pred.and_(pos_prop.to_pred(var))
                    true_state.var_types[var] = RefType(
                        existing.binder, STR_TYPE, refined_pred,
                    )

        elif isinstance(guard, ast.Compare):
            if (len(guard.ops) == 1 and isinstance(guard.left, ast.Call)
                    and isinstance(guard.left.func, ast.Name)
                    and guard.left.func.id == "len" and guard.left.args):
                len_arg = self._name_of(guard.left.args[0])
                if len_arg == var and isinstance(guard.comparators[0], ast.Constant):
                    n = guard.comparators[0].value
                    if isinstance(n, int):
                        op = guard.ops[0]
                        t_pred, f_pred = self._len_compare_preds(var, op, n)
                        true_state = true_state.add_pred(var, t_pred)
                        false_state = false_state.add_pred(var, f_pred)

        return true_state, false_state

    def _model_split(self, var: str, sep: Optional[str],
                     maxsplit: Optional[int]) -> RefType:
        """``str.split`` always returns a list with ≥ 1 element."""
        binder = "_split"
        min_len = 1
        pred = Pred.len_ge(binder, min_len)
        if maxsplit is not None and maxsplit >= 0:
            pred = pred.and_(Pred.len_ge(binder, min_len))
        return RefType(binder, LIST_OF_STR, pred)

    def _model_strip(self, var: str, chars: Optional[str]) -> RefType:
        """``str.strip`` returns a string no longer than the input."""
        binder = "_strip"
        pred = Pred.len_ge(binder, 0)
        return RefType(binder, STR_TYPE, pred)

    def _model_replace(self, var: str, old: str, new: str) -> RefType:
        """Model ``str.replace(old, new)``."""
        binder = "_repl"
        if not old:
            # Replacing empty string — result length changes
            return RefType.trivial(STR_TYPE)
        pred = Pred.len_ge(binder, 0)
        if old == new:
            # No-op replace preserves everything
            pred = Pred.true_()
        return RefType(binder, STR_TYPE, pred)

    def _model_join(self, sep_var: str, items_var: str,
                    state: AnalysisState) -> RefType:
        """Model ``sep.join(items)``."""
        binder = "_join"
        pred = Pred.len_ge(binder, 0)

        items_type = state.get_type(items_var)
        if items_type is not None:
            for p in state.predicates.get(items_var, []):
                if p.op == PredOp.LEN_EQ:
                    n_items = p.args[1]
                    if isinstance(n_items, int) and n_items > 0:
                        # Result length ≥ n_items - 1 (separators only)
                        sep_type = state.get_type(sep_var)
                        pred = Pred.len_ge(binder, n_items - 1)
        return RefType(binder, STR_TYPE, pred)

    def _model_encode_decode(self, var: str, method: str,
                             encoding: str) -> RefType:
        """Model ``str.encode`` → bytes and ``bytes.decode`` → str."""
        binder = "_codec"
        if method == "encode":
            return RefType(binder, BYTES_TYPE, Pred.true_())
        return RefType(binder, STR_TYPE, Pred.len_ge(binder, 0))

    def _extract_format_keys(self, format_str: str) -> List[str]:
        """Extract field names from a ``str.format`` / f-string template."""
        keys: List[str] = []
        try:
            for _, field_name, _, _ in self._format_parser.parse(format_str):
                if field_name is not None:
                    # field_name may be '' for positional, or 'name' for keyword
                    base = field_name.split(".")[0].split("[")[0]
                    keys.append(base)
        except (ValueError, KeyError):
            pass
        return keys

    def _regex_to_refinement(self, pattern: str) -> Optional[Pred]:
        """Convert simple regex patterns to refinement predicates."""
        var = "_rx"

        m = re.fullmatch(r"\^\.\\{(\d+)\\}\$", pattern)
        if m:
            return Pred.len_eq(var, int(m.group(1)))

        if pattern in (r"^\d+$", r"^[0-9]+$"):
            return Pred.hasattr_(var, "__isdigit__").and_(Pred.len_ge(var, 1))

        if pattern in (r"^[a-zA-Z]+$", r"^[A-Za-z]+$"):
            return Pred.hasattr_(var, "__isalpha__").and_(Pred.len_ge(var, 1))

        if pattern.startswith("^https?://") or pattern.startswith("^http"):
            return Pred.hasattr_(var, "__startswith_http")

        if re.fullmatch(r"\\d\{3\}-\\d\{2\}-\\d\{4\}", pattern):
            return Pred.len_eq(var, 11).and_(
                Pred.hasattr_(var, f"__matches_{pattern}")
            )

        total_len = self._compute_fixed_length(pattern)
        if total_len is not None:
            return Pred.len_eq(var, total_len)

        return None

    def _compute_fixed_length(self, pattern: str) -> Optional[int]:
        """Compute fixed length if regex matches only fixed-width strings."""
        stripped = pattern.lstrip("^").rstrip("$")
        if not stripped or any(c in stripped for c in "*+?|()"):
            return None
        length, i = 0, 0
        while i < len(stripped):
            ch = stripped[i]
            if ch == "\\":
                i += 2
                if i > len(stripped):
                    return None
            elif ch == "[":
                close = stripped.find("]", i)
                if close == -1:
                    return None
                i = close + 1
            elif ch == ".":
                i += 1
            else:
                i += 1
            # Check for quantifier {N}
            if i < len(stripped) and stripped[i:i+1] == "{":
                close = stripped.find("}", i)
                if close == -1:
                    return None
                quant = stripped[i + 1:close]
                if not quant.isdigit():
                    return None
                length += int(quant)
                i = close + 1
            else:
                length += 1
        return length if length > 0 else None

    def _propagate_taint(self, op: str, sources: List[str],
                         state: AnalysisState) -> TaintStatus:
        """Compute taint for a result derived from *sources* via *op*."""
        combined_source: Optional[TaintSource] = None
        any_unsafe = False
        sanitizers: List[str] = []
        for src in sources:
            ts = state.taint.get(src)
            if ts is not None and ts.source is not None:
                combined_source = ts.source
                if not ts.is_safe:
                    any_unsafe = True
                sanitizers.extend(ts.sanitizers_applied)
        if combined_source is None:
            return TaintStatus(is_safe=True)
        sanitising_ops = {"escape", "quote", "sanitize", "clean", "encode",
                          "html_escape", "urlencode", "parameterize"}
        is_safe = not any_unsafe or op.lower() in sanitising_ops
        if op.lower() in sanitising_ops:
            sanitizers.append(op)
        return TaintStatus(combined_source, sanitizers, is_safe)

    @staticmethod
    def _extract_const_str(node: ast.expr) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    @staticmethod
    def _extract_const_int(node: ast.expr) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        return None

    @staticmethod
    def _name_of(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{StringRefinementAnalyzer._name_of(node.value)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            return f"{StringRefinementAnalyzer._name_of(node.value)}[]"
        return "_expr"

    @staticmethod
    def _try_resolve_const(var: str, state: AnalysisState) -> Optional[str]:
        typ = state.get_type(var)
        if typ is None:
            return None
        for sp in state.string_props.get(var, []):
            if sp.prop == StringProperty.MATCHES_REGEX and sp.argument:
                return None
        return None

    @staticmethod
    def _len_compare_preds(var: str, op: ast.cmpop, n: int) -> Tuple[Pred, Pred]:
        """Produce true/false branch predicates for ``len(var) <op> n``."""
        if isinstance(op, ast.Gt):
            return Pred.len_gt(var, n), Pred.len_ge(var, 0).and_(Pred.var_le(f"len({var})", n))
        if isinstance(op, ast.GtE):
            return Pred.len_ge(var, n), Pred.var_lt(f"len({var})", n)
        if isinstance(op, ast.Lt):
            return Pred.var_lt(f"len({var})", n), Pred.len_ge(var, n)
        if isinstance(op, ast.LtE):
            return Pred.var_le(f"len({var})", n), Pred.len_gt(var, n)
        if isinstance(op, ast.Eq):
            return Pred.len_eq(var, n), Pred.len_eq(var, n).not_()
        if isinstance(op, ast.NotEq):
            return Pred.len_eq(var, n).not_(), Pred.len_eq(var, n)
        return Pred.true_(), Pred.true_()
