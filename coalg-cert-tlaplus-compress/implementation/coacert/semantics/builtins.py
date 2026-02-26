"""
Built-in TLA+ operators and standard modules.

Registers implementations of the standard TLA+ modules:
* ``Naturals`` – arithmetic on non-negative integers.
* ``Integers`` – arithmetic on all integers.
* ``FiniteSets`` – ``IsFiniteSet``, ``Cardinality``.
* ``Sequences`` – ``Seq``, ``Len``, ``Head``, ``Tail``, ``Append``, etc.
* ``TLC`` – ``Print``, ``Assert``, ``JavaTime`` (stub).
* ``Bags`` – basic bag operations.

All operators are registered through a ``ModuleRegistry`` which the
``Environment`` can import.
"""

from __future__ import annotations

import itertools
import sys
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .values import (
    TLAValue,
    TLAValueError,
    IntValue,
    BoolValue,
    StringValue,
    SetValue,
    FunctionValue,
    TupleValue,
    RecordValue,
    SequenceValue,
    ModelValue,
)
from .environment import Environment, BuiltinEntry


# ===================================================================
# Module registry
# ===================================================================

class ModuleRegistry:
    """Central registry of all built-in TLA+ modules and their operators.

    Usage::

        registry = ModuleRegistry()
        registry.register_all()
        registry.install(env)           # install into an Environment
        registry.install(env, "Naturals")  # install only one module
    """

    def __init__(self) -> None:
        self._modules: Dict[str, List[BuiltinEntry]] = {}

    def add(self, module: str, name: str, arity: int,
            fn: Callable[..., TLAValue], *, is_lazy: bool = False) -> None:
        entry = BuiltinEntry(
            name=name, module=module, arity=arity,
            evaluator=fn, is_lazy=is_lazy,
        )
        self._modules.setdefault(module, []).append(entry)

    def install(self, env: Environment, module_name: str | None = None) -> None:
        """Install built-in operators into *env*.

        If *module_name* is given, install only that module; otherwise
        install all registered modules.
        """
        if module_name is not None:
            entries = self._modules.get(module_name, [])
            for entry in entries:
                env.register_builtin(entry)
            env.import_module(module_name)
        else:
            for mod, entries in self._modules.items():
                for entry in entries:
                    env.register_builtin(entry)
                env.import_module(mod)

    @property
    def module_names(self) -> List[str]:
        return sorted(self._modules.keys())

    def operators_in(self, module: str) -> List[str]:
        return [e.name for e in self._modules.get(module, [])]

    def register_all(self) -> None:
        """Register all standard module operators."""
        _register_naturals(self)
        _register_integers(self)
        _register_finitesets(self)
        _register_sequences(self)
        _register_tlc(self)
        _register_bags(self)


# ===================================================================
# Naturals module
# ===================================================================

# Upper bound for Nat to keep model checking finite.
_NAT_BOUND = 1000

def _register_naturals(reg: ModuleRegistry) -> None:
    mod = "Naturals"

    def nat_plus(a: TLAValue, b: TLAValue) -> TLAValue:
        return IntValue(_require_int(a, "+") + _require_int(b, "+"))

    def nat_minus(a: TLAValue, b: TLAValue) -> TLAValue:
        return IntValue(_require_int(a, "-") - _require_int(b, "-"))

    def nat_times(a: TLAValue, b: TLAValue) -> TLAValue:
        return IntValue(_require_int(a, "*") * _require_int(b, "*"))

    def nat_div(a: TLAValue, b: TLAValue) -> TLAValue:
        bv = _require_int(b, "\\div")
        if bv == 0:
            raise TLAValueError("Division by zero in Naturals!\\div")
        av = _require_int(a, "\\div")
        # TLA+ integer division: truncation toward zero
        if (av < 0) != (bv < 0) and av % bv != 0:
            return IntValue(av // bv + 1)
        return IntValue(av // bv)

    def nat_mod(a: TLAValue, b: TLAValue) -> TLAValue:
        bv = _require_int(b, "%")
        if bv == 0:
            raise TLAValueError("Modulo by zero in Naturals!%")
        return IntValue(_require_int(a, "%") % bv)

    def nat_dotdot(a: TLAValue, b: TLAValue) -> TLAValue:
        lo = _require_int(a, "..")
        hi = _require_int(b, "..")
        return SetValue(IntValue(i) for i in range(lo, hi + 1))

    def nat_leq(a: TLAValue, b: TLAValue) -> TLAValue:
        return BoolValue(_require_int(a, "<=") <= _require_int(b, "<="))

    def nat_geq(a: TLAValue, b: TLAValue) -> TLAValue:
        return BoolValue(_require_int(a, ">=") >= _require_int(b, ">="))

    def nat_lt(a: TLAValue, b: TLAValue) -> TLAValue:
        return BoolValue(_require_int(a, "<") < _require_int(b, "<"))

    def nat_gt(a: TLAValue, b: TLAValue) -> TLAValue:
        return BoolValue(_require_int(a, ">") > _require_int(b, ">"))

    def nat_set() -> TLAValue:
        """Nat as a bounded set for model checking."""
        return SetValue(IntValue(i) for i in range(0, _NAT_BOUND + 1))

    def nat_exp(a: TLAValue, b: TLAValue) -> TLAValue:
        base = _require_int(a, "^")
        exp = _require_int(b, "^")
        if exp < 0:
            raise TLAValueError("Negative exponent in Naturals!^")
        return IntValue(base ** exp)

    reg.add(mod, "+", 2, nat_plus)
    reg.add(mod, "-", 2, nat_minus)
    reg.add(mod, "*", 2, nat_times)
    reg.add(mod, "\\div", 2, nat_div)
    reg.add(mod, "%", 2, nat_mod)
    reg.add(mod, "..", 2, nat_dotdot)
    reg.add(mod, "\\leq", 2, nat_leq)
    reg.add(mod, "\\geq", 2, nat_geq)
    reg.add(mod, "<", 2, nat_lt)
    reg.add(mod, ">", 2, nat_gt)
    reg.add(mod, "^", 2, nat_exp)
    reg.add(mod, "Nat", 0, nat_set)


# ===================================================================
# Integers module
# ===================================================================

_INT_BOUND = 1000

def _register_integers(reg: ModuleRegistry) -> None:
    mod = "Integers"

    def int_neg(a: TLAValue) -> TLAValue:
        return IntValue(-_require_int(a, "unary -"))

    def int_set() -> TLAValue:
        """Int as a bounded set for model checking."""
        return SetValue(IntValue(i) for i in range(-_INT_BOUND, _INT_BOUND + 1))

    def int_abs(a: TLAValue) -> TLAValue:
        return IntValue(abs(_require_int(a, "abs")))

    reg.add(mod, "-.", 1, int_neg)
    reg.add(mod, "Int", 0, int_set)
    reg.add(mod, "abs", 1, int_abs)


# ===================================================================
# FiniteSets module
# ===================================================================

def _register_finitesets(reg: ModuleRegistry) -> None:
    mod = "FiniteSets"

    def is_finite_set(s: TLAValue) -> TLAValue:
        return BoolValue(isinstance(s, SetValue))

    def cardinality(s: TLAValue) -> TLAValue:
        if not isinstance(s, SetValue):
            raise TLAValueError(f"Cardinality requires a set, got {type(s).__name__}")
        return IntValue(s.cardinality())

    reg.add(mod, "IsFiniteSet", 1, is_finite_set)
    reg.add(mod, "Cardinality", 1, cardinality)


# ===================================================================
# Sequences module
# ===================================================================

_SEQ_MAX_LEN = 10

def _register_sequences(reg: ModuleRegistry) -> None:
    mod = "Sequences"

    def seq_set(s: TLAValue) -> TLAValue:
        """Seq(S): the set of all finite sequences over S (bounded)."""
        if not isinstance(s, SetValue):
            raise TLAValueError(f"Seq requires a set, got {type(s).__name__}")
        elems = sorted(s)
        result: List[TLAValue] = [SequenceValue()]  # empty sequence
        for length in range(1, _SEQ_MAX_LEN + 1):
            for combo in itertools.product(elems, repeat=length):
                result.append(SequenceValue(combo))
        return SetValue(result)

    def seq_len(s: TLAValue) -> TLAValue:
        seq = _require_seq(s, "Len")
        return IntValue(seq.length())

    def seq_head(s: TLAValue) -> TLAValue:
        return _require_seq(s, "Head").head()

    def seq_tail(s: TLAValue) -> TLAValue:
        return _require_seq(s, "Tail").tail()

    def seq_append(s: TLAValue, e: TLAValue) -> TLAValue:
        return _require_seq(s, "Append").append(e)

    def seq_concat(s: TLAValue, t: TLAValue) -> TLAValue:
        return _require_seq(s, "\\o left").concat(_require_seq(t, "\\o right"))

    def seq_subseq(s: TLAValue, m: TLAValue, n: TLAValue) -> TLAValue:
        seq = _require_seq(s, "SubSeq")
        mv = _require_int(m, "SubSeq start")
        nv = _require_int(n, "SubSeq end")
        return seq.sub_seq(mv, nv)

    def seq_selectseq(s: TLAValue, test: TLAValue) -> TLAValue:
        """SelectSeq when both arguments are pre-evaluated.

        The test argument must be a FunctionValue acting as a predicate.
        """
        seq = _require_seq(s, "SelectSeq")
        if isinstance(test, FunctionValue):
            def pred(e: TLAValue) -> bool:
                result = test.apply(e)
                if not isinstance(result, BoolValue):
                    raise TLAValueError("SelectSeq predicate must return BOOLEAN")
                return result.val
            return seq.select_seq(pred)
        raise TLAValueError(f"SelectSeq test must be a function, got {type(test).__name__}")

    def seq_reverse(s: TLAValue) -> TLAValue:
        seq = _require_seq(s, "Reverse")
        return SequenceValue(reversed(list(seq.elements)))

    reg.add(mod, "Seq", 1, seq_set)
    reg.add(mod, "Len", 1, seq_len)
    reg.add(mod, "Head", 1, seq_head)
    reg.add(mod, "Tail", 1, seq_tail)
    reg.add(mod, "Append", 2, seq_append)
    reg.add(mod, "\\o", 2, seq_concat)
    reg.add(mod, "Concat", 2, seq_concat)
    reg.add(mod, "SubSeq", 3, seq_subseq)
    reg.add(mod, "SelectSeq", 2, seq_selectseq)
    reg.add(mod, "Reverse", 1, seq_reverse)


# ===================================================================
# TLC module
# ===================================================================

def _register_tlc(reg: ModuleRegistry) -> None:
    mod = "TLC"

    def tlc_print(out: TLAValue, val: TLAValue) -> TLAValue:
        """Print(out, val): write *out* to stderr and return *val*."""
        sys.stderr.write(f"TLC!Print: {out.pretty()}\n")
        return val

    def tlc_print_t(val: TLAValue) -> TLAValue:
        """PrintT(val): write *val* to stderr and return TRUE."""
        sys.stderr.write(f"TLC!PrintT: {val.pretty()}\n")
        return BoolValue(True)

    def tlc_assert(cond: TLAValue, msg: TLAValue) -> TLAValue:
        if not isinstance(cond, BoolValue):
            raise TLAValueError(f"Assert condition must be BOOLEAN, got {type(cond).__name__}")
        if not cond.val:
            msg_str = msg.pretty() if not isinstance(msg, StringValue) else msg.val
            raise TLAValueError(f"TLC!Assert failed: {msg_str}")
        return BoolValue(True)

    def tlc_java_time() -> TLAValue:
        """JavaTime: return current epoch milliseconds as IntValue."""
        return IntValue(int(time.time() * 1000))

    def tlc_any() -> TLAValue:
        """Any: placeholder that model-checks as the set of model values."""
        return ModelValue("ANY")

    def tlc_permutations(s: TLAValue) -> TLAValue:
        """Permutations(S): the set of all permutations of set S."""
        if not isinstance(s, SetValue):
            raise TLAValueError(f"Permutations requires a set, got {type(s).__name__}")
        elems = sorted(s)
        perms: List[TLAValue] = []
        for perm in itertools.permutations(elems):
            mapping: Dict[TLAValue, TLAValue] = {}
            for orig, target in zip(elems, perm):
                mapping[orig] = target
            perms.append(FunctionValue(mapping))
        return SetValue(perms)

    def tlc_sort_seq(s: TLAValue) -> TLAValue:
        """SortSeq: sort a sequence of comparable values."""
        seq = _require_seq(s, "SortSeq")
        return SequenceValue(sorted(seq.elements))

    def tlc_to_string(val: TLAValue) -> TLAValue:
        """ToString(val): convert value to string representation."""
        return StringValue(val.pretty())

    reg.add(mod, "Print", 2, tlc_print)
    reg.add(mod, "PrintT", 1, tlc_print_t)
    reg.add(mod, "Assert", 2, tlc_assert)
    reg.add(mod, "JavaTime", 0, tlc_java_time)
    reg.add(mod, "Any", 0, tlc_any)
    reg.add(mod, "Permutations", 1, tlc_permutations)
    reg.add(mod, "SortSeq", 1, tlc_sort_seq)
    reg.add(mod, "ToString", 1, tlc_to_string)


# ===================================================================
# Bags module (basic)
# ===================================================================

def _register_bags(reg: ModuleRegistry) -> None:
    mod = "Bags"

    def empty_bag() -> TLAValue:
        """EmptyBag == [x \\in {} |-> 0]."""
        return FunctionValue()

    def set_to_bag(s: TLAValue) -> TLAValue:
        """SetToBag(S): each element has count 1."""
        if not isinstance(s, SetValue):
            raise TLAValueError(f"SetToBag requires a set, got {type(s).__name__}")
        mapping = {elem: IntValue(1) for elem in s}
        return FunctionValue(mapping)

    def bag_to_set(b: TLAValue) -> TLAValue:
        """BagToSet(B): domain of the bag (elements with count > 0)."""
        if not isinstance(b, FunctionValue):
            raise TLAValueError(f"BagToSet requires a bag (function), got {type(b).__name__}")
        elems: List[TLAValue] = []
        for k, v in b.mapping.items():
            if isinstance(v, IntValue) and v.val > 0:
                elems.append(k)
        return SetValue(elems)

    def bag_in(e: TLAValue, b: TLAValue) -> TLAValue:
        """BagIn(e, B): TRUE iff e is in the bag with count > 0."""
        if not isinstance(b, FunctionValue):
            raise TLAValueError(f"BagIn requires a bag (function), got {type(b).__name__}")
        try:
            count = b.apply(e)
            if isinstance(count, IntValue):
                return BoolValue(count.val > 0)
            return BoolValue(False)
        except TLAValueError:
            return BoolValue(False)

    def copies_in(e: TLAValue, b: TLAValue) -> TLAValue:
        """CopiesIn(e, B): the number of copies of e in bag B."""
        if not isinstance(b, FunctionValue):
            raise TLAValueError(f"CopiesIn requires a bag (function), got {type(b).__name__}")
        try:
            count = b.apply(e)
            if isinstance(count, IntValue):
                return count
            return IntValue(0)
        except TLAValueError:
            return IntValue(0)

    def bag_union(a: TLAValue, b: TLAValue) -> TLAValue:
        """BagUnion(A, B): add counts element-wise."""
        fa = _require_func(a, "BagUnion")
        fb = _require_func(b, "BagUnion")
        merged: Dict[TLAValue, TLAValue] = {}
        all_keys = set(fa.mapping.keys()) | set(fb.mapping.keys())
        for k in all_keys:
            ca = _bag_count(fa, k)
            cb = _bag_count(fb, k)
            merged[k] = IntValue(ca + cb)
        return FunctionValue(merged)

    def bag_diff(a: TLAValue, b: TLAValue) -> TLAValue:
        """BagDifference(A, B): subtract counts (floor at 0)."""
        fa = _require_func(a, "(-)")
        fb = _require_func(b, "(-)")
        result: Dict[TLAValue, TLAValue] = {}
        for k, v in fa.mapping.items():
            ca = _require_int_val(v)
            cb = _bag_count(fb, k)
            diff = max(0, ca - cb)
            if diff > 0:
                result[k] = IntValue(diff)
        return FunctionValue(result)

    def is_a_bag(b: TLAValue) -> TLAValue:
        """IsABag(B): TRUE iff B is a function to positive integers."""
        if not isinstance(b, FunctionValue):
            return BoolValue(False)
        for v in b.mapping.values():
            if not isinstance(v, IntValue) or v.val < 0:
                return BoolValue(False)
        return BoolValue(True)

    def bag_cardinality(b: TLAValue) -> TLAValue:
        """BagCardinality(B): sum of all counts."""
        if not isinstance(b, FunctionValue):
            raise TLAValueError(f"BagCardinality requires a bag, got {type(b).__name__}")
        total = 0
        for v in b.mapping.values():
            if isinstance(v, IntValue):
                total += v.val
        return IntValue(total)

    reg.add(mod, "EmptyBag", 0, empty_bag)
    reg.add(mod, "SetToBag", 1, set_to_bag)
    reg.add(mod, "BagToSet", 1, bag_to_set)
    reg.add(mod, "BagIn", 2, bag_in)
    reg.add(mod, "CopiesIn", 2, copies_in)
    reg.add(mod, "(+)", 2, bag_union)
    reg.add(mod, "(-)", 2, bag_diff)
    reg.add(mod, "IsABag", 1, is_a_bag)
    reg.add(mod, "BagCardinality", 1, bag_cardinality)


# ===================================================================
# Helpers
# ===================================================================

def _require_int(v: TLAValue, ctx: str) -> int:
    if not isinstance(v, IntValue):
        raise TLAValueError(f"{ctx}: expected integer, got {type(v).__name__} ({v.pretty()})")
    return v.val

def _require_int_val(v: TLAValue) -> int:
    if isinstance(v, IntValue):
        return v.val
    return 0

def _require_seq(v: TLAValue, ctx: str) -> SequenceValue:
    if isinstance(v, SequenceValue):
        return v
    if isinstance(v, TupleValue):
        return SequenceValue(v.elements)
    raise TLAValueError(f"{ctx}: expected sequence, got {type(v).__name__} ({v.pretty()})")

def _require_func(v: TLAValue, ctx: str) -> FunctionValue:
    if not isinstance(v, FunctionValue):
        raise TLAValueError(f"{ctx}: expected function, got {type(v).__name__} ({v.pretty()})")
    return v

def _bag_count(bag: FunctionValue, key: TLAValue) -> int:
    try:
        v = bag.apply(key)
        return _require_int_val(v)
    except TLAValueError:
        return 0


# ===================================================================
# Default registry singleton
# ===================================================================

_default_registry: Optional[ModuleRegistry] = None

def get_default_registry() -> ModuleRegistry:
    """Return the default module registry (lazily created, singleton)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ModuleRegistry()
        _default_registry.register_all()
    return _default_registry


def install_standard_modules(env: Environment) -> None:
    """Install all standard TLA+ modules into *env*."""
    get_default_registry().install(env)


def install_module(env: Environment, module_name: str) -> None:
    """Install a specific standard module into *env*."""
    get_default_registry().install(env, module_name)
