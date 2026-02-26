"""Comprehensive test suite for coacert.semantics."""

import pytest

from coacert.semantics.values import (
    IntValue,
    BoolValue,
    StringValue,
    ModelValue,
    SetValue,
    FunctionValue,
    TupleValue,
    RecordValue,
    SequenceValue,
    TLAValue,
    TLAValueError,
    value_from_python,
    values_to_json_string,
    value_from_json_string,
)
from coacert.semantics.state import TLAState, StateSignature, StateSpace
from coacert.semantics.environment import (
    Environment,
    OpDef,
    ConstantDecl,
    BuiltinEntry,
)
from coacert.semantics.evaluator import (
    Expr,
    ExprKind,
    EvalError,
    evaluate,
    int_lit,
    bool_lit,
    string_lit,
    name_ref,
    primed_ref,
    unary_op,
    binary_op,
    if_then_else,
    let_in,
    case_expr,
    quant_forall,
    quant_exists,
    choose,
    set_enum,
    set_comp,
    set_filter,
    func_construct,
    func_apply,
    func_except,
    domain_op,
    tuple_construct,
    record_construct,
    record_access,
    record_except,
    unchanged,
)
from coacert.semantics.actions import (
    ActionEvaluator,
    ActionExpr,
    action_conj,
    action_disj,
    action_exists,
    action_unchanged,
    action_from_expr,
    compute_successors,
    compute_initial_states,
    detect_stuttering,
)
from coacert.semantics.builtins import (
    ModuleRegistry,
    install_standard_modules,
    get_default_registry,
)
from coacert.semantics.type_system import (
    TLAType,
    TypeKind,
    TypeChecker,
    ANY_TYPE,
    BOOL_TYPE,
    INT_TYPE,
    STRING_TYPE,
    set_type,
    function_type,
    tuple_type,
    record_type,
    sequence_type,
    model_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_with_stdlib() -> Environment:
    """Return a fresh environment with standard modules installed."""
    env = Environment()
    install_standard_modules(env)
    return env


def _eval(expr: Expr, env=None, state=None) -> TLAValue:
    if env is None:
        env = _env_with_stdlib()
    if state is None:
        state = TLAState()
    return evaluate(expr, env, state)


# ===================================================================
# 1. TestIntValue
# ===================================================================

class TestIntValue:

    def test_creation(self):
        v = IntValue(42)
        assert v.val == 42

    @pytest.mark.parametrize("a,b,expected", [
        (3, 5, 8),
        (-1, 1, 0),
        (0, 0, 0),
    ])
    def test_addition_via_eval(self, a, b, expected):
        result = _eval(binary_op("+", int_lit(a), int_lit(b)))
        assert result == IntValue(expected)

    @pytest.mark.parametrize("a,b,expected", [
        (10, 3, 7),
        (0, 5, -5),
    ])
    def test_subtraction_via_eval(self, a, b, expected):
        result = _eval(binary_op("-", int_lit(a), int_lit(b)))
        assert result == IntValue(expected)

    @pytest.mark.parametrize("a,b,expected", [
        (4, 5, 20),
        (0, 100, 0),
        (-3, 7, -21),
    ])
    def test_multiplication_via_eval(self, a, b, expected):
        result = _eval(binary_op("*", int_lit(a), int_lit(b)))
        assert result == IntValue(expected)

    def test_equality(self):
        assert IntValue(7) == IntValue(7)
        assert IntValue(7) != IntValue(8)

    def test_hash_consistency(self):
        a, b = IntValue(99), IntValue(99)
        assert hash(a) == hash(b)
        assert {a, b} == {IntValue(99)}

    def test_ordering(self):
        assert IntValue(1) < IntValue(2)
        assert IntValue(5) >= IntValue(5)

    def test_sort_key(self):
        vals = [IntValue(3), IntValue(1), IntValue(2)]
        assert sorted(vals) == [IntValue(1), IntValue(2), IntValue(3)]


# ===================================================================
# 2. TestBoolValue
# ===================================================================

class TestBoolValue:

    @pytest.mark.parametrize("val", [True, False])
    def test_creation(self, val):
        assert BoolValue(val).val is val

    def test_equality(self):
        assert BoolValue(True) == BoolValue(True)
        assert BoolValue(True) != BoolValue(False)

    def test_hash(self):
        assert hash(BoolValue(True)) == hash(BoolValue(True))

    @pytest.mark.parametrize("a,b,expected", [
        (True, True, True),
        (True, False, False),
        (False, False, False),
    ])
    def test_conjunction_via_eval(self, a, b, expected):
        result = _eval(binary_op("/\\", bool_lit(a), bool_lit(b)))
        assert result == BoolValue(expected)

    @pytest.mark.parametrize("a,b,expected", [
        (True, False, True),
        (False, False, False),
    ])
    def test_disjunction_via_eval(self, a, b, expected):
        result = _eval(binary_op("\\/", bool_lit(a), bool_lit(b)))
        assert result == BoolValue(expected)

    def test_negation_via_eval(self):
        assert _eval(unary_op("~", bool_lit(True))) == BoolValue(False)
        assert _eval(unary_op("~", bool_lit(False))) == BoolValue(True)

    def test_implication_via_eval(self):
        result = _eval(binary_op("=>", bool_lit(False), bool_lit(False)))
        assert result == BoolValue(True)


# ===================================================================
# 3. TestSetValue
# ===================================================================

class TestSetValue:

    def test_empty_set(self):
        s = SetValue()
        assert s.cardinality() == 0
        assert len(s) == 0

    def test_creation_from_elements(self):
        s = SetValue([IntValue(1), IntValue(2), IntValue(3)])
        assert s.cardinality() == 3

    def test_contains(self):
        s = SetValue([IntValue(1), IntValue(2)])
        assert s.contains(IntValue(1))
        assert not s.contains(IntValue(3))

    def test_membership_via_in_operator(self):
        s = SetValue([IntValue(10)])
        assert IntValue(10) in s

    def test_union(self):
        a = SetValue([IntValue(1), IntValue(2)])
        b = SetValue([IntValue(2), IntValue(3)])
        u = a.union(b)
        assert u.cardinality() == 3
        assert u.contains(IntValue(3))

    def test_intersection(self):
        a = SetValue([IntValue(1), IntValue(2), IntValue(3)])
        b = SetValue([IntValue(2), IntValue(3), IntValue(4)])
        i = a.intersect(b)
        assert i == SetValue([IntValue(2), IntValue(3)])

    def test_difference(self):
        a = SetValue([IntValue(1), IntValue(2), IntValue(3)])
        b = SetValue([IntValue(2)])
        d = a.difference(b)
        assert d == SetValue([IntValue(1), IntValue(3)])

    def test_subset(self):
        a = SetValue([IntValue(1)])
        b = SetValue([IntValue(1), IntValue(2)])
        assert a.is_subset(b)
        assert not b.is_subset(a)

    def test_equality_ignores_order(self):
        a = SetValue([IntValue(2), IntValue(1)])
        b = SetValue([IntValue(1), IntValue(2)])
        assert a == b

    def test_hash_equal_sets(self):
        a = SetValue([IntValue(1), IntValue(2)])
        b = SetValue([IntValue(2), IntValue(1)])
        assert hash(a) == hash(b)

    def test_powerset(self):
        s = SetValue([IntValue(1), IntValue(2)])
        ps = s.powerset()
        assert ps.cardinality() == 4  # 2^2

    def test_cardinality_no_duplicates(self):
        s = SetValue([IntValue(1), IntValue(1), IntValue(1)])
        assert s.cardinality() == 1

    @pytest.mark.parametrize("elems,card", [
        ([], 0),
        ([IntValue(5)], 1),
        ([IntValue(i) for i in range(5)], 5),
    ])
    def test_cardinality_parametrized(self, elems, card):
        assert SetValue(elems).cardinality() == card

    def test_iteration(self):
        s = SetValue([IntValue(1), IntValue(2)])
        assert set(s) == {IntValue(1), IntValue(2)}

    def test_cross_product(self):
        a = SetValue([IntValue(1), IntValue(2)])
        b = SetValue([BoolValue(True)])
        c = a.cross(b)
        assert c.cardinality() == 2

    def test_big_union(self):
        inner1 = SetValue([IntValue(1), IntValue(2)])
        inner2 = SetValue([IntValue(2), IntValue(3)])
        outer = SetValue([inner1, inner2])
        u = outer.big_union()
        assert u == SetValue([IntValue(1), IntValue(2), IntValue(3)])


# ===================================================================
# 4. TestFunctionValue
# ===================================================================

class TestFunctionValue:

    def test_construction_from_mapping(self):
        f = FunctionValue(mapping={IntValue(1): StringValue("a")})
        assert f.apply(IntValue(1)) == StringValue("a")

    def test_construction_from_pairs(self):
        f = FunctionValue(pairs=[(IntValue(1), IntValue(10)),
                                 (IntValue(2), IntValue(20))])
        assert f.apply(IntValue(2)) == IntValue(20)

    def test_domain(self):
        f = FunctionValue(mapping={IntValue(1): IntValue(10),
                                   IntValue(2): IntValue(20)})
        assert f.domain() == SetValue([IntValue(1), IntValue(2)])

    def test_range(self):
        f = FunctionValue(mapping={IntValue(1): IntValue(10)})
        assert f.range() == SetValue([IntValue(10)])

    def test_apply_missing_key(self):
        f = FunctionValue(mapping={IntValue(1): IntValue(10)})
        with pytest.raises((TLAValueError, KeyError)):
            f.apply(IntValue(99))

    def test_except_update(self):
        f = FunctionValue(mapping={IntValue(1): IntValue(10)})
        g = f.except_update(IntValue(1), IntValue(99))
        assert g.apply(IntValue(1)) == IntValue(99)
        # original unchanged
        assert f.apply(IntValue(1)) == IntValue(10)

    def test_except_multi(self):
        f = FunctionValue(mapping={IntValue(1): IntValue(10),
                                   IntValue(2): IntValue(20)})
        g = f.except_multi([(IntValue(1), IntValue(11)),
                            (IntValue(2), IntValue(22))])
        assert g.apply(IntValue(1)) == IntValue(11)
        assert g.apply(IntValue(2)) == IntValue(22)

    def test_equality(self):
        f1 = FunctionValue(mapping={IntValue(1): IntValue(10)})
        f2 = FunctionValue(mapping={IntValue(1): IntValue(10)})
        assert f1 == f2

    def test_len(self):
        f = FunctionValue(mapping={IntValue(1): IntValue(10),
                                   IntValue(2): IntValue(20)})
        assert len(f) == 2


# ===================================================================
# 5. TestSequenceValue
# ===================================================================

class TestSequenceValue:

    def test_empty_sequence(self):
        s = SequenceValue()
        assert s.length() == 0
        assert len(s) == 0

    def test_elements(self):
        s = SequenceValue([IntValue(1), IntValue(2)])
        assert s.elements == (IntValue(1), IntValue(2))

    def test_head(self):
        s = SequenceValue([IntValue(10), IntValue(20)])
        assert s.head() == IntValue(10)

    def test_head_empty_raises(self):
        with pytest.raises((TLAValueError, IndexError)):
            SequenceValue().head()

    def test_tail(self):
        s = SequenceValue([IntValue(1), IntValue(2), IntValue(3)])
        assert s.tail() == SequenceValue([IntValue(2), IntValue(3)])

    def test_append(self):
        s = SequenceValue([IntValue(1)])
        s2 = s.append(IntValue(2))
        assert s2.length() == 2
        assert s2.index(2) == IntValue(2)
        # original unchanged
        assert s.length() == 1

    def test_concat(self):
        a = SequenceValue([IntValue(1)])
        b = SequenceValue([IntValue(2), IntValue(3)])
        c = a.concat(b)
        assert c.length() == 3
        assert c.elements == (IntValue(1), IntValue(2), IntValue(3))

    def test_sub_seq(self):
        s = SequenceValue([IntValue(i) for i in range(1, 6)])
        sub = s.sub_seq(2, 4)
        assert sub == SequenceValue([IntValue(2), IntValue(3), IntValue(4)])

    @pytest.mark.parametrize("idx,expected", [
        (1, 10),
        (2, 20),
        (3, 30),
    ])
    def test_index_1_based(self, idx, expected):
        s = SequenceValue([IntValue(10), IntValue(20), IntValue(30)])
        assert s.index(idx) == IntValue(expected)

    def test_iteration(self):
        s = SequenceValue([IntValue(1), IntValue(2)])
        assert list(s) == [IntValue(1), IntValue(2)]

    def test_to_function(self):
        s = SequenceValue([IntValue(10), IntValue(20)])
        f = s.to_function()
        assert f.apply(IntValue(1)) == IntValue(10)
        assert f.apply(IntValue(2)) == IntValue(20)

    def test_select_seq(self):
        s = SequenceValue([IntValue(1), IntValue(2), IntValue(3), IntValue(4)])
        filtered = s.select_seq(lambda v: v.val % 2 == 0)
        assert filtered == SequenceValue([IntValue(2), IntValue(4)])


# ===================================================================
# 6. TestRecordValue
# ===================================================================

class TestRecordValue:

    def test_creation(self):
        r = RecordValue({"x": IntValue(1), "y": IntValue(2)})
        assert r.access("x") == IntValue(1)

    def test_field_names(self):
        r = RecordValue({"a": IntValue(1), "b": IntValue(2)})
        assert r.field_names() == frozenset({"a", "b"})

    def test_access_missing_field(self):
        r = RecordValue({"x": IntValue(1)})
        with pytest.raises((TLAValueError, KeyError)):
            r.access("z")

    def test_except_update(self):
        r = RecordValue({"x": IntValue(1), "y": IntValue(2)})
        r2 = r.except_update("x", IntValue(99))
        assert r2.access("x") == IntValue(99)
        assert r.access("x") == IntValue(1)  # original unchanged

    def test_except_multi(self):
        r = RecordValue({"a": IntValue(1), "b": IntValue(2)})
        r2 = r.except_multi({"a": IntValue(10), "b": IntValue(20)})
        assert r2.access("a") == IntValue(10)
        assert r2.access("b") == IntValue(20)

    def test_equality(self):
        r1 = RecordValue({"x": IntValue(1)})
        r2 = RecordValue({"x": IntValue(1)})
        assert r1 == r2

    def test_to_function(self):
        r = RecordValue({"x": IntValue(1)})
        f = r.to_function()
        assert f.apply(StringValue("x")) == IntValue(1)

    def test_len(self):
        r = RecordValue({"a": IntValue(1), "b": IntValue(2), "c": IntValue(3)})
        assert len(r) == 3


# ===================================================================
# 7. TestTupleValue
# ===================================================================

class TestTupleValue:

    def test_creation(self):
        t = TupleValue([IntValue(1), IntValue(2)])
        assert t.elements == (IntValue(1), IntValue(2))

    @pytest.mark.parametrize("idx,expected", [
        (1, 10),
        (2, 20),
    ])
    def test_index_1_based(self, idx, expected):
        t = TupleValue([IntValue(10), IntValue(20)])
        assert t.index(idx) == IntValue(expected)

    def test_equality(self):
        a = TupleValue([IntValue(1), IntValue(2)])
        b = TupleValue([IntValue(1), IntValue(2)])
        assert a == b

    def test_inequality_different_length(self):
        a = TupleValue([IntValue(1)])
        b = TupleValue([IntValue(1), IntValue(2)])
        assert a != b

    def test_hash(self):
        a = TupleValue([IntValue(1), IntValue(2)])
        b = TupleValue([IntValue(1), IntValue(2)])
        assert hash(a) == hash(b)

    def test_len(self):
        assert len(TupleValue([IntValue(1), IntValue(2), IntValue(3)])) == 3

    def test_empty(self):
        t = TupleValue()
        assert len(t) == 0


# ===================================================================
# 8. TestTLAState
# ===================================================================

class TestTLAState:

    def test_empty_state(self):
        s = TLAState()
        assert len(s) == 0

    def test_creation_with_bindings(self):
        s = TLAState({"x": IntValue(1), "y": IntValue(2)})
        assert s["x"] == IntValue(1)
        assert s.get("y") == IntValue(2)

    def test_variables(self):
        s = TLAState({"a": IntValue(1), "b": IntValue(2)})
        assert s.variables == frozenset({"a", "b"})

    def test_has_var(self):
        s = TLAState({"x": IntValue(1)})
        assert s.has_var("x")
        assert not s.has_var("z")

    def test_with_update(self):
        s = TLAState({"x": IntValue(1)})
        s2 = s.with_update("x", IntValue(99))
        assert s2["x"] == IntValue(99)
        assert s["x"] == IntValue(1)  # original unchanged

    def test_with_updates(self):
        s = TLAState({"x": IntValue(1), "y": IntValue(2)})
        s2 = s.with_updates({"x": IntValue(10), "y": IntValue(20)})
        assert s2["x"] == IntValue(10)
        assert s2["y"] == IntValue(20)

    def test_project(self):
        s = TLAState({"x": IntValue(1), "y": IntValue(2), "z": IntValue(3)})
        p = s.project(["x", "z"])
        assert p.variables == frozenset({"x", "z"})
        assert not p.has_var("y")

    def test_equality(self):
        s1 = TLAState({"x": IntValue(1)})
        s2 = TLAState({"x": IntValue(1)})
        assert s1 == s2

    def test_inequality(self):
        s1 = TLAState({"x": IntValue(1)})
        s2 = TLAState({"x": IntValue(2)})
        assert s1 != s2

    def test_hash_consistency(self):
        s1 = TLAState({"x": IntValue(1), "y": IntValue(2)})
        s2 = TLAState({"y": IntValue(2), "x": IntValue(1)})
        assert hash(s1) == hash(s2)

    def test_in_set(self):
        s = TLAState({"x": IntValue(1)})
        assert s in {s}

    def test_contains(self):
        s = TLAState({"x": IntValue(1)})
        assert "x" in s

    def test_items(self):
        s = TLAState({"a": IntValue(1)})
        items = list(s.items())
        assert ("a", IntValue(1)) in items

    def test_fingerprint(self):
        s = TLAState({"x": IntValue(1)})
        fp = s.fingerprint()
        assert isinstance(fp, StateSignature)

    def test_fingerprint_deterministic(self):
        s1 = TLAState({"x": IntValue(1), "y": IntValue(2)})
        s2 = TLAState({"y": IntValue(2), "x": IntValue(1)})
        assert s1.fingerprint() == s2.fingerprint()


# ===================================================================
# 9. TestEnvironment
# ===================================================================

class TestEnvironment:

    def test_bind_and_lookup(self):
        env = Environment()
        env.bind("x", IntValue(42))
        assert env.lookup("x") == IntValue(42)

    def test_lookup_missing_returns_none(self):
        env = Environment()
        assert env.lookup("missing") is None

    def test_resolve_missing_raises(self):
        env = Environment()
        with pytest.raises(Exception):
            env.resolve("missing")

    def test_push_pop_scope(self):
        env = Environment()
        env.bind("x", IntValue(1))
        env.push_scope("inner")
        env.bind("x", IntValue(2))
        assert env.lookup("x") == IntValue(2)
        env.pop_scope()
        assert env.lookup("x") == IntValue(1)

    def test_scope_depth(self):
        env = Environment()
        d0 = env.scope_depth
        env.push_scope()
        assert env.scope_depth == d0 + 1
        env.pop_scope()
        assert env.scope_depth == d0

    def test_scope_context_manager(self):
        env = Environment()
        env.bind("x", IntValue(1))
        with env.scope("test"):
            env.bind("x", IntValue(2))
            assert env.lookup("x") == IntValue(2)
        assert env.lookup("x") == IntValue(1)

    def test_has_binding(self):
        env = Environment()
        env.bind("x", IntValue(1))
        assert env.has_binding("x")
        assert not env.has_binding("y")

    def test_snapshot(self):
        env = Environment()
        env.bind("x", IntValue(1))
        snap = env.snapshot()
        env.bind("x", IntValue(999))
        assert snap.lookup("x") == IntValue(1)

    def test_declare_and_assign_constant(self):
        env = Environment()
        env.declare_constant("N")
        assert "N" in env.unresolved_constants()
        env.assign_constant("N", IntValue(5))
        assert env.constant_value("N") == IntValue(5)
        assert "N" not in env.unresolved_constants()

    def test_define_and_get_operator(self):
        env = Environment()
        op = OpDef(name="Inc", params=("x",))
        env.define_operator(op)
        assert env.has_operator("Inc")
        assert env.get_operator("Inc") == op

    def test_operator_names(self):
        env = Environment()
        env.define_operator(OpDef(name="A"))
        env.define_operator(OpDef(name="B"))
        assert "A" in env.operator_names()
        assert "B" in env.operator_names()

    def test_register_builtin(self):
        env = Environment()
        entry = BuiltinEntry(
            name="TestOp", module="TestMod", arity=1,
            evaluator=lambda x: x,
        )
        env.register_builtin(entry)
        assert env.get_builtin("TestOp") is not None

    def test_import_module(self):
        env = Environment()
        install_standard_modules(env)
        assert env.is_module_imported("Naturals")


# ===================================================================
# 10. TestEvaluator
# ===================================================================

class TestEvaluator:

    def test_int_literal(self):
        assert _eval(int_lit(42)) == IntValue(42)

    def test_bool_literal(self):
        assert _eval(bool_lit(True)) == BoolValue(True)

    def test_string_literal(self):
        assert _eval(string_lit("hello")) == StringValue("hello")

    def test_name_ref(self):
        env = _env_with_stdlib()
        env.bind("x", IntValue(7))
        assert _eval(name_ref("x"), env) == IntValue(7)

    def test_set_enum(self):
        result = _eval(set_enum(int_lit(1), int_lit(2), int_lit(3)))
        assert result == SetValue([IntValue(1), IntValue(2), IntValue(3)])

    def test_set_comp(self):
        # {x * 2 : x \in {1, 2, 3}}
        result = _eval(set_comp("x", set_enum(int_lit(1), int_lit(2), int_lit(3)),
                                binary_op("*", name_ref("x"), int_lit(2))))
        assert result == SetValue([IntValue(2), IntValue(4), IntValue(6)])

    def test_set_filter(self):
        # {x \in {1,2,3,4} : x > 2}
        result = _eval(set_filter("x",
                                  set_enum(int_lit(1), int_lit(2), int_lit(3), int_lit(4)),
                                  binary_op(">", name_ref("x"), int_lit(2))))
        assert result == SetValue([IntValue(3), IntValue(4)])

    def test_func_construct(self):
        # [x \in {1,2} |-> x + 10]
        f = _eval(func_construct("x",
                                 set_enum(int_lit(1), int_lit(2)),
                                 binary_op("+", name_ref("x"), int_lit(10))))
        assert isinstance(f, FunctionValue)
        assert f.apply(IntValue(1)) == IntValue(11)
        assert f.apply(IntValue(2)) == IntValue(12)

    def test_func_apply(self):
        env = _env_with_stdlib()
        env.bind("f", FunctionValue(mapping={IntValue(1): IntValue(100)}))
        result = _eval(func_apply(name_ref("f"), int_lit(1)), env)
        assert result == IntValue(100)

    def test_func_except(self):
        env = _env_with_stdlib()
        env.bind("f", FunctionValue(mapping={IntValue(1): IntValue(10),
                                             IntValue(2): IntValue(20)}))
        result = _eval(func_except(name_ref("f"), int_lit(1), int_lit(99)), env)
        assert isinstance(result, FunctionValue)
        assert result.apply(IntValue(1)) == IntValue(99)
        assert result.apply(IntValue(2)) == IntValue(20)

    def test_domain_op(self):
        env = _env_with_stdlib()
        env.bind("f", FunctionValue(mapping={IntValue(1): IntValue(10),
                                             IntValue(2): IntValue(20)}))
        result = _eval(domain_op(name_ref("f")), env)
        assert result == SetValue([IntValue(1), IntValue(2)])

    def test_tuple_construct(self):
        result = _eval(tuple_construct(int_lit(1), int_lit(2), int_lit(3)))
        assert isinstance(result, TupleValue)
        assert result.index(1) == IntValue(1)
        assert result.index(3) == IntValue(3)

    def test_record_construct(self):
        result = _eval(record_construct((("a", int_lit(1)), ("b", int_lit(2)))))
        assert isinstance(result, RecordValue)
        assert result.access("a") == IntValue(1)

    def test_record_access(self):
        env = _env_with_stdlib()
        env.bind("r", RecordValue({"x": IntValue(42)}))
        result = _eval(record_access(name_ref("r"), "x"), env)
        assert result == IntValue(42)

    def test_record_except(self):
        env = _env_with_stdlib()
        env.bind("r", RecordValue({"x": IntValue(1), "y": IntValue(2)}))
        result = _eval(record_except(name_ref("r"), "x", int_lit(99)), env)
        assert isinstance(result, RecordValue)
        assert result.access("x") == IntValue(99)
        assert result.access("y") == IntValue(2)

    def test_if_then_else_true(self):
        result = _eval(if_then_else(bool_lit(True), int_lit(1), int_lit(2)))
        assert result == IntValue(1)

    def test_if_then_else_false(self):
        result = _eval(if_then_else(bool_lit(False), int_lit(1), int_lit(2)))
        assert result == IntValue(2)

    def test_let_in(self):
        # LET x == 5 IN x + 1
        result = _eval(let_in((("x", int_lit(5)),),
                              binary_op("+", name_ref("x"), int_lit(1))))
        assert result == IntValue(6)

    def test_case_expr_match(self):
        # CASE TRUE -> 1 [] FALSE -> 2
        result = _eval(case_expr(
            ((bool_lit(True), int_lit(1)),
             (bool_lit(False), int_lit(2))),
        ))
        assert result == IntValue(1)

    def test_case_expr_other(self):
        result = _eval(case_expr(
            ((bool_lit(False), int_lit(1)),),
            other=int_lit(99),
        ))
        assert result == IntValue(99)

    def test_quant_forall_true(self):
        # \A x \in {1,2,3} : x > 0
        result = _eval(quant_forall("x",
                                    set_enum(int_lit(1), int_lit(2), int_lit(3)),
                                    binary_op(">", name_ref("x"), int_lit(0))))
        assert result == BoolValue(True)

    def test_quant_forall_false(self):
        # \A x \in {1,2,3} : x > 1
        result = _eval(quant_forall("x",
                                    set_enum(int_lit(1), int_lit(2), int_lit(3)),
                                    binary_op(">", name_ref("x"), int_lit(1))))
        assert result == BoolValue(False)

    def test_quant_exists_true(self):
        # \E x \in {1,2,3} : x = 2
        result = _eval(quant_exists("x",
                                    set_enum(int_lit(1), int_lit(2), int_lit(3)),
                                    binary_op("=", name_ref("x"), int_lit(2))))
        assert result == BoolValue(True)

    def test_quant_exists_false(self):
        # \E x \in {1,2,3} : x = 99
        result = _eval(quant_exists("x",
                                    set_enum(int_lit(1), int_lit(2), int_lit(3)),
                                    binary_op("=", name_ref("x"), int_lit(99))))
        assert result == BoolValue(False)

    def test_choose(self):
        # CHOOSE x \in {1,2,3} : x > 2
        result = _eval(choose("x",
                              set_enum(int_lit(1), int_lit(2), int_lit(3)),
                              binary_op(">", name_ref("x"), int_lit(2))))
        assert result == IntValue(3)

    def test_choose_no_match_raises(self):
        with pytest.raises(EvalError):
            _eval(choose("x",
                         set_enum(int_lit(1), int_lit(2)),
                         binary_op(">", name_ref("x"), int_lit(100))))

    @pytest.mark.parametrize("op,a,b,expected", [
        ("=", 1, 1, True),
        ("=", 1, 2, False),
        ("/=", 1, 2, True),
        ("<", 1, 2, True),
        ("<", 2, 1, False),
        ("<=", 2, 2, True),
        (">", 3, 2, True),
        (">=", 2, 2, True),
    ])
    def test_comparison_operators(self, op, a, b, expected):
        result = _eval(binary_op(op, int_lit(a), int_lit(b)))
        assert result == BoolValue(expected)

    def test_set_membership_via_eval(self):
        # 2 \in {1,2,3}
        result = _eval(binary_op("\\in", int_lit(2),
                                 set_enum(int_lit(1), int_lit(2), int_lit(3))))
        assert result == BoolValue(True)

    def test_set_not_membership_via_eval(self):
        result = _eval(binary_op("\\notin", int_lit(5),
                                 set_enum(int_lit(1), int_lit(2))))
        assert result == BoolValue(True)

    def test_set_union_via_eval(self):
        result = _eval(binary_op("\\union",
                                 set_enum(int_lit(1)),
                                 set_enum(int_lit(2))))
        assert result == SetValue([IntValue(1), IntValue(2)])

    def test_set_intersect_via_eval(self):
        result = _eval(binary_op("\\intersect",
                                 set_enum(int_lit(1), int_lit(2)),
                                 set_enum(int_lit(2), int_lit(3))))
        assert result == SetValue([IntValue(2)])

    def test_set_difference_via_eval(self):
        result = _eval(binary_op("\\",
                                 set_enum(int_lit(1), int_lit(2)),
                                 set_enum(int_lit(2))))
        assert result == SetValue([IntValue(1)])

    def test_subset_via_eval(self):
        result = _eval(binary_op("\\subseteq",
                                 set_enum(int_lit(1)),
                                 set_enum(int_lit(1), int_lit(2))))
        assert result == BoolValue(True)

    def test_unary_minus(self):
        result = _eval(unary_op("-", int_lit(5)))
        assert result == IntValue(-5)

    def test_nested_if_then_else(self):
        expr = if_then_else(
            binary_op(">", int_lit(3), int_lit(5)),
            int_lit(1),
            if_then_else(
                binary_op(">", int_lit(3), int_lit(2)),
                int_lit(2),
                int_lit(3),
            ),
        )
        assert _eval(expr) == IntValue(2)


# ===================================================================
# 11. TestActionEvaluation
# ===================================================================

class TestActionEvaluation:

    def _make_env_and_vars(self):
        env = _env_with_stdlib()
        state_vars = ("x",)
        return env, state_vars

    def test_compute_initial_states(self):
        env, svars = self._make_env_and_vars()
        # Init == x = 0
        init_expr = binary_op("=", name_ref("x"), int_lit(0))
        states = compute_initial_states(init_expr, env, svars)
        assert len(states) >= 1
        for s in states:
            assert s["x"] == IntValue(0)

    def test_compute_successors_simple(self):
        env, svars = self._make_env_and_vars()
        # x' = x + 1
        action = action_from_expr(
            binary_op("=", primed_ref("x"),
                      binary_op("+", name_ref("x"), int_lit(1)))
        )
        state = TLAState({"x": IntValue(0)})
        succs = compute_successors(action, state, env, svars)
        assert len(succs) >= 1
        for s in succs:
            assert s["x"] == IntValue(1)

    def test_action_disjunction(self):
        env = _env_with_stdlib()
        svars = ("x",)
        # x' = x + 1 \/ x' = x + 2
        a1 = action_from_expr(
            binary_op("=", primed_ref("x"),
                      binary_op("+", name_ref("x"), int_lit(1)))
        )
        a2 = action_from_expr(
            binary_op("=", primed_ref("x"),
                      binary_op("+", name_ref("x"), int_lit(2)))
        )
        action = action_disj(a1, a2)
        state = TLAState({"x": IntValue(0)})
        succs = compute_successors(action, state, env, svars)
        vals = {s["x"] for s in succs}
        assert IntValue(1) in vals
        assert IntValue(2) in vals

    def test_action_conjunction(self):
        env = _env_with_stdlib()
        svars = ("x", "y")
        a1 = action_from_expr(
            binary_op("=", primed_ref("x"),
                      binary_op("+", name_ref("x"), int_lit(1)))
        )
        a2 = action_from_expr(
            binary_op("=", primed_ref("y"), name_ref("y"))
        )
        action = action_conj(a1, a2)
        state = TLAState({"x": IntValue(0), "y": IntValue(10)})
        succs = compute_successors(action, state, env, svars)
        assert len(succs) >= 1
        for s in succs:
            assert s["x"] == IntValue(1)
            assert s["y"] == IntValue(10)

    def test_unchanged(self):
        env = _env_with_stdlib()
        svars = ("x",)
        action = action_unchanged("x")
        state = TLAState({"x": IntValue(5)})
        succs = compute_successors(action, state, env, svars)
        assert len(succs) >= 1
        for s in succs:
            assert s["x"] == IntValue(5)

    def test_detect_stuttering(self):
        env = _env_with_stdlib()
        svars = ("x",)
        action = action_unchanged("x")
        state = TLAState({"x": IntValue(5)})
        assert detect_stuttering(action, state, env, svars)

    def test_detect_no_stuttering(self):
        env = _env_with_stdlib()
        svars = ("x",)
        action = action_from_expr(
            binary_op("=", primed_ref("x"),
                      binary_op("+", name_ref("x"), int_lit(1)))
        )
        state = TLAState({"x": IntValue(0)})
        assert not detect_stuttering(action, state, env, svars)

    def test_quantified_action(self):
        env = _env_with_stdlib()
        svars = ("x",)
        # \E v \in {1,2,3} : x' = v
        body = action_from_expr(
            binary_op("=", primed_ref("x"), name_ref("v"))
        )
        action = action_exists("v", set_enum(int_lit(1), int_lit(2), int_lit(3)), body)
        state = TLAState({"x": IntValue(0)})
        succs = compute_successors(action, state, env, svars)
        vals = {s["x"] for s in succs}
        assert IntValue(1) in vals
        assert IntValue(2) in vals
        assert IntValue(3) in vals

    def test_action_evaluator_class(self):
        env = _env_with_stdlib()
        svars = ("x",)
        ae = ActionEvaluator(env, svars)
        assert ae.state_vars == svars

    def test_action_evaluator_evaluate_init(self):
        env = _env_with_stdlib()
        svars = ("x",)
        ae = ActionEvaluator(env, svars)
        init_expr = binary_op("=", name_ref("x"), int_lit(0))
        states = ae.evaluate_init(init_expr)
        assert len(states) >= 1

    def test_action_evaluator_evaluate_action(self):
        env = _env_with_stdlib()
        svars = ("x",)
        ae = ActionEvaluator(env, svars)
        action = action_from_expr(
            binary_op("=", primed_ref("x"),
                      binary_op("+", name_ref("x"), int_lit(1)))
        )
        state = TLAState({"x": IntValue(0)})
        succs = ae.evaluate_action(action, state)
        assert any(s["x"] == IntValue(1) for s in succs)

    def test_is_enabled(self):
        env = _env_with_stdlib()
        svars = ("x",)
        ae = ActionEvaluator(env, svars)
        # x' = x + 1 /\ x < 3
        action = action_conj(
            action_from_expr(binary_op("=", primed_ref("x"),
                                       binary_op("+", name_ref("x"), int_lit(1)))),
            action_from_expr(binary_op("<", name_ref("x"), int_lit(3))),
        )
        assert ae.is_enabled(action, TLAState({"x": IntValue(0)}))


# ===================================================================
# 12. TestBuiltins
# ===================================================================

class TestBuiltins:

    def test_install_standard_modules(self):
        env = Environment()
        install_standard_modules(env)
        assert env.is_module_imported("Naturals")
        assert env.is_module_imported("Integers")

    def test_default_registry(self):
        reg = get_default_registry()
        assert isinstance(reg, ModuleRegistry)
        assert "Naturals" in reg.module_names

    def test_registry_operators_in(self):
        reg = get_default_registry()
        nat_ops = reg.operators_in("Naturals")
        assert len(nat_ops) > 0

    def test_nat_addition(self):
        env = _env_with_stdlib()
        env.bind("a", IntValue(3))
        env.bind("b", IntValue(4))
        result = _eval(binary_op("+", name_ref("a"), name_ref("b")), env)
        assert result == IntValue(7)

    def test_nat_subtraction(self):
        result = _eval(binary_op("-", int_lit(10), int_lit(3)))
        assert result == IntValue(7)

    def test_nat_multiplication(self):
        result = _eval(binary_op("*", int_lit(6), int_lit(7)))
        assert result == IntValue(42)

    def test_nat_comparison(self):
        result = _eval(binary_op("<", int_lit(3), int_lit(5)))
        assert result == BoolValue(True)

    def test_finite_sets_cardinality(self):
        env = _env_with_stdlib()
        env.bind("S", SetValue([IntValue(1), IntValue(2), IntValue(3)]))
        # Cardinality is typically a builtin call
        assert env.get_builtin("Cardinality") is not None or \
               env.get_module_builtin("FiniteSets", "Cardinality") is not None

    def test_sequences_module_installed(self):
        env = _env_with_stdlib()
        assert env.is_module_imported("Sequences")

    def test_install_module_idempotent(self):
        env = Environment()
        install_standard_modules(env)
        install_standard_modules(env)
        assert env.is_module_imported("Naturals")


# ===================================================================
# 13. TestValueSerialization
# ===================================================================

class TestValueSerialization:

    @pytest.mark.parametrize("value", [
        IntValue(42),
        BoolValue(True),
        BoolValue(False),
        StringValue("hello"),
    ])
    def test_roundtrip_primitives(self, value):
        json_str = values_to_json_string(value)
        recovered = value_from_json_string(json_str)
        assert recovered == value

    def test_roundtrip_set(self):
        v = SetValue([IntValue(1), IntValue(2), IntValue(3)])
        json_str = values_to_json_string(v)
        recovered = value_from_json_string(json_str)
        assert recovered == v

    def test_roundtrip_tuple(self):
        v = TupleValue([IntValue(1), StringValue("a")])
        json_str = values_to_json_string(v)
        recovered = value_from_json_string(json_str)
        assert recovered == v

    def test_roundtrip_record(self):
        v = RecordValue({"x": IntValue(1), "y": BoolValue(True)})
        json_str = values_to_json_string(v)
        recovered = value_from_json_string(json_str)
        assert recovered == v

    def test_roundtrip_function(self):
        v = FunctionValue(mapping={IntValue(1): IntValue(10),
                                   IntValue(2): IntValue(20)})
        json_str = values_to_json_string(v)
        recovered = value_from_json_string(json_str)
        assert recovered == v

    def test_roundtrip_sequence(self):
        v = SequenceValue([IntValue(1), IntValue(2), IntValue(3)])
        json_str = values_to_json_string(v)
        recovered = value_from_json_string(json_str)
        assert recovered == v

    def test_roundtrip_nested(self):
        v = RecordValue({
            "counter": IntValue(5),
            "flags": SetValue([BoolValue(True), BoolValue(False)]),
            "log": SequenceValue([StringValue("a"), StringValue("b")]),
        })
        json_str = values_to_json_string(v)
        recovered = value_from_json_string(json_str)
        assert recovered == v

    def test_to_json_and_back(self):
        v = IntValue(99)
        j = v.to_json()
        recovered = TLAValue.from_json(j)
        assert recovered == v

    def test_value_from_python_int(self):
        assert value_from_python(42) == IntValue(42)

    def test_value_from_python_bool(self):
        assert value_from_python(True) == BoolValue(True)

    def test_value_from_python_str(self):
        assert value_from_python("hi") == StringValue("hi")


# ===================================================================
# 14. TestStateSpace
# ===================================================================

class TestStateSpace:

    def test_empty(self):
        ss = StateSpace()
        assert len(ss) == 0

    def test_add_state(self):
        ss = StateSpace()
        s = TLAState({"x": IntValue(1)})
        ss.add(s, is_initial=True)
        assert s in ss
        assert len(ss) == 1

    def test_initial_states(self):
        ss = StateSpace()
        s1 = TLAState({"x": IntValue(1)})
        s2 = TLAState({"x": IntValue(2)})
        ss.add(s1, is_initial=True)
        ss.add(s2, is_initial=False)
        inits = ss.initial_states()
        assert s1 in inits
        assert s2 not in inits

    def test_add_transition(self):
        ss = StateSpace()
        s1 = TLAState({"x": IntValue(1)})
        s2 = TLAState({"x": IntValue(2)})
        ss.add(s1, is_initial=True)
        ss.add(s2)
        ss.add_transition(s1, s2)
        assert ss.num_transitions >= 1

    def test_successors(self):
        ss = StateSpace()
        s1 = TLAState({"x": IntValue(0)})
        s2 = TLAState({"x": IntValue(1)})
        ss.add(s1, is_initial=True)
        ss.add(s2)
        ss.add_transition(s1, s2)
        succs = ss.successors(s1)
        assert s2 in succs

    def test_deadlock_states(self):
        ss = StateSpace()
        s1 = TLAState({"x": IntValue(0)})
        s2 = TLAState({"x": IntValue(1)})
        ss.add(s1, is_initial=True)
        ss.add(s2)
        ss.add_transition(s1, s2)
        # s2 has no outgoing transitions → deadlock
        deads = ss.deadlock_states()
        assert s2 in deads

    def test_stats(self):
        ss = StateSpace()
        s = TLAState({"x": IntValue(0)})
        ss.add(s, is_initial=True)
        stats = ss.stats()
        assert isinstance(stats, dict)

    def test_iteration(self):
        ss = StateSpace()
        s1 = TLAState({"x": IntValue(0)})
        s2 = TLAState({"x": IntValue(1)})
        ss.add(s1)
        ss.add(s2)
        all_states = list(ss)
        assert len(all_states) == 2

    def test_clear(self):
        ss = StateSpace()
        ss.add(TLAState({"x": IntValue(0)}))
        ss.clear()
        assert len(ss) == 0

    def test_add_returns_false_for_duplicate(self):
        ss = StateSpace()
        s = TLAState({"x": IntValue(0)})
        assert ss.add(s) is True
        assert ss.add(s) is False


# ===================================================================
# 15. TestTypeSystem
# ===================================================================

class TestTypeSystem:

    def test_infer_int(self):
        tc = TypeChecker()
        assert tc.infer_value_type(IntValue(1)).kind == TypeKind.INT

    def test_infer_bool(self):
        tc = TypeChecker()
        assert tc.infer_value_type(BoolValue(True)).kind == TypeKind.BOOL

    def test_infer_string(self):
        tc = TypeChecker()
        assert tc.infer_value_type(StringValue("x")).kind == TypeKind.STRING

    def test_infer_set(self):
        tc = TypeChecker()
        t = tc.infer_value_type(SetValue([IntValue(1)]))
        assert t.kind == TypeKind.SET

    def test_check_int(self):
        tc = TypeChecker()
        assert tc.check_int(IntValue(1))
        assert not tc.check_int(BoolValue(True))

    def test_check_bool(self):
        tc = TypeChecker()
        assert tc.check_bool(BoolValue(False))
        assert not tc.check_bool(IntValue(0))

    def test_check_value_type(self):
        tc = TypeChecker()
        assert tc.check_value_type(IntValue(1), INT_TYPE)
        assert not tc.check_value_type(IntValue(1), BOOL_TYPE)

    def test_set_type_constructor(self):
        t = set_type(INT_TYPE)
        assert t.kind == TypeKind.SET
        assert t.element_type == INT_TYPE

    def test_function_type_constructor(self):
        t = function_type(INT_TYPE, BOOL_TYPE)
        assert t.kind == TypeKind.FUNCTION
        assert t.domain_type == INT_TYPE
        assert t.range_type == BOOL_TYPE

    def test_tuple_type_constructor(self):
        t = tuple_type(INT_TYPE, BOOL_TYPE)
        assert t.kind == TypeKind.TUPLE
        assert t.tuple_types == (INT_TYPE, BOOL_TYPE)

    def test_pretty(self):
        t = set_type(INT_TYPE)
        p = t.pretty()
        assert isinstance(p, str)
        assert len(p) > 0

    def test_model_type(self):
        t = model_type("Color")
        assert t.kind == TypeKind.MODEL
        assert t.model_sort == "Color"


# ===================================================================
# 16. TestModelValue
# ===================================================================

class TestModelValue:

    def test_creation(self):
        v = ModelValue("red", "Color")
        assert v.name == "red"
        assert v.sort_name == "Color"

    def test_equality_same(self):
        a = ModelValue("red", "Color")
        b = ModelValue("red", "Color")
        assert a == b

    def test_inequality_different_name(self):
        a = ModelValue("red", "Color")
        b = ModelValue("blue", "Color")
        assert a != b

    def test_inequality_different_sort(self):
        a = ModelValue("x", "A")
        b = ModelValue("x", "B")
        assert a != b

    def test_hash(self):
        a = ModelValue("red", "Color")
        b = ModelValue("red", "Color")
        assert hash(a) == hash(b)

    def test_in_set(self):
        s = SetValue([ModelValue("r", "C"), ModelValue("g", "C")])
        assert s.contains(ModelValue("r", "C"))


# ===================================================================
# 17. TestStringValue
# ===================================================================

class TestStringValue:

    def test_creation(self):
        assert StringValue("abc").val == "abc"

    def test_equality(self):
        assert StringValue("x") == StringValue("x")
        assert StringValue("x") != StringValue("y")

    def test_hash(self):
        assert hash(StringValue("a")) == hash(StringValue("a"))

    def test_empty_string(self):
        assert StringValue("").val == ""

    def test_ordering(self):
        assert StringValue("a") < StringValue("b")


# ===================================================================
# 18. TestStateSerialization
# ===================================================================

class TestStateSerialization:

    def test_roundtrip(self):
        s = TLAState({"x": IntValue(1), "y": BoolValue(True)})
        j = s.to_json_string()
        recovered = TLAState.from_json_string(j)
        assert recovered == s

    def test_to_json(self):
        s = TLAState({"x": IntValue(42)})
        j = s.to_json()
        assert isinstance(j, dict)

    def test_pretty(self):
        s = TLAState({"x": IntValue(1)})
        p = s.pretty()
        assert isinstance(p, str)
        assert "x" in p
