"""Utility functions for programmatic TLA-lite AST construction.

Provides builder helpers that streamline the creation of common AST
patterns: variable/constant declarations, set/function/record
constructions, logical connectives, quantifiers, primed variables,
unchanged expressions, fairness, and property wrappers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..parser.ast_nodes import (
    AlwaysExpr,
    Assumption,
    BoolLiteral,
    ChooseExpr,
    ConstantDecl,
    Definition,
    DomainExpr,
    EventuallyExpr,
    ExceptExpr,
    Expression,
    FairnessExpr,
    FunctionApplication,
    FunctionConstruction,
    Identifier,
    IfThenElse,
    IntLiteral,
    InvariantProperty,
    LeadsToExpr,
    LivenessProperty,
    Module,
    Operator,
    OperatorApplication,
    OperatorDef,
    PrimedIdentifier,
    Property,
    QuantifiedExpr,
    RecordAccess,
    RecordConstruction,
    SafetyProperty,
    SetComprehension,
    SetEnumeration,
    StutteringAction,
    StringLiteral,
    TemporalProperty,
    TupleLiteral,
    UnchangedExpr,
    VariableDecl,
)


# ---------------------------------------------------------------------------
# Identifier / literal helpers
# ---------------------------------------------------------------------------

def ident(name: str) -> Identifier:
    """Create an Identifier node."""
    return Identifier(name=name)


def primed(name: str) -> PrimedIdentifier:
    """Create a primed identifier (x')."""
    return PrimedIdentifier(name=name)


def int_lit(value: int) -> IntLiteral:
    """Create an integer literal."""
    return IntLiteral(value=value)


def bool_lit(value: bool) -> BoolLiteral:
    """Create a boolean literal."""
    return BoolLiteral(value=value)


def str_lit(value: str) -> StringLiteral:
    """Create a string literal."""
    return StringLiteral(value=value)


TRUE = BoolLiteral(value=True)
FALSE = BoolLiteral(value=False)


# ---------------------------------------------------------------------------
# Declaration helpers
# ---------------------------------------------------------------------------

def make_variable_decl(*names: str) -> VariableDecl:
    """VARIABLE v1, v2, …"""
    return VariableDecl(names=list(names))


def make_constant_decl(*names: str) -> ConstantDecl:
    """CONSTANT c1, c2, …"""
    return ConstantDecl(names=list(names))


def make_operator_def(name: str, body: Expression,
                      params: Optional[List[str]] = None,
                      is_local: bool = False) -> OperatorDef:
    """name(params) == body"""
    return OperatorDef(
        name=name,
        params=params or [],
        body=body,
        is_local=is_local,
    )


# ---------------------------------------------------------------------------
# Set construction helpers
# ---------------------------------------------------------------------------

def make_set_enum(*elements: Expression) -> SetEnumeration:
    """{e1, e2, …}"""
    return SetEnumeration(elements=list(elements))


def make_string_set(*values: str) -> SetEnumeration:
    """Convenience: {"a", "b", …}"""
    return SetEnumeration(elements=[StringLiteral(value=v) for v in values])


def make_int_set(*values: int) -> SetEnumeration:
    """Convenience: {1, 2, 3}"""
    return SetEnumeration(elements=[IntLiteral(value=v) for v in values])


def make_range(lo: Expression, hi: Expression) -> OperatorApplication:
    """lo .. hi"""
    return OperatorApplication(operator=Operator.RANGE, operands=[lo, hi])


def make_int_range(lo: int, hi: int) -> OperatorApplication:
    """lo..hi with integer literals."""
    return make_range(int_lit(lo), int_lit(hi))


def make_set_comprehension(var: str, set_expr: Expression,
                           predicate: Expression) -> SetComprehension:
    """{var \\in S : P(var)}"""
    return SetComprehension(variable=var, set_expr=set_expr,
                            predicate=predicate)


def make_set_map(var: str, set_expr: Expression,
                 map_expr: Expression) -> SetComprehension:
    """{e : var \\in S}"""
    return SetComprehension(variable=var, set_expr=set_expr,
                            map_expr=map_expr)


def make_powerset(s: Expression) -> OperatorApplication:
    """SUBSET S"""
    return OperatorApplication(operator=Operator.POWERSET, operands=[s])


def make_union_all(s: Expression) -> OperatorApplication:
    """UNION S (flattened union of a set of sets)."""
    return OperatorApplication(operator=Operator.UNION_ALL, operands=[s])


def make_cross(a: Expression, b: Expression) -> OperatorApplication:
    """A \\X B"""
    return OperatorApplication(operator=Operator.CROSS, operands=[a, b])


# ---------------------------------------------------------------------------
# Set operator helpers
# ---------------------------------------------------------------------------

def make_in(elem: Expression, s: Expression) -> OperatorApplication:
    """elem \\in S"""
    return OperatorApplication(operator=Operator.IN, operands=[elem, s])


def make_notin(elem: Expression, s: Expression) -> OperatorApplication:
    """elem \\notin S"""
    return OperatorApplication(operator=Operator.NOTIN, operands=[elem, s])


def make_union(a: Expression, b: Expression) -> OperatorApplication:
    """A \\union B"""
    return OperatorApplication(operator=Operator.UNION, operands=[a, b])


def make_intersect(a: Expression, b: Expression) -> OperatorApplication:
    """A \\intersect B"""
    return OperatorApplication(operator=Operator.INTERSECT, operands=[a, b])


def make_setdiff(a: Expression, b: Expression) -> OperatorApplication:
    """A \\ B"""
    return OperatorApplication(operator=Operator.SETDIFF, operands=[a, b])


def make_subseteq(a: Expression, b: Expression) -> OperatorApplication:
    """A \\subseteq B"""
    return OperatorApplication(operator=Operator.SUBSETEQ, operands=[a, b])


# ---------------------------------------------------------------------------
# Function / record / tuple construction
# ---------------------------------------------------------------------------

def make_function_construction(var: str, set_expr: Expression,
                               body: Expression) -> FunctionConstruction:
    """[var \\in S |-> body]"""
    return FunctionConstruction(variable=var, set_expr=set_expr, body=body)


def make_func_apply(func: Expression,
                    arg: Expression) -> FunctionApplication:
    """f[arg]"""
    return FunctionApplication(function=func, argument=arg)


def make_record(*fields: Tuple[str, Expression]) -> RecordConstruction:
    """[f1 |-> e1, f2 |-> e2, …]"""
    return RecordConstruction(fields=list(fields))


def make_record_access(rec: Expression, field_name: str) -> RecordAccess:
    """rec.field"""
    return RecordAccess(record=rec, field_name=field_name)


def make_tuple(*elems: Expression) -> TupleLiteral:
    """<<e1, e2, …>>"""
    return TupleLiteral(elements=list(elems))


def make_except(base: Expression,
                substitutions: List[Tuple[List[Expression], Expression]]
                ) -> ExceptExpr:
    """[base EXCEPT ![k1] = v1, ![k2] = v2, …]"""
    return ExceptExpr(base=base, substitutions=substitutions)


def make_domain(expr: Expression) -> DomainExpr:
    """DOMAIN expr"""
    return DomainExpr(expr=expr)


def make_choose(var: str, set_expr: Expression,
                predicate: Expression) -> ChooseExpr:
    """CHOOSE var \\in set_expr : predicate"""
    return ChooseExpr(variable=var, set_expr=set_expr, predicate=predicate)


# ---------------------------------------------------------------------------
# Logical connective helpers
# ---------------------------------------------------------------------------

def make_eq(a: Expression, b: Expression) -> OperatorApplication:
    """a = b"""
    return OperatorApplication(operator=Operator.EQ, operands=[a, b])


def make_neq(a: Expression, b: Expression) -> OperatorApplication:
    """a # b (or /=)"""
    return OperatorApplication(operator=Operator.NEQ, operands=[a, b])


def make_land(a: Expression, b: Expression) -> OperatorApplication:
    """a /\\ b"""
    return OperatorApplication(operator=Operator.LAND, operands=[a, b])


def make_lor(a: Expression, b: Expression) -> OperatorApplication:
    """a \\/ b"""
    return OperatorApplication(operator=Operator.LOR, operands=[a, b])


def make_lnot(a: Expression) -> OperatorApplication:
    """~a"""
    return OperatorApplication(operator=Operator.LNOT, operands=[a])


def make_implies(a: Expression, b: Expression) -> OperatorApplication:
    """a => b"""
    return OperatorApplication(operator=Operator.IMPLIES, operands=[a, b])


def make_equiv(a: Expression, b: Expression) -> OperatorApplication:
    """a <=> b"""
    return OperatorApplication(operator=Operator.EQUIV, operands=[a, b])


def make_conjunction(exprs: Sequence[Expression]) -> Expression:
    """Build a conjunction (/\\) of arbitrarily many expressions.

    Returns TRUE for an empty list, the single expression for length-1,
    and a left-folded LAND tree otherwise.
    """
    items = list(exprs)
    if not items:
        return TRUE
    result = items[0]
    for item in items[1:]:
        result = make_land(result, item)
    return result


def make_disjunction(exprs: Sequence[Expression]) -> Expression:
    """Build a disjunction (\\/) of arbitrarily many expressions.

    Returns FALSE for an empty list, the single expression for length-1,
    and a left-folded LOR tree otherwise.
    """
    items = list(exprs)
    if not items:
        return FALSE
    result = items[0]
    for item in items[1:]:
        result = make_lor(result, item)
    return result


# ---------------------------------------------------------------------------
# Comparison / arithmetic helpers
# ---------------------------------------------------------------------------

def make_lt(a: Expression, b: Expression) -> OperatorApplication:
    return OperatorApplication(operator=Operator.LT, operands=[a, b])


def make_gt(a: Expression, b: Expression) -> OperatorApplication:
    return OperatorApplication(operator=Operator.GT, operands=[a, b])


def make_leq(a: Expression, b: Expression) -> OperatorApplication:
    return OperatorApplication(operator=Operator.LEQ, operands=[a, b])


def make_geq(a: Expression, b: Expression) -> OperatorApplication:
    return OperatorApplication(operator=Operator.GEQ, operands=[a, b])


def make_plus(a: Expression, b: Expression) -> OperatorApplication:
    return OperatorApplication(operator=Operator.PLUS, operands=[a, b])


def make_minus(a: Expression, b: Expression) -> OperatorApplication:
    return OperatorApplication(operator=Operator.MINUS, operands=[a, b])


def make_times(a: Expression, b: Expression) -> OperatorApplication:
    return OperatorApplication(operator=Operator.TIMES, operands=[a, b])


def make_div(a: Expression, b: Expression) -> OperatorApplication:
    return OperatorApplication(operator=Operator.DIV, operands=[a, b])


def make_mod(a: Expression, b: Expression) -> OperatorApplication:
    return OperatorApplication(operator=Operator.MOD, operands=[a, b])


# ---------------------------------------------------------------------------
# Quantifier helpers
# ---------------------------------------------------------------------------

def make_forall(bindings: List[Tuple[str, Expression]],
                body: Expression) -> QuantifiedExpr:
    """\\A x \\in S, y \\in T : body"""
    return QuantifiedExpr(quantifier="forall", variables=bindings, body=body)


def make_exists(bindings: List[Tuple[str, Expression]],
                body: Expression) -> QuantifiedExpr:
    """\\E x \\in S, y \\in T : body"""
    return QuantifiedExpr(quantifier="exists", variables=bindings, body=body)


def make_forall_single(var: str, set_expr: Expression,
                       body: Expression) -> QuantifiedExpr:
    """\\A var \\in set_expr : body"""
    return make_forall([(var, set_expr)], body)


def make_exists_single(var: str, set_expr: Expression,
                       body: Expression) -> QuantifiedExpr:
    """\\E var \\in set_expr : body"""
    return make_exists([(var, set_expr)], body)


# ---------------------------------------------------------------------------
# Primed / unchanged / action helpers
# ---------------------------------------------------------------------------

def make_primed(name: str) -> PrimedIdentifier:
    """x' — alias for primed()."""
    return primed(name)


def make_unchanged(*names: str) -> UnchangedExpr:
    """UNCHANGED <<v1, v2, …>>"""
    return UnchangedExpr(variables=[Identifier(name=n) for n in names])


def make_unchanged_exprs(exprs: Sequence[Expression]) -> UnchangedExpr:
    """UNCHANGED <<e1, e2, …>> (arbitrary expressions)."""
    return UnchangedExpr(variables=list(exprs))


def make_stuttering_action(action: Expression, variables: Expression,
                           is_angle: bool = False) -> StutteringAction:
    """[A]_v or <A>_v"""
    return StutteringAction(action=action, variables=variables,
                            is_angle=is_angle)


# ---------------------------------------------------------------------------
# Fairness helpers
# ---------------------------------------------------------------------------

def make_wf(variables: Expression, action: Expression) -> FairnessExpr:
    """WF_vars(action)"""
    return FairnessExpr(kind="WF", variables=variables, action=action)


def make_sf(variables: Expression, action: Expression) -> FairnessExpr:
    """SF_vars(action)"""
    return FairnessExpr(kind="SF", variables=variables, action=action)


def make_fairness(kind: str, variables: Expression,
                  action: Expression) -> FairnessExpr:
    """WF_vars(action) or SF_vars(action) depending on kind."""
    assert kind in ("WF", "SF"), f"Fairness kind must be WF or SF, got {kind}"
    return FairnessExpr(kind=kind, variables=variables, action=action)


# ---------------------------------------------------------------------------
# Temporal operator helpers
# ---------------------------------------------------------------------------

def make_always(expr: Expression) -> AlwaysExpr:
    """[]P"""
    return AlwaysExpr(expr=expr)


def make_eventually(expr: Expression) -> EventuallyExpr:
    """<>P"""
    return EventuallyExpr(expr=expr)


def make_leads_to(left: Expression, right: Expression) -> LeadsToExpr:
    """P ~> Q"""
    return LeadsToExpr(left=left, right=right)


# ---------------------------------------------------------------------------
# Property helpers
# ---------------------------------------------------------------------------

def make_invariant_property(name: str,
                            expr: Expression) -> InvariantProperty:
    """Wrap an expression as a named invariant property."""
    return InvariantProperty(name=name, expr=expr)


def make_safety_property(name: str, expr: Expression) -> SafetyProperty:
    """Wrap an expression as a named safety property."""
    return SafetyProperty(name=name, expr=expr)


def make_liveness_property(name: str, expr: Expression) -> LivenessProperty:
    """Wrap an expression as a named liveness property."""
    return LivenessProperty(name=name, expr=expr)


def make_temporal_property(name: str, expr: Expression) -> TemporalProperty:
    """Wrap an expression as a named temporal property."""
    return TemporalProperty(name=name, expr=expr)


# ---------------------------------------------------------------------------
# If-then-else helper
# ---------------------------------------------------------------------------

def make_ite(cond: Expression, then_e: Expression,
             else_e: Expression) -> IfThenElse:
    """IF cond THEN then_e ELSE else_e"""
    return IfThenElse(condition=cond, then_expr=then_e, else_expr=else_e)


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------

class ModuleBuilder:
    """Incremental builder for TLA-lite Module AST nodes.

    Usage::

        m = ModuleBuilder("MySpec")
        m.add_extends("Naturals", "FiniteSets")
        m.add_constants("N")
        m.add_variables("x", "y")
        m.add_definition("Init", make_conjunction([...]))
        m.add_invariant("TypeOK", type_ok_expr)
        module = m.build()
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._extends: List[str] = []
        self._constants: List[ConstantDecl] = []
        self._variables: List[VariableDecl] = []
        self._definitions: List[Definition] = []
        self._assumptions: List[Assumption] = []
        self._properties: List[Property] = []

    # -- extends -------------------------------------------------------
    def add_extends(self, *modules: str) -> "ModuleBuilder":
        self._extends.extend(modules)
        return self

    # -- constants / variables -----------------------------------------
    def add_constants(self, *names: str) -> "ModuleBuilder":
        self._constants.append(make_constant_decl(*names))
        return self

    def add_variables(self, *names: str) -> "ModuleBuilder":
        self._variables.append(make_variable_decl(*names))
        return self

    # -- definitions ---------------------------------------------------
    def add_definition(self, name: str, body: Expression,
                       params: Optional[List[str]] = None,
                       is_local: bool = False) -> "ModuleBuilder":
        self._definitions.append(
            make_operator_def(name, body, params, is_local))
        return self

    def add_raw_definition(self, defn: Definition) -> "ModuleBuilder":
        self._definitions.append(defn)
        return self

    # -- assumptions ---------------------------------------------------
    def add_assumption(self, expr: Expression) -> "ModuleBuilder":
        self._assumptions.append(Assumption(expr=expr))
        return self

    # -- properties ----------------------------------------------------
    def add_invariant(self, name: str, expr: Expression) -> "ModuleBuilder":
        self._properties.append(make_invariant_property(name, expr))
        return self

    def add_safety(self, name: str, expr: Expression) -> "ModuleBuilder":
        self._properties.append(make_safety_property(name, expr))
        return self

    def add_liveness(self, name: str, expr: Expression) -> "ModuleBuilder":
        self._properties.append(make_liveness_property(name, expr))
        return self

    def add_temporal(self, name: str, expr: Expression) -> "ModuleBuilder":
        self._properties.append(make_temporal_property(name, expr))
        return self

    def add_property(self, prop: Property) -> "ModuleBuilder":
        self._properties.append(prop)
        return self

    # -- build ---------------------------------------------------------
    def build(self) -> Module:
        """Return the constructed Module AST node."""
        return Module(
            name=self._name,
            extends=list(self._extends),
            constants=list(self._constants),
            variables=list(self._variables),
            definitions=list(self._definitions),
            assumptions=list(self._assumptions),
            properties=list(self._properties),
        )


# ---------------------------------------------------------------------------
# High-level pattern helpers
# ---------------------------------------------------------------------------

def make_state_function_eq(var_name: str, domain: Expression,
                           value: Expression) -> Expression:
    """var = [x \\in domain |-> value]  — initialise a state function."""
    return make_eq(
        ident(var_name),
        make_function_construction("x", domain, value),
    )


def make_primed_eq(var_name: str, value: Expression) -> Expression:
    """var' = value"""
    return make_eq(primed(var_name), value)


def make_primed_func_update(var_name: str, key: Expression,
                            value: Expression) -> Expression:
    """var' = [var EXCEPT ![key] = value]"""
    return make_eq(
        primed(var_name),
        make_except(ident(var_name), [([key], value)]),
    )


def make_guard(condition: Expression, updates: List[Expression],
               unchanged_vars: List[str]) -> Expression:
    """Combine a guard condition with primed-variable updates and UNCHANGED.

    Returns: condition /\\ update1 /\\ update2 /\\ … /\\ UNCHANGED <<…>>
    """
    parts: List[Expression] = [condition]
    parts.extend(updates)
    if unchanged_vars:
        parts.append(make_unchanged(*unchanged_vars))
    return make_conjunction(parts)


def make_vars_tuple(*var_names: str) -> TupleLiteral:
    """<<v1, v2, …>> — tuple of variable identifiers."""
    return make_tuple(*(ident(n) for n in var_names))


def make_spec_with_fairness(init_name: str, next_name: str,
                            vars_tuple: Expression,
                            fairness_exprs: Sequence[Expression]
                            ) -> Expression:
    """Init /\\ [][Next]_vars /\\ WF1 /\\ WF2 /\\ …"""
    init_ref = ident(init_name)
    next_ref = ident(next_name)
    box_next = make_always(
        make_stuttering_action(next_ref, vars_tuple, is_angle=False)
    )
    parts: List[Expression] = [init_ref, box_next]
    parts.extend(fairness_exprs)
    return make_conjunction(parts)


def make_cardinality_call(set_expr: Expression) -> OperatorApplication:
    """Cardinality(S) — user-defined operator application."""
    return OperatorApplication(
        operator=Operator.FUNC_APPLY,
        operands=[set_expr],
        operator_name="Cardinality",
    )


def make_user_op(name: str, *args: Expression) -> OperatorApplication:
    """Apply a user-defined operator: name(arg1, arg2, …)."""
    return OperatorApplication(
        operator=Operator.FUNC_APPLY,
        operands=list(args),
        operator_name=name,
    )
