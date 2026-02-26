"""Type inference and checking for TLA-lite expressions.

Builds a type environment from declarations, infers expression types,
checks operator argument types, and reports type errors with location
information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .ast_nodes import (
    AnyType,
    AlwaysExpr,
    ASTNode,
    ASTVisitor,
    Assumption,
    BoolLiteral,
    BoolType,
    CaseArm,
    CaseExpr,
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
    FunctionDef,
    FunctionType,
    Identifier,
    IfThenElse,
    IntLiteral,
    IntType,
    LeadsToExpr,
    LetIn,
    Module,
    Operator,
    OperatorApplication,
    OperatorDef,
    OperatorType,
    PrimedIdentifier,
    QuantifiedExpr,
    RecordAccess,
    RecordConstruction,
    RecordType,
    SequenceLiteral,
    SequenceType,
    SetComprehension,
    SetEnumeration,
    SetType,
    StringLiteral,
    StringType,
    StutteringAction,
    TemporalExistsExpr,
    TemporalForallExpr,
    TupleLiteral,
    TupleType,
    TypeAnnotation,
    UnchangedExpr,
    VariableDecl,
)
from .source_map import SourceLocation


# ============================================================================
# Type errors
# ============================================================================

@dataclass
class TypeError_:
    """A single type error with location."""
    message: str
    location: SourceLocation
    expected: Optional[TypeAnnotation] = None
    actual: Optional[TypeAnnotation] = None

    def __str__(self) -> str:
        s = f"{self.location}: type error: {self.message}"
        if self.expected and self.actual:
            s += f" (expected {_type_str(self.expected)}, got {_type_str(self.actual)})"
        return s


def _type_str(t: TypeAnnotation) -> str:
    """Pretty-print a type annotation."""
    if isinstance(t, IntType):
        return "Int"
    if isinstance(t, BoolType):
        return "Bool"
    if isinstance(t, StringType):
        return "String"
    if isinstance(t, SetType):
        return f"Set({_type_str(t.element_type)})"
    if isinstance(t, FunctionType):
        return f"[{_type_str(t.domain_type)} -> {_type_str(t.range_type)}]"
    if isinstance(t, TupleType):
        elems = " \\X ".join(_type_str(e) for e in t.element_types)
        return f"<<{elems}>>"
    if isinstance(t, RecordType):
        fields = ", ".join(f"{k}: {_type_str(v)}" for k, v in t.field_types.items())
        return f"[{fields}]"
    if isinstance(t, SequenceType):
        return f"Seq({_type_str(t.element_type)})"
    if isinstance(t, AnyType):
        return "Any"
    if isinstance(t, OperatorType):
        params = ", ".join(_type_str(p) for p in t.param_types)
        return f"({params}) -> {_type_str(t.return_type)}"
    return str(type(t).__name__)


# ============================================================================
# Type environment
# ============================================================================

class TypeEnv:
    """A stack of scopes mapping names to types."""

    def __init__(self) -> None:
        self._scopes: List[Dict[str, TypeAnnotation]] = [{}]

    def push_scope(self) -> None:
        self._scopes.append({})

    def pop_scope(self) -> None:
        if len(self._scopes) > 1:
            self._scopes.pop()

    def bind(self, name: str, typ: TypeAnnotation) -> None:
        self._scopes[-1][name] = typ

    def lookup(self, name: str) -> Optional[TypeAnnotation]:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def clone(self) -> TypeEnv:
        env = TypeEnv()
        env._scopes = [dict(s) for s in self._scopes]
        return env


# ============================================================================
# Unification helpers
# ============================================================================

def _unify(t1: TypeAnnotation, t2: TypeAnnotation) -> TypeAnnotation:
    """Return the unified type or AnyType if incompatible."""
    if isinstance(t1, AnyType):
        return t2
    if isinstance(t2, AnyType):
        return t1
    if type(t1) == type(t2):
        if isinstance(t1, SetType) and isinstance(t2, SetType):
            return SetType(element_type=_unify(t1.element_type, t2.element_type))
        if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
            return FunctionType(
                domain_type=_unify(t1.domain_type, t2.domain_type),
                range_type=_unify(t1.range_type, t2.range_type),
            )
        if isinstance(t1, SequenceType) and isinstance(t2, SequenceType):
            return SequenceType(element_type=_unify(t1.element_type, t2.element_type))
        if isinstance(t1, TupleType) and isinstance(t2, TupleType):
            if len(t1.element_types) == len(t2.element_types):
                return TupleType(
                    element_types=[
                        _unify(a, b)
                        for a, b in zip(t1.element_types, t2.element_types)
                    ]
                )
        if isinstance(t1, RecordType) and isinstance(t2, RecordType):
            merged: Dict[str, TypeAnnotation] = {}
            for k in set(t1.field_types) | set(t2.field_types):
                a = t1.field_types.get(k, AnyType())
                b = t2.field_types.get(k, AnyType())
                merged[k] = _unify(a, b)
            return RecordType(field_types=merged)
        return t1
    return AnyType()


def _is_compatible(actual: TypeAnnotation, expected: TypeAnnotation) -> bool:
    """Check if *actual* is compatible with *expected*."""
    if isinstance(expected, AnyType) or isinstance(actual, AnyType):
        return True
    if type(actual) == type(expected):
        if isinstance(actual, SetType) and isinstance(expected, SetType):
            return _is_compatible(actual.element_type, expected.element_type)
        if isinstance(actual, FunctionType) and isinstance(expected, FunctionType):
            return (
                _is_compatible(actual.domain_type, expected.domain_type)
                and _is_compatible(actual.range_type, expected.range_type)
            )
        if isinstance(actual, SequenceType) and isinstance(expected, SequenceType):
            return _is_compatible(actual.element_type, expected.element_type)
        if isinstance(actual, TupleType) and isinstance(expected, TupleType):
            if len(actual.element_types) != len(expected.element_types):
                return False
            return all(
                _is_compatible(a, e)
                for a, e in zip(actual.element_types, expected.element_types)
            )
        if isinstance(actual, RecordType) and isinstance(expected, RecordType):
            for k, v in expected.field_types.items():
                if k not in actual.field_types:
                    return False
                if not _is_compatible(actual.field_types[k], v):
                    return False
            return True
        return True
    return False


# ============================================================================
# Type checker visitor
# ============================================================================

class TypeChecker(ASTVisitor[TypeAnnotation]):
    """Infer and check types across a TLA-lite module."""

    def __init__(self) -> None:
        self.env = TypeEnv()
        self.errors: List[TypeError_] = []
        # Populate built-in names
        self.env.bind("Nat", SetType(element_type=IntType()))
        self.env.bind("Int", SetType(element_type=IntType()))
        self.env.bind("BOOLEAN", SetType(element_type=BoolType()))
        self.env.bind("STRING", SetType(element_type=StringType()))
        self.env.bind("TRUE", BoolType())
        self.env.bind("FALSE", BoolType())

    def check_module(self, module: Module) -> List[TypeError_]:
        """Type-check an entire module. Returns list of errors."""
        self.errors.clear()
        # Register constants (unknown types)
        for cdecl in module.constants:
            for name in cdecl.names:
                self.env.bind(name, AnyType())
        # Register variables (unknown types)
        for vdecl in module.variables:
            for name in vdecl.names:
                self.env.bind(name, AnyType())
        # Process definitions
        for defn in module.definitions:
            self._check_definition(defn)
        # Check assumptions
        for assumption in module.assumptions:
            if assumption.expr:
                t = self._infer(assumption.expr)
                self._expect_type(t, BoolType(), assumption.source_location, "ASSUME")
        # Check theorems
        for thm in module.theorems:
            if thm.expr:
                t = self._infer(thm.expr)
                self._expect_type(t, BoolType(), thm.source_location, "THEOREM")
        return self.errors

    def _check_definition(self, defn: Definition) -> None:
        if isinstance(defn, OperatorDef):
            self.env.push_scope()
            for p in defn.params:
                self.env.bind(p, AnyType())
            if defn.body:
                ret_type = self._infer(defn.body)
            else:
                ret_type = AnyType()
            self.env.pop_scope()
            op_type = OperatorType(
                param_types=[AnyType() for _ in defn.params],
                return_type=ret_type,
            )
            self.env.bind(defn.name, op_type)

        elif isinstance(defn, FunctionDef):
            self.env.push_scope()
            domain_type = self._infer(defn.set_expr) if defn.set_expr else AnyType()
            elem_type = self._extract_element_type(domain_type)
            self.env.bind(defn.variable, elem_type)
            range_type = self._infer(defn.body) if defn.body else AnyType()
            self.env.pop_scope()
            self.env.bind(defn.name, FunctionType(
                domain_type=elem_type, range_type=range_type
            ))

    # ── Inference ───────────────────────────────────────────────────

    def _infer(self, expr: Expression) -> TypeAnnotation:
        """Infer the type of an expression."""
        result = expr.accept(self)
        expr.inferred_type = result
        return result

    def visit_IntLiteral(self, node: IntLiteral) -> TypeAnnotation:
        return IntType()

    def visit_BoolLiteral(self, node: BoolLiteral) -> TypeAnnotation:
        return BoolType()

    def visit_StringLiteral(self, node: StringLiteral) -> TypeAnnotation:
        return StringType()

    def visit_Identifier(self, node: Identifier) -> TypeAnnotation:
        t = self.env.lookup(node.name)
        if t is None:
            self._report(f"Undeclared identifier '{node.name}'", node.source_location)
            return AnyType()
        return t

    def visit_PrimedIdentifier(self, node: PrimedIdentifier) -> TypeAnnotation:
        t = self.env.lookup(node.name)
        if t is None:
            self._report(f"Undeclared variable '{node.name}'", node.source_location)
            return AnyType()
        return t

    def visit_OperatorApplication(self, node: OperatorApplication) -> TypeAnnotation:
        op = node.operator
        arg_types = [self._infer(a) for a in node.operands]

        # User-defined operator call
        if node.operator_name and op == Operator.FUNC_APPLY:
            op_t = self.env.lookup(node.operator_name)
            if isinstance(op_t, OperatorType):
                if len(arg_types) != len(op_t.param_types):
                    self._report(
                        f"Operator '{node.operator_name}' expects "
                        f"{len(op_t.param_types)} args, got {len(arg_types)}",
                        node.source_location,
                    )
                return op_t.return_type
            return AnyType()

        # Arithmetic
        if op in (Operator.PLUS, Operator.MINUS, Operator.TIMES, Operator.DIV, Operator.MOD):
            for i, at in enumerate(arg_types):
                self._expect_type(at, IntType(), node.source_location, f"arithmetic operand {i+1}")
            return IntType()

        if op == Operator.UMINUS:
            if arg_types:
                self._expect_type(arg_types[0], IntType(), node.source_location, "unary minus")
            return IntType()

        if op == Operator.RANGE:
            for i, at in enumerate(arg_types):
                self._expect_type(at, IntType(), node.source_location, f"range operand {i+1}")
            return SetType(element_type=IntType())

        # Logical
        if op in (Operator.LAND, Operator.LOR, Operator.IMPLIES, Operator.EQUIV):
            for i, at in enumerate(arg_types):
                self._expect_type(at, BoolType(), node.source_location, f"logical operand {i+1}")
            return BoolType()

        if op == Operator.LNOT:
            if arg_types:
                self._expect_type(arg_types[0], BoolType(), node.source_location, "logical not")
            return BoolType()

        # Comparison
        if op in (Operator.EQ, Operator.NEQ, Operator.LT, Operator.GT, Operator.LEQ, Operator.GEQ):
            return BoolType()

        # Set membership
        if op in (Operator.IN, Operator.NOTIN):
            if len(arg_types) == 2:
                self._check_set_membership(arg_types[0], arg_types[1], node.source_location)
            return BoolType()

        if op == Operator.SUBSETEQ:
            return BoolType()

        # Set operations
        if op in (Operator.UNION, Operator.INTERSECT, Operator.SETDIFF):
            if len(arg_types) == 2:
                return _unify(arg_types[0], arg_types[1])
            return SetType()

        if op == Operator.CROSS:
            elem_types = [self._extract_element_type(t) for t in arg_types]
            return SetType(element_type=TupleType(element_types=elem_types))

        if op == Operator.POWERSET:
            if arg_types:
                return SetType(element_type=arg_types[0])
            return SetType()

        if op == Operator.UNION_ALL:
            if arg_types and isinstance(arg_types[0], SetType):
                return arg_types[0].element_type if isinstance(arg_types[0].element_type, SetType) else SetType()
            return SetType()

        # Function operators
        if op == Operator.COLON_GT:
            if len(arg_types) == 2:
                return FunctionType(domain_type=SetType(element_type=arg_types[0]), range_type=arg_types[1])
            return FunctionType()

        if op == Operator.AT_AT:
            if len(arg_types) == 2:
                return _unify(arg_types[0], arg_types[1])
            return FunctionType()

        # Sequence operators
        if op == Operator.HEAD:
            if arg_types and isinstance(arg_types[0], SequenceType):
                return arg_types[0].element_type
            return AnyType()

        if op == Operator.TAIL:
            if arg_types:
                return arg_types[0]
            return SequenceType()

        if op == Operator.LEN:
            return IntType()

        if op == Operator.APPEND:
            if arg_types:
                return arg_types[0]
            return SequenceType()

        if op == Operator.SUBSEQ:
            if arg_types:
                return arg_types[0]
            return SequenceType()

        if op == Operator.SEQ:
            if arg_types:
                return SetType(element_type=SequenceType(
                    element_type=self._extract_element_type(arg_types[0])
                ))
            return SetType()

        # Temporal / action
        if op in (Operator.PRIME, Operator.ALWAYS, Operator.EVENTUALLY,
                  Operator.ENABLED_OP, Operator.UNCHANGED_OP):
            if arg_types:
                return arg_types[0]
            return AnyType()

        if op == Operator.LEADS_TO:
            return BoolType()

        return AnyType()

    def visit_SetEnumeration(self, node: SetEnumeration) -> TypeAnnotation:
        if not node.elements:
            return SetType(element_type=AnyType())
        elem_types = [self._infer(e) for e in node.elements]
        unified = elem_types[0]
        for t in elem_types[1:]:
            unified = _unify(unified, t)
        return SetType(element_type=unified)

    def visit_SetComprehension(self, node: SetComprehension) -> TypeAnnotation:
        self.env.push_scope()
        if node.set_expr:
            set_type = self._infer(node.set_expr)
            elem_type = self._extract_element_type(set_type)
            self.env.bind(node.variable, elem_type)
        if node.map_expr:
            result_type = self._infer(node.map_expr)
            self.env.pop_scope()
            return SetType(element_type=result_type)
        if node.predicate:
            pred_type = self._infer(node.predicate)
            self._expect_type(pred_type, BoolType(), node.source_location, "set filter")
        self.env.pop_scope()
        return SetType(element_type=self.env.lookup(node.variable) or AnyType()) if node.set_expr else SetType()

    def visit_FunctionConstruction(self, node: FunctionConstruction) -> TypeAnnotation:
        self.env.push_scope()
        domain_type = AnyType()
        if node.set_expr:
            set_type = self._infer(node.set_expr)
            domain_type = self._extract_element_type(set_type)
        self.env.bind(node.variable, domain_type)
        range_type = self._infer(node.body) if node.body else AnyType()
        self.env.pop_scope()
        return FunctionType(domain_type=domain_type, range_type=range_type)

    def visit_FunctionApplication(self, node: FunctionApplication) -> TypeAnnotation:
        if node.function:
            func_type = self._infer(node.function)
            if node.argument:
                self._infer(node.argument)
            if isinstance(func_type, FunctionType):
                return func_type.range_type
            if isinstance(func_type, SequenceType):
                return func_type.element_type
        return AnyType()

    def visit_RecordConstruction(self, node: RecordConstruction) -> TypeAnnotation:
        ft: Dict[str, TypeAnnotation] = {}
        for name, val in node.fields:
            ft[name] = self._infer(val)
        return RecordType(field_types=ft)

    def visit_RecordAccess(self, node: RecordAccess) -> TypeAnnotation:
        if node.record:
            rec_type = self._infer(node.record)
            if isinstance(rec_type, RecordType):
                t = rec_type.field_types.get(node.field_name)
                if t is None:
                    self._report(
                        f"Record has no field '{node.field_name}'",
                        node.source_location,
                    )
                    return AnyType()
                return t
        return AnyType()

    def visit_TupleLiteral(self, node: TupleLiteral) -> TypeAnnotation:
        elem_types = [self._infer(e) for e in node.elements]
        return TupleType(element_types=elem_types)

    def visit_SequenceLiteral(self, node: SequenceLiteral) -> TypeAnnotation:
        if not node.elements:
            return SequenceType(element_type=AnyType())
        elem_types = [self._infer(e) for e in node.elements]
        unified = elem_types[0]
        for t in elem_types[1:]:
            unified = _unify(unified, t)
        return SequenceType(element_type=unified)

    def visit_QuantifiedExpr(self, node: QuantifiedExpr) -> TypeAnnotation:
        self.env.push_scope()
        for var, set_expr in node.variables:
            set_type = self._infer(set_expr)
            self.env.bind(var, self._extract_element_type(set_type))
        if node.body:
            body_type = self._infer(node.body)
            self._expect_type(body_type, BoolType(), node.source_location, "quantifier body")
        self.env.pop_scope()
        return BoolType()

    def visit_IfThenElse(self, node: IfThenElse) -> TypeAnnotation:
        if node.condition:
            ct = self._infer(node.condition)
            self._expect_type(ct, BoolType(), node.source_location, "IF condition")
        then_type = self._infer(node.then_expr) if node.then_expr else AnyType()
        else_type = self._infer(node.else_expr) if node.else_expr else AnyType()
        return _unify(then_type, else_type)

    def visit_LetIn(self, node: LetIn) -> TypeAnnotation:
        self.env.push_scope()
        for defn in node.definitions:
            self._check_definition(defn)
        result = self._infer(node.body) if node.body else AnyType()
        self.env.pop_scope()
        return result

    def visit_CaseExpr(self, node: CaseExpr) -> TypeAnnotation:
        result_type: TypeAnnotation = AnyType()
        for arm in node.arms:
            if arm.condition:
                ct = self._infer(arm.condition)
                self._expect_type(ct, BoolType(), arm.source_location, "CASE guard")
            if arm.value:
                vt = self._infer(arm.value)
                result_type = _unify(result_type, vt)
        if node.other:
            ot = self._infer(node.other)
            result_type = _unify(result_type, ot)
        return result_type

    def visit_ChooseExpr(self, node: ChooseExpr) -> TypeAnnotation:
        self.env.push_scope()
        elem_type: TypeAnnotation = AnyType()
        if node.set_expr:
            set_type = self._infer(node.set_expr)
            elem_type = self._extract_element_type(set_type)
        self.env.bind(node.variable, elem_type)
        if node.predicate:
            pt = self._infer(node.predicate)
            self._expect_type(pt, BoolType(), node.source_location, "CHOOSE predicate")
        self.env.pop_scope()
        return elem_type

    def visit_UnchangedExpr(self, node: UnchangedExpr) -> TypeAnnotation:
        for v in node.variables:
            self._infer(v)
        return BoolType()

    def visit_ExceptExpr(self, node: ExceptExpr) -> TypeAnnotation:
        base_type = self._infer(node.base) if node.base else AnyType()
        for path, val in node.substitutions:
            for p in path:
                self._infer(p)
            self._infer(val)
        return base_type

    def visit_DomainExpr(self, node: DomainExpr) -> TypeAnnotation:
        if node.expr:
            t = self._infer(node.expr)
            if isinstance(t, FunctionType):
                return SetType(element_type=t.domain_type)
        return SetType()

    # Temporal / action
    def visit_AlwaysExpr(self, node: AlwaysExpr) -> TypeAnnotation:
        if node.expr:
            self._infer(node.expr)
        return BoolType()

    def visit_EventuallyExpr(self, node: EventuallyExpr) -> TypeAnnotation:
        if node.expr:
            self._infer(node.expr)
        return BoolType()

    def visit_LeadsToExpr(self, node: LeadsToExpr) -> TypeAnnotation:
        if node.left:
            self._infer(node.left)
        if node.right:
            self._infer(node.right)
        return BoolType()

    def visit_TemporalForallExpr(self, node: TemporalForallExpr) -> TypeAnnotation:
        self.env.push_scope()
        self.env.bind(node.variable, AnyType())
        if node.body:
            self._infer(node.body)
        self.env.pop_scope()
        return BoolType()

    def visit_TemporalExistsExpr(self, node: TemporalExistsExpr) -> TypeAnnotation:
        self.env.push_scope()
        self.env.bind(node.variable, AnyType())
        if node.body:
            self._infer(node.body)
        self.env.pop_scope()
        return BoolType()

    def visit_StutteringAction(self, node: StutteringAction) -> TypeAnnotation:
        if node.action:
            self._infer(node.action)
        if node.variables:
            self._infer(node.variables)
        return BoolType()

    def visit_FairnessExpr(self, node: FairnessExpr) -> TypeAnnotation:
        if node.variables:
            self._infer(node.variables)
        if node.action:
            self._infer(node.action)
        return BoolType()

    # ── Helpers ─────────────────────────────────────────────────────

    def _extract_element_type(self, set_type: TypeAnnotation) -> TypeAnnotation:
        if isinstance(set_type, SetType):
            return set_type.element_type
        return AnyType()

    def _expect_type(
        self,
        actual: TypeAnnotation,
        expected: TypeAnnotation,
        location: SourceLocation,
        context: str,
    ) -> None:
        if not _is_compatible(actual, expected):
            self.errors.append(TypeError_(
                message=f"Type mismatch in {context}",
                location=location,
                expected=expected,
                actual=actual,
            ))

    def _check_set_membership(
        self,
        elem_type: TypeAnnotation,
        set_type: TypeAnnotation,
        location: SourceLocation,
    ) -> None:
        if isinstance(set_type, SetType):
            if not _is_compatible(elem_type, set_type.element_type):
                self.errors.append(TypeError_(
                    message="Element type incompatible with set",
                    location=location,
                    expected=set_type.element_type,
                    actual=elem_type,
                ))

    def _report(self, message: str, location: SourceLocation) -> None:
        self.errors.append(TypeError_(message=message, location=location))


# ============================================================================
# Convenience
# ============================================================================

def check_types(module: Module) -> List[TypeError_]:
    """Type-check a module and return the list of errors."""
    checker = TypeChecker()
    return checker.check_module(module)
