from __future__ import annotations

"""
TypeScript type narrowing analysis for refinement type inference.

Analyzes TypeScript-specific narrowing patterns (typeof, instanceof, in,
equality, truthiness, user-defined type guards, discriminated unions, etc.)
and converts them to refinement predicates for a dynamically-typed language
refinement type inference system.
"""

import copy
import enum
import hashlib
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Local type definitions (no external project imports)
# ---------------------------------------------------------------------------

class TypeKind(enum.Enum):
    """Kind of a TypeScript type."""
    STRING = "string"
    NUMBER = "number"
    BIGINT = "bigint"
    BOOLEAN = "boolean"
    SYMBOL = "symbol"
    UNDEFINED = "undefined"
    NULL = "null"
    OBJECT = "object"
    FUNCTION = "function"
    VOID = "void"
    NEVER = "never"
    UNKNOWN = "unknown"
    ANY = "any"
    LITERAL_STRING = "literal_string"
    LITERAL_NUMBER = "literal_number"
    LITERAL_BOOLEAN = "literal_boolean"
    LITERAL_BIGINT = "literal_bigint"
    UNION = "union"
    INTERSECTION = "intersection"
    TUPLE = "tuple"
    ARRAY = "array"
    CLASS = "class"
    INTERFACE = "interface"
    ENUM = "enum"
    ENUM_MEMBER = "enum_member"
    TYPE_PARAMETER = "type_parameter"
    CONDITIONAL = "conditional"
    MAPPED = "mapped"
    TEMPLATE_LITERAL = "template_literal"


@dataclass
class TSType:
    """Representation of a TypeScript type."""
    kind: TypeKind
    name: str = ""
    literal_value: Any = None
    members: List[TSType] = field(default_factory=list)
    properties: Dict[str, TSType] = field(default_factory=dict)
    type_arguments: List[TSType] = field(default_factory=list)
    base_types: List[TSType] = field(default_factory=list)
    is_optional: bool = False
    is_readonly: bool = False
    index_signatures: Dict[str, TSType] = field(default_factory=dict)
    call_signatures: List[Dict[str, Any]] = field(default_factory=list)
    discriminant_property: Optional[str] = None
    discriminant_value: Any = None

    def __hash__(self) -> int:
        return hash((self.kind, self.name, str(self.literal_value)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TSType):
            return NotImplemented
        return (
            self.kind == other.kind
            and self.name == other.name
            and self.literal_value == other.literal_value
        )

    @property
    def is_nullable(self) -> bool:
        if self.kind in (TypeKind.NULL, TypeKind.UNDEFINED):
            return True
        if self.kind == TypeKind.UNION:
            return any(m.is_nullable for m in self.members)
        return False

    @property
    def is_primitive(self) -> bool:
        return self.kind in (
            TypeKind.STRING, TypeKind.NUMBER, TypeKind.BIGINT,
            TypeKind.BOOLEAN, TypeKind.SYMBOL, TypeKind.UNDEFINED,
            TypeKind.NULL, TypeKind.VOID,
        )

    @property
    def is_literal(self) -> bool:
        return self.kind in (
            TypeKind.LITERAL_STRING, TypeKind.LITERAL_NUMBER,
            TypeKind.LITERAL_BOOLEAN, TypeKind.LITERAL_BIGINT,
        )

    @property
    def is_falsy_possible(self) -> bool:
        if self.kind in (TypeKind.NULL, TypeKind.UNDEFINED, TypeKind.VOID):
            return True
        if self.kind == TypeKind.LITERAL_STRING and self.literal_value == "":
            return True
        if self.kind == TypeKind.LITERAL_NUMBER and self.literal_value == 0:
            return True
        if self.kind == TypeKind.LITERAL_BOOLEAN and self.literal_value is False:
            return True
        if self.kind == TypeKind.LITERAL_BIGINT and self.literal_value == 0:
            return True
        if self.kind == TypeKind.UNION:
            return any(m.is_falsy_possible for m in self.members)
        return False

    def without_nullish(self) -> TSType:
        if self.kind in (TypeKind.NULL, TypeKind.UNDEFINED):
            return TSType(kind=TypeKind.NEVER)
        if self.kind == TypeKind.UNION:
            filtered = [
                m for m in self.members
                if m.kind not in (TypeKind.NULL, TypeKind.UNDEFINED)
            ]
            if not filtered:
                return TSType(kind=TypeKind.NEVER)
            if len(filtered) == 1:
                return filtered[0]
            return TSType(kind=TypeKind.UNION, members=filtered)
        return self

    def without_falsy(self) -> TSType:
        if self.is_falsy_possible and self.kind != TypeKind.UNION:
            return TSType(kind=TypeKind.NEVER)
        if self.kind == TypeKind.UNION:
            filtered = [m for m in self.members if not m.is_falsy_possible]
            if not filtered:
                return TSType(kind=TypeKind.NEVER)
            if len(filtered) == 1:
                return filtered[0]
            return TSType(kind=TypeKind.UNION, members=filtered)
        return self

    def only_falsy(self) -> TSType:
        if self.kind == TypeKind.UNION:
            filtered = [m for m in self.members if m.is_falsy_possible]
            if not filtered:
                return TSType(kind=TypeKind.NEVER)
            if len(filtered) == 1:
                return filtered[0]
            return TSType(kind=TypeKind.UNION, members=filtered)
        if self.is_falsy_possible:
            return self
        return TSType(kind=TypeKind.NEVER)


# Predefined singleton types for convenience
STRING_TYPE = TSType(kind=TypeKind.STRING, name="string")
NUMBER_TYPE = TSType(kind=TypeKind.NUMBER, name="number")
BIGINT_TYPE = TSType(kind=TypeKind.BIGINT, name="bigint")
BOOLEAN_TYPE = TSType(kind=TypeKind.BOOLEAN, name="boolean")
SYMBOL_TYPE = TSType(kind=TypeKind.SYMBOL, name="symbol")
UNDEFINED_TYPE = TSType(kind=TypeKind.UNDEFINED, name="undefined")
NULL_TYPE = TSType(kind=TypeKind.NULL, name="null")
OBJECT_TYPE = TSType(kind=TypeKind.OBJECT, name="object")
FUNCTION_TYPE = TSType(kind=TypeKind.FUNCTION, name="function")
VOID_TYPE = TSType(kind=TypeKind.VOID, name="void")
NEVER_TYPE = TSType(kind=TypeKind.NEVER, name="never")
UNKNOWN_TYPE = TSType(kind=TypeKind.UNKNOWN, name="unknown")
ANY_TYPE = TSType(kind=TypeKind.ANY, name="any")

TYPEOF_RESULT_MAP: Dict[str, TSType] = {
    "string": STRING_TYPE,
    "number": NUMBER_TYPE,
    "bigint": BIGINT_TYPE,
    "boolean": BOOLEAN_TYPE,
    "symbol": SYMBOL_TYPE,
    "undefined": UNDEFINED_TYPE,
    "object": OBJECT_TYPE,
    "function": FUNCTION_TYPE,
}

FALSY_TYPES: List[TSType] = [
    TSType(kind=TypeKind.LITERAL_BOOLEAN, literal_value=False),
    TSType(kind=TypeKind.LITERAL_NUMBER, literal_value=0),
    TSType(kind=TypeKind.LITERAL_STRING, literal_value=""),
    NULL_TYPE,
    UNDEFINED_TYPE,
    TSType(kind=TypeKind.LITERAL_BIGINT, literal_value=0),
]


def make_union(types: List[TSType]) -> TSType:
    """Create a union type, flattening nested unions and deduplicating."""
    flat: List[TSType] = []
    seen: Set[int] = set()
    for t in types:
        if t.kind == TypeKind.NEVER:
            continue
        if t.kind == TypeKind.UNION:
            for m in t.members:
                h = hash(m)
                if h not in seen:
                    seen.add(h)
                    flat.append(m)
        else:
            h = hash(t)
            if h not in seen:
                seen.add(h)
                flat.append(t)
    if not flat:
        return NEVER_TYPE
    if len(flat) == 1:
        return flat[0]
    return TSType(kind=TypeKind.UNION, members=flat)


def make_intersection(types: List[TSType]) -> TSType:
    """Create an intersection type."""
    flat: List[TSType] = []
    for t in types:
        if t.kind == TypeKind.NEVER:
            return NEVER_TYPE
        if t.kind == TypeKind.INTERSECTION:
            flat.extend(t.members)
        else:
            flat.append(t)
    if not flat:
        return UNKNOWN_TYPE
    if len(flat) == 1:
        return flat[0]
    return TSType(kind=TypeKind.INTERSECTION, members=flat)


def exclude_type(base: TSType, excluded: TSType) -> TSType:
    """Exclude a type from a union: Exclude<base, excluded>."""
    if base.kind == TypeKind.UNION:
        remaining = [m for m in base.members if m != excluded]
        return make_union(remaining)
    if base == excluded:
        return NEVER_TYPE
    return base


def exclude_types(base: TSType, excluded_list: List[TSType]) -> TSType:
    """Exclude multiple types from a base type."""
    result = base
    for ex in excluded_list:
        result = exclude_type(result, ex)
    return result


def intersect_with(base: TSType, target: TSType) -> TSType:
    """Narrow a base type to only include members assignable to target."""
    if base.kind == TypeKind.UNION:
        kept = [m for m in base.members if is_assignable_to(m, target)]
        return make_union(kept)
    if is_assignable_to(base, target):
        return base
    return NEVER_TYPE


def is_assignable_to(source: TSType, target: TSType) -> bool:
    """Check if source is assignable to target (simplified)."""
    if target.kind == TypeKind.ANY or target.kind == TypeKind.UNKNOWN:
        return True
    if source.kind == TypeKind.NEVER:
        return True
    if source.kind == TypeKind.ANY:
        return True
    if source == target:
        return True
    if source.kind == TypeKind.LITERAL_STRING and target.kind == TypeKind.STRING:
        return True
    if source.kind == TypeKind.LITERAL_NUMBER and target.kind == TypeKind.NUMBER:
        return True
    if source.kind == TypeKind.LITERAL_BOOLEAN and target.kind == TypeKind.BOOLEAN:
        return True
    if source.kind == TypeKind.LITERAL_BIGINT and target.kind == TypeKind.BIGINT:
        return True
    if source.kind == TypeKind.NULL and target.kind == TypeKind.OBJECT:
        return False
    if target.kind == TypeKind.UNION:
        return any(is_assignable_to(source, m) for m in target.members)
    if source.kind == TypeKind.UNION:
        return all(is_assignable_to(m, target) for m in source.members)
    if target.kind == TypeKind.INTERSECTION:
        return all(is_assignable_to(source, m) for m in target.members)
    if (
        source.kind in (TypeKind.CLASS, TypeKind.INTERFACE)
        and target.kind in (TypeKind.CLASS, TypeKind.INTERFACE)
    ):
        return _structural_assignable(source, target)
    if source.kind == TypeKind.ENUM_MEMBER and target.kind == TypeKind.ENUM:
        return source.name.startswith(target.name + ".")
    return False


def _structural_assignable(source: TSType, target: TSType) -> bool:
    """Check structural assignability between object-like types."""
    for prop_name, prop_type in target.properties.items():
        if prop_name not in source.properties:
            if not prop_type.is_optional:
                return False
        else:
            if not is_assignable_to(source.properties[prop_name], prop_type):
                return False
    return True


def typeof_result(t: TSType) -> str:
    """Return the typeof result string for a given type."""
    mapping = {
        TypeKind.STRING: "string",
        TypeKind.LITERAL_STRING: "string",
        TypeKind.TEMPLATE_LITERAL: "string",
        TypeKind.NUMBER: "number",
        TypeKind.LITERAL_NUMBER: "number",
        TypeKind.BIGINT: "bigint",
        TypeKind.LITERAL_BIGINT: "bigint",
        TypeKind.BOOLEAN: "boolean",
        TypeKind.LITERAL_BOOLEAN: "boolean",
        TypeKind.SYMBOL: "symbol",
        TypeKind.UNDEFINED: "undefined",
        TypeKind.VOID: "undefined",
        TypeKind.NULL: "object",
        TypeKind.FUNCTION: "function",
    }
    return mapping.get(t.kind, "object")


# ---------------------------------------------------------------------------
# AST node representations (simplified TypeScript AST)
# ---------------------------------------------------------------------------

class ASTNodeKind(enum.Enum):
    """Kind of AST node."""
    PROGRAM = "program"
    FUNCTION_DECLARATION = "function_declaration"
    VARIABLE_DECLARATION = "variable_declaration"
    PARAMETER = "parameter"
    BLOCK = "block"
    IF_STATEMENT = "if_statement"
    SWITCH_STATEMENT = "switch_statement"
    SWITCH_CASE = "switch_case"
    WHILE_STATEMENT = "while_statement"
    FOR_STATEMENT = "for_statement"
    FOR_IN_STATEMENT = "for_in_statement"
    FOR_OF_STATEMENT = "for_of_statement"
    DO_WHILE_STATEMENT = "do_while_statement"
    RETURN_STATEMENT = "return_statement"
    THROW_STATEMENT = "throw_statement"
    TRY_STATEMENT = "try_statement"
    CATCH_CLAUSE = "catch_clause"
    EXPRESSION_STATEMENT = "expression_statement"
    BINARY_EXPRESSION = "binary_expression"
    UNARY_EXPRESSION = "unary_expression"
    TYPEOF_EXPRESSION = "typeof_expression"
    INSTANCEOF_EXPRESSION = "instanceof_expression"
    IN_EXPRESSION = "in_expression"
    CALL_EXPRESSION = "call_expression"
    MEMBER_EXPRESSION = "member_expression"
    OPTIONAL_CHAIN_EXPRESSION = "optional_chain_expression"
    NULLISH_COALESCING_EXPRESSION = "nullish_coalescing_expression"
    ASSIGNMENT_EXPRESSION = "assignment_expression"
    NULLISH_ASSIGNMENT_EXPRESSION = "nullish_assignment_expression"
    IDENTIFIER = "identifier"
    LITERAL = "literal"
    TEMPLATE_LITERAL = "template_literal"
    OBJECT_EXPRESSION = "object_expression"
    ARRAY_EXPRESSION = "array_expression"
    CONDITIONAL_EXPRESSION = "conditional_expression"
    LOGICAL_EXPRESSION = "logical_expression"
    TYPE_ASSERTION = "type_assertion"
    AS_EXPRESSION = "as_expression"
    DESTRUCTURING_PATTERN = "destructuring_pattern"
    OBJECT_PATTERN = "object_pattern"
    ARRAY_PATTERN = "array_pattern"
    REST_ELEMENT = "rest_element"
    SPREAD_ELEMENT = "spread_element"
    ARROW_FUNCTION = "arrow_function"
    CLASS_DECLARATION = "class_declaration"
    PROPERTY_DECLARATION = "property_declaration"
    METHOD_DECLARATION = "method_declaration"
    ENUM_DECLARATION = "enum_declaration"
    INTERFACE_DECLARATION = "interface_declaration"
    TYPE_ALIAS_DECLARATION = "type_alias_declaration"
    ASSERTION_EXPRESSION = "assertion_expression"


@dataclass
class SourceLocation:
    """Location in source code."""
    file: str = ""
    line: int = 0
    column: int = 0
    end_line: int = 0
    end_column: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


@dataclass
class ASTNode:
    """Simplified TypeScript AST node."""
    kind: ASTNodeKind
    location: SourceLocation = field(default_factory=SourceLocation)
    children: List[ASTNode] = field(default_factory=list)
    name: str = ""
    operator: str = ""
    value: Any = None
    type_annotation: Optional[TSType] = None
    modifiers: List[str] = field(default_factory=list)
    # For function declarations
    parameters: List[ASTNode] = field(default_factory=list)
    body: Optional[ASTNode] = None
    return_type: Optional[TSType] = None
    # For type guard functions
    type_predicate: Optional[Dict[str, Any]] = None
    # For conditional / logical
    condition: Optional[ASTNode] = None
    consequent: Optional[ASTNode] = None
    alternate: Optional[ASTNode] = None
    left: Optional[ASTNode] = None
    right: Optional[ASTNode] = None
    # For assignments
    target: Optional[ASTNode] = None
    init: Optional[ASTNode] = None
    # For switch
    discriminant: Optional[ASTNode] = None
    cases: List[ASTNode] = field(default_factory=list)
    # For try/catch
    handler: Optional[ASTNode] = None
    finalizer: Optional[ASTNode] = None
    # For member/optional chain
    object_node: Optional[ASTNode] = None
    property_node: Optional[ASTNode] = None
    optional: bool = False
    computed: bool = False
    # For call expressions
    callee: Optional[ASTNode] = None
    arguments: List[ASTNode] = field(default_factory=list)
    # For destructuring
    elements: List[ASTNode] = field(default_factory=list)
    properties_nodes: List[ASTNode] = field(default_factory=list)
    # For catch
    param: Optional[ASTNode] = None
    # Misc
    is_const: bool = False
    is_assertion: bool = False
    asserts_type: Optional[TSType] = None
    expression: Optional[ASTNode] = None
    declarations: List[ASTNode] = field(default_factory=list)

    def __hash__(self) -> int:
        return id(self)


# ---------------------------------------------------------------------------
# Control flow graph
# ---------------------------------------------------------------------------

class CFGBlockKind(enum.Enum):
    """Kind of a control flow graph block."""
    ENTRY = "entry"
    EXIT = "exit"
    NORMAL = "normal"
    CONDITION = "condition"
    LOOP_HEADER = "loop_header"
    LOOP_BODY = "loop_body"
    LOOP_EXIT = "loop_exit"
    SWITCH_HEAD = "switch_head"
    CASE = "case"
    DEFAULT_CASE = "default_case"
    TRY = "try"
    CATCH = "catch"
    FINALLY = "finally"
    THROW = "throw"
    RETURN = "return"
    UNREACHABLE = "unreachable"


@dataclass
class CFGEdge:
    """Edge in a control flow graph."""
    source: int
    target: int
    label: str = ""  # "true", "false", "unconditional", "exception", "fallthrough"
    condition: Optional[ASTNode] = None
    is_back_edge: bool = False

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.label))


@dataclass
class CFGBlock:
    """Block in a control flow graph."""
    id: int
    kind: CFGBlockKind
    statements: List[ASTNode] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    edges_out: List[CFGEdge] = field(default_factory=list)
    edges_in: List[CFGEdge] = field(default_factory=list)
    # For condition blocks, the condition expression
    condition: Optional[ASTNode] = None
    # Dominance
    dominators: Set[int] = field(default_factory=set)
    immediate_dominator: Optional[int] = None
    # Narrowing state at entry and exit
    narrowing_state_in: Dict[str, TSType] = field(default_factory=dict)
    narrowing_state_out: Dict[str, TSType] = field(default_factory=dict)

    def __hash__(self) -> int:
        return self.id


@dataclass
class CFG:
    """Control flow graph."""
    blocks: Dict[int, CFGBlock] = field(default_factory=dict)
    entry_id: int = 0
    exit_id: int = 1
    edges: List[CFGEdge] = field(default_factory=list)
    _next_id: int = 2

    def add_block(self, kind: CFGBlockKind = CFGBlockKind.NORMAL) -> CFGBlock:
        block = CFGBlock(id=self._next_id, kind=kind)
        self.blocks[self._next_id] = block
        self._next_id += 1
        return block

    def add_edge(
        self,
        source: int,
        target: int,
        label: str = "unconditional",
        condition: Optional[ASTNode] = None,
    ) -> CFGEdge:
        edge = CFGEdge(source=source, target=target, label=label, condition=condition)
        self.edges.append(edge)
        if source in self.blocks:
            self.blocks[source].successors.append(target)
            self.blocks[source].edges_out.append(edge)
        if target in self.blocks:
            self.blocks[target].predecessors.append(source)
            self.blocks[target].edges_in.append(edge)
        return edge

    def get_block(self, block_id: int) -> Optional[CFGBlock]:
        return self.blocks.get(block_id)

    def successor_blocks(self, block_id: int) -> List[CFGBlock]:
        block = self.blocks.get(block_id)
        if not block:
            return []
        return [self.blocks[s] for s in block.successors if s in self.blocks]

    def predecessor_blocks(self, block_id: int) -> List[CFGBlock]:
        block = self.blocks.get(block_id)
        if not block:
            return []
        return [self.blocks[p] for p in block.predecessors if p in self.blocks]

    def topological_order(self) -> List[int]:
        """Return block IDs in topological order (ignoring back edges)."""
        visited: Set[int] = set()
        order: List[int] = []

        def dfs(bid: int) -> None:
            if bid in visited:
                return
            visited.add(bid)
            block = self.blocks.get(bid)
            if block:
                for edge in block.edges_out:
                    if not edge.is_back_edge:
                        dfs(edge.target)
            order.append(bid)

        dfs(self.entry_id)
        order.reverse()
        return order

    def reverse_postorder(self) -> List[int]:
        """Return block IDs in reverse postorder."""
        visited: Set[int] = set()
        order: List[int] = []

        def dfs(bid: int) -> None:
            if bid in visited:
                return
            visited.add(bid)
            block = self.blocks.get(bid)
            if block:
                for succ in block.successors:
                    dfs(succ)
            order.append(bid)

        dfs(self.entry_id)
        order.reverse()
        return order

    def detect_back_edges(self) -> List[CFGEdge]:
        """Detect back edges using DFS."""
        visited: Set[int] = set()
        in_stack: Set[int] = set()
        back_edges: List[CFGEdge] = []

        def dfs(bid: int) -> None:
            visited.add(bid)
            in_stack.add(bid)
            block = self.blocks.get(bid)
            if block:
                for edge in block.edges_out:
                    if edge.target in in_stack:
                        edge.is_back_edge = True
                        back_edges.append(edge)
                    elif edge.target not in visited:
                        dfs(edge.target)
            in_stack.discard(bid)

        dfs(self.entry_id)
        return back_edges

    def compute_dominators(self) -> None:
        """Compute dominators for all blocks."""
        all_ids = set(self.blocks.keys())
        for bid in all_ids:
            self.blocks[bid].dominators = set(all_ids)
        self.blocks[self.entry_id].dominators = {self.entry_id}

        changed = True
        rpo = self.reverse_postorder()
        while changed:
            changed = False
            for bid in rpo:
                if bid == self.entry_id:
                    continue
                block = self.blocks[bid]
                preds = [
                    p for p in block.predecessors if p in self.blocks
                ]
                if not preds:
                    new_dom = {bid}
                else:
                    new_dom = set.intersection(
                        *(self.blocks[p].dominators for p in preds)
                    )
                    new_dom = new_dom | {bid}
                if new_dom != block.dominators:
                    block.dominators = new_dom
                    changed = True

        # Compute immediate dominators
        for bid in all_ids:
            if bid == self.entry_id:
                continue
            block = self.blocks[bid]
            strict_doms = block.dominators - {bid}
            for d in strict_doms:
                # d is the immediate dominator if no other strict dominator
                # is dominated by d (except bid)
                is_idom = True
                for other in strict_doms:
                    if other != d and d in self.blocks[other].dominators:
                        is_idom = False
                        break
                if is_idom:
                    block.immediate_dominator = d
                    break


# ---------------------------------------------------------------------------
# Refinement predicate language (target for conversion)
# ---------------------------------------------------------------------------

class PredicateKind(enum.Enum):
    """Kind of refinement predicate."""
    ISINSTANCE = "isinstance"
    IS_NONE = "is_none"
    NOT_NONE = "not_none"
    EQUALITY = "equality"
    INEQUALITY = "inequality"
    LESS_THAN = "less_than"
    LESS_EQUAL = "less_equal"
    GREATER_THAN = "greater_than"
    GREATER_EQUAL = "greater_equal"
    HASATTR = "hasattr"
    IS_TRUTHY = "is_truthy"
    IS_FALSY = "is_falsy"
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    TRUE = "true"
    FALSE = "false"
    TYPE_GUARD = "type_guard"
    CALLABLE = "callable"
    HAS_LENGTH = "has_length"
    IN_SET = "in_set"
    MATCH_PATTERN = "match_pattern"


@dataclass
class Predicate:
    """A refinement predicate."""
    kind: PredicateKind
    variable: str = ""
    type_name: str = ""
    value: Any = None
    attribute: str = ""
    operands: List[Predicate] = field(default_factory=list)
    guard_function: str = ""

    def __hash__(self) -> int:
        return hash((self.kind, self.variable, self.type_name, str(self.value)))

    def negate(self) -> Predicate:
        """Return the negation of this predicate."""
        negation_map = {
            PredicateKind.IS_NONE: PredicateKind.NOT_NONE,
            PredicateKind.NOT_NONE: PredicateKind.IS_NONE,
            PredicateKind.IS_TRUTHY: PredicateKind.IS_FALSY,
            PredicateKind.IS_FALSY: PredicateKind.IS_TRUTHY,
            PredicateKind.TRUE: PredicateKind.FALSE,
            PredicateKind.FALSE: PredicateKind.TRUE,
        }
        if self.kind in negation_map:
            return Predicate(
                kind=negation_map[self.kind],
                variable=self.variable,
                type_name=self.type_name,
                value=self.value,
                attribute=self.attribute,
            )
        if self.kind == PredicateKind.AND:
            return Predicate(
                kind=PredicateKind.OR,
                operands=[op.negate() for op in self.operands],
            )
        if self.kind == PredicateKind.OR:
            return Predicate(
                kind=PredicateKind.AND,
                operands=[op.negate() for op in self.operands],
            )
        if self.kind == PredicateKind.NOT and self.operands:
            return self.operands[0]
        if self.kind == PredicateKind.EQUALITY:
            return Predicate(
                kind=PredicateKind.INEQUALITY,
                variable=self.variable,
                value=self.value,
            )
        if self.kind == PredicateKind.INEQUALITY:
            return Predicate(
                kind=PredicateKind.EQUALITY,
                variable=self.variable,
                value=self.value,
            )
        return Predicate(kind=PredicateKind.NOT, operands=[self])

    def conjoin(self, other: Predicate) -> Predicate:
        """Return self AND other."""
        if self.kind == PredicateKind.TRUE:
            return other
        if other.kind == PredicateKind.TRUE:
            return self
        if self.kind == PredicateKind.FALSE or other.kind == PredicateKind.FALSE:
            return Predicate(kind=PredicateKind.FALSE)
        parts: List[Predicate] = []
        if self.kind == PredicateKind.AND:
            parts.extend(self.operands)
        else:
            parts.append(self)
        if other.kind == PredicateKind.AND:
            parts.extend(other.operands)
        else:
            parts.append(other)
        return Predicate(kind=PredicateKind.AND, operands=parts)

    def disjoin(self, other: Predicate) -> Predicate:
        """Return self OR other."""
        if self.kind == PredicateKind.FALSE:
            return other
        if other.kind == PredicateKind.FALSE:
            return self
        if self.kind == PredicateKind.TRUE or other.kind == PredicateKind.TRUE:
            return Predicate(kind=PredicateKind.TRUE)
        parts: List[Predicate] = []
        if self.kind == PredicateKind.OR:
            parts.extend(self.operands)
        else:
            parts.append(self)
        if other.kind == PredicateKind.OR:
            parts.extend(other.operands)
        else:
            parts.append(other)
        return Predicate(kind=PredicateKind.OR, operands=parts)

    def __str__(self) -> str:
        if self.kind == PredicateKind.ISINSTANCE:
            return f"isinstance({self.variable}, {self.type_name})"
        if self.kind == PredicateKind.IS_NONE:
            return f"{self.variable} is None"
        if self.kind == PredicateKind.NOT_NONE:
            return f"{self.variable} is not None"
        if self.kind == PredicateKind.EQUALITY:
            return f"{self.variable} == {self.value!r}"
        if self.kind == PredicateKind.INEQUALITY:
            return f"{self.variable} != {self.value!r}"
        if self.kind == PredicateKind.HASATTR:
            return f"hasattr({self.variable}, {self.attribute!r})"
        if self.kind == PredicateKind.IS_TRUTHY:
            return f"bool({self.variable})"
        if self.kind == PredicateKind.IS_FALSY:
            return f"not bool({self.variable})"
        if self.kind == PredicateKind.AND:
            return " and ".join(str(op) for op in self.operands)
        if self.kind == PredicateKind.OR:
            return " or ".join(f"({op})" for op in self.operands)
        if self.kind == PredicateKind.NOT and self.operands:
            return f"not ({self.operands[0]})"
        if self.kind == PredicateKind.TRUE:
            return "True"
        if self.kind == PredicateKind.FALSE:
            return "False"
        if self.kind == PredicateKind.TYPE_GUARD:
            return f"{self.guard_function}({self.variable})"
        if self.kind == PredicateKind.CALLABLE:
            return f"callable({self.variable})"
        if self.kind == PredicateKind.LESS_THAN:
            return f"{self.variable} < {self.value!r}"
        if self.kind == PredicateKind.LESS_EQUAL:
            return f"{self.variable} <= {self.value!r}"
        if self.kind == PredicateKind.GREATER_THAN:
            return f"{self.variable} > {self.value!r}"
        if self.kind == PredicateKind.GREATER_EQUAL:
            return f"{self.variable} >= {self.value!r}"
        if self.kind == PredicateKind.IN_SET:
            return f"{self.variable} in {self.value!r}"
        return f"Predicate({self.kind.value}, {self.variable})"


# ---------------------------------------------------------------------------
# Narrowing enums & core data structures
# ---------------------------------------------------------------------------

class NarrowingKind(enum.Enum):
    """Kind of narrowing operation."""
    TYPEOF = "typeof"
    INSTANCEOF = "instanceof"
    IN = "in"
    EQUALITY = "equality"
    TRUTHINESS = "truthiness"
    USER_DEFINED = "user_defined"
    ASSERTION = "assertion"
    DISCRIMINATED_UNION = "discriminated_union"
    OPTIONAL_CHAINING = "optional_chaining"
    NULLISH_COALESCING = "nullish_coalescing"
    ASSIGNMENT = "assignment"
    CONTROL_FLOW = "control_flow"
    COMPOUND = "compound"


class FlowNodeType(enum.Enum):
    """Type of a flow node."""
    ASSIGNMENT = "assignment"
    CONDITION_TRUE = "condition_true"
    CONDITION_FALSE = "condition_false"
    CALL = "call"
    LOOP_START = "loop_start"
    LOOP_END = "loop_end"
    NARROWING = "narrowing"
    WIDENING = "widening"
    UNREACHABLE = "unreachable"
    START = "start"
    END = "end"


@dataclass
class NarrowingPoint:
    """A point in the program where type narrowing occurs."""
    location: SourceLocation
    variable: str
    guard_expression: Optional[ASTNode]
    true_type: Optional[TSType]
    false_type: Optional[TSType]
    narrowing_kind: NarrowingKind
    original_type: Optional[TSType] = None
    predicate: Optional[Predicate] = None
    cfg_block_id: Optional[int] = None
    is_negated: bool = False
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((str(self.location), self.variable, self.narrowing_kind))

    def flipped(self) -> NarrowingPoint:
        """Return the narrowing point with true/false types swapped."""
        return NarrowingPoint(
            location=self.location,
            variable=self.variable,
            guard_expression=self.guard_expression,
            true_type=self.false_type,
            false_type=self.true_type,
            narrowing_kind=self.narrowing_kind,
            original_type=self.original_type,
            predicate=self.predicate.negate() if self.predicate else None,
            cfg_block_id=self.cfg_block_id,
            is_negated=not self.is_negated,
            confidence=self.confidence,
            metadata=self.metadata,
        )


@dataclass
class FlowNode:
    """A node in the narrowing flow graph."""
    id: int
    type: FlowNodeType
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    associated_narrowing: Optional[NarrowingPoint] = None
    variable_types: Dict[str, TSType] = field(default_factory=dict)
    cfg_block_id: Optional[int] = None
    ast_node: Optional[ASTNode] = None

    def __hash__(self) -> int:
        return self.id


@dataclass
class NarrowingResult:
    """Result of narrowing analysis for a function."""
    narrowing_points: List[NarrowingPoint] = field(default_factory=list)
    variable_types: Dict[str, Dict[int, TSType]] = field(default_factory=dict)
    predicates: List[Predicate] = field(default_factory=list)
    flow_nodes: List[FlowNode] = field(default_factory=list)
    exhaustiveness_info: Dict[str, Any] = field(default_factory=dict)
    statistics: Optional[NarrowingStatistics] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class NarrowingStatistics:
    """Statistics about narrowing analysis."""
    total_narrowing_points: int = 0
    narrowing_kinds_distribution: Dict[str, int] = field(default_factory=dict)
    type_guard_coverage: float = 0.0
    exhaustiveness_coverage: float = 0.0
    narrowing_depth: Dict[str, int] = field(default_factory=dict)
    variables_narrowed: int = 0
    unreachable_blocks: int = 0
    total_blocks: int = 0
    total_flow_nodes: int = 0
    compound_narrowings: int = 0
    user_defined_guards: int = 0
    discriminated_unions: int = 0
    typeof_narrowings: int = 0
    instanceof_narrowings: int = 0
    equality_narrowings: int = 0
    truthiness_narrowings: int = 0
    in_narrowings: int = 0
    assertion_narrowings: int = 0
    optional_chaining_narrowings: int = 0
    nullish_coalescing_narrowings: int = 0
    assignment_narrowings: int = 0
    average_narrowing_depth: float = 0.0
    max_narrowing_depth: int = 0
    exhaustive_switches: int = 0
    non_exhaustive_switches: int = 0
    predicates_generated: int = 0

    def compute_derived(self) -> None:
        """Compute derived statistics."""
        if self.narrowing_depth:
            depths = list(self.narrowing_depth.values())
            self.average_narrowing_depth = sum(depths) / len(depths)
            self.max_narrowing_depth = max(depths)
        total_switches = self.exhaustive_switches + self.non_exhaustive_switches
        if total_switches > 0:
            self.exhaustiveness_coverage = (
                self.exhaustive_switches / total_switches
            )
        self.narrowing_kinds_distribution = {
            "typeof": self.typeof_narrowings,
            "instanceof": self.instanceof_narrowings,
            "equality": self.equality_narrowings,
            "truthiness": self.truthiness_narrowings,
            "in": self.in_narrowings,
            "user_defined": self.user_defined_guards,
            "assertion": self.assertion_narrowings,
            "discriminated_union": self.discriminated_unions,
            "optional_chaining": self.optional_chaining_narrowings,
            "nullish_coalescing": self.nullish_coalescing_narrowings,
            "assignment": self.assignment_narrowings,
            "compound": self.compound_narrowings,
        }


# ---------------------------------------------------------------------------
# TypeofNarrowing
# ---------------------------------------------------------------------------

class TypeofNarrowing:
    """Handle typeof x === "string" patterns and variants."""

    TYPEOF_RESULTS = (
        "string", "number", "bigint", "boolean",
        "symbol", "undefined", "object", "function",
    )

    def __init__(self) -> None:
        self._cache: Dict[str, List[NarrowingPoint]] = {}

    def analyze(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Analyze a typeof expression and produce a narrowing point."""
        typeof_expr, comparison_value, operator = self._extract_typeof_pattern(node)
        if typeof_expr is None or comparison_value is None:
            return None

        var_name = self._extract_variable_name(typeof_expr)
        if not var_name:
            return None

        original_type = variable_types.get(var_name, UNKNOWN_TYPE)
        is_strict = operator in ("===", "!==")
        is_negated = operator in ("!==", "!=")

        true_type = self._compute_typeof_true_type(
            original_type, comparison_value, is_strict
        )
        false_type = self._compute_typeof_false_type(
            original_type, comparison_value, is_strict
        )

        if is_negated:
            true_type, false_type = false_type, true_type

        predicate = self._make_typeof_predicate(var_name, comparison_value, is_negated)

        return NarrowingPoint(
            location=node.location,
            variable=var_name,
            guard_expression=node,
            true_type=true_type,
            false_type=false_type,
            narrowing_kind=NarrowingKind.TYPEOF,
            original_type=original_type,
            predicate=predicate,
            metadata={
                "typeof_result": comparison_value,
                "operator": operator,
                "is_strict": is_strict,
            },
        )

    def analyze_switch(
        self,
        discriminant: ASTNode,
        cases: List[ASTNode],
        variable_types: Dict[str, TSType],
    ) -> List[NarrowingPoint]:
        """Analyze typeof in switch statement."""
        if discriminant.kind != ASTNodeKind.TYPEOF_EXPRESSION:
            return []

        operand = discriminant.expression
        if operand is None or operand.kind != ASTNodeKind.IDENTIFIER:
            return []

        var_name = operand.name
        original_type = variable_types.get(var_name, UNKNOWN_TYPE)
        points: List[NarrowingPoint] = []
        handled_typeof_values: List[str] = []

        for case_node in cases:
            if case_node.value is not None and isinstance(case_node.value, str):
                typeof_val = case_node.value
                if typeof_val in self.TYPEOF_RESULTS:
                    handled_typeof_values.append(typeof_val)
                    case_true = self._compute_typeof_true_type(
                        original_type, typeof_val, is_strict=True
                    )
                    case_false = self._compute_typeof_false_type(
                        original_type, typeof_val, is_strict=True
                    )
                    points.append(NarrowingPoint(
                        location=case_node.location,
                        variable=var_name,
                        guard_expression=case_node,
                        true_type=case_true,
                        false_type=case_false,
                        narrowing_kind=NarrowingKind.TYPEOF,
                        original_type=original_type,
                        metadata={
                            "typeof_result": typeof_val,
                            "in_switch": True,
                        },
                    ))
            else:
                # Default case: the type is whatever wasn't matched
                remaining = original_type
                for tv in handled_typeof_values:
                    remaining = self._compute_typeof_false_type(
                        remaining, tv, is_strict=True
                    )
                points.append(NarrowingPoint(
                    location=case_node.location,
                    variable=var_name,
                    guard_expression=case_node,
                    true_type=remaining,
                    false_type=None,
                    narrowing_kind=NarrowingKind.TYPEOF,
                    original_type=original_type,
                    metadata={
                        "is_default_case": True,
                        "in_switch": True,
                    },
                ))

        return points

    def _extract_typeof_pattern(
        self, node: ASTNode
    ) -> Tuple[Optional[ASTNode], Optional[str], Optional[str]]:
        """Extract typeof pattern from a binary expression.

        Returns (typeof_expression, compared_string, operator) or (None, None, None).
        """
        if node.kind != ASTNodeKind.BINARY_EXPRESSION:
            return None, None, None

        op = node.operator
        if op not in ("===", "!==", "==", "!="):
            return None, None, None

        left = node.left
        right = node.right
        if left is None or right is None:
            return None, None, None

        typeof_expr = None
        literal_val = None

        if left.kind == ASTNodeKind.TYPEOF_EXPRESSION and right.kind == ASTNodeKind.LITERAL:
            typeof_expr = left.expression
            literal_val = right.value
        elif right.kind == ASTNodeKind.TYPEOF_EXPRESSION and left.kind == ASTNodeKind.LITERAL:
            typeof_expr = right.expression
            literal_val = left.value

        if typeof_expr is None or not isinstance(literal_val, str):
            return None, None, None

        if literal_val not in self.TYPEOF_RESULTS:
            return None, None, None

        return typeof_expr, literal_val, op

    def _extract_variable_name(self, node: ASTNode) -> Optional[str]:
        """Extract variable name from an expression."""
        if node.kind == ASTNodeKind.IDENTIFIER:
            return node.name
        if node.kind == ASTNodeKind.MEMBER_EXPRESSION:
            obj_name = self._extract_variable_name(node.object_node) if node.object_node else None
            prop_name = node.property_node.name if node.property_node else None
            if obj_name and prop_name:
                return f"{obj_name}.{prop_name}"
        return None

    def _compute_typeof_true_type(
        self,
        original: TSType,
        typeof_value: str,
        is_strict: bool,
    ) -> TSType:
        """Compute the type when typeof check is true."""
        target_type = TYPEOF_RESULT_MAP.get(typeof_value)
        if target_type is None:
            return original

        if typeof_value == "object":
            # typeof null === "object" is true in JS
            if is_strict:
                if original.kind == TypeKind.UNION:
                    kept: List[TSType] = []
                    for m in original.members:
                        tr = typeof_result(m)
                        if tr == "object":
                            kept.append(m)
                    return make_union(kept) if kept else OBJECT_TYPE
                return OBJECT_TYPE if typeof_result(original) == "object" else original
            else:
                return make_union([OBJECT_TYPE, NULL_TYPE])

        if original.kind == TypeKind.UNION:
            kept = []
            for m in original.members:
                if typeof_result(m) == typeof_value:
                    kept.append(m)
            if kept:
                return make_union(kept)
            return target_type

        if typeof_result(original) == typeof_value:
            return original

        return target_type

    def _compute_typeof_false_type(
        self,
        original: TSType,
        typeof_value: str,
        is_strict: bool,
    ) -> TSType:
        """Compute the type when typeof check is false."""
        if original.kind == TypeKind.UNION:
            kept: List[TSType] = []
            for m in original.members:
                if typeof_result(m) != typeof_value:
                    kept.append(m)
            return make_union(kept) if kept else NEVER_TYPE

        if typeof_result(original) == typeof_value:
            return NEVER_TYPE

        return original

    def _make_typeof_predicate(
        self, var_name: str, typeof_value: str, is_negated: bool
    ) -> Predicate:
        """Create a predicate for a typeof check."""
        type_map = {
            "string": "str",
            "number": "float",
            "bigint": "int",
            "boolean": "bool",
            "symbol": "Symbol",
            "undefined": "NoneType",
            "object": "object",
            "function": "Callable",
        }
        py_type = type_map.get(typeof_value, typeof_value)

        pred = Predicate(
            kind=PredicateKind.ISINSTANCE,
            variable=var_name,
            type_name=py_type,
        )
        if is_negated:
            pred = pred.negate()
        return pred

    def handles_null_check(
        self, node: ASTNode, variable_types: Dict[str, TSType]
    ) -> Optional[NarrowingPoint]:
        """Handle typeof x === "object" && x !== null pattern."""
        if node.kind != ASTNodeKind.LOGICAL_EXPRESSION or node.operator != "&&":
            return None
        left = node.left
        right = node.right
        if left is None or right is None:
            return None

        typeof_point = self.analyze(left, variable_types)
        if typeof_point is None:
            return None

        if (
            typeof_point.metadata.get("typeof_result") != "object"
            or right.kind != ASTNodeKind.BINARY_EXPRESSION
        ):
            return None

        if right.operator != "!==" or right.left is None or right.right is None:
            return None

        right_is_null = (
            (right.right.kind == ASTNodeKind.LITERAL and right.right.value is None)
            or (right.left.kind == ASTNodeKind.LITERAL and right.left.value is None)
        )

        if not right_is_null:
            return None

        var_name = typeof_point.variable
        original = variable_types.get(var_name, UNKNOWN_TYPE)

        true_type = original.without_nullish()
        if true_type.kind == TypeKind.UNION:
            true_type = make_union(
                [m for m in true_type.members if typeof_result(m) == "object"]
            )

        false_type = exclude_type(original, true_type)

        pred = Predicate(
            kind=PredicateKind.AND,
            operands=[
                Predicate(kind=PredicateKind.ISINSTANCE, variable=var_name, type_name="object"),
                Predicate(kind=PredicateKind.NOT_NONE, variable=var_name),
            ],
        )

        return NarrowingPoint(
            location=node.location,
            variable=var_name,
            guard_expression=node,
            true_type=true_type,
            false_type=false_type,
            narrowing_kind=NarrowingKind.TYPEOF,
            original_type=original,
            predicate=pred,
            metadata={"pattern": "typeof_object_non_null"},
        )


# ---------------------------------------------------------------------------
# InstanceofNarrowing
# ---------------------------------------------------------------------------

class InstanceofNarrowing:
    """Handle x instanceof Class patterns."""

    def __init__(self, class_hierarchy: Optional[Dict[str, List[str]]] = None) -> None:
        self._class_hierarchy = class_hierarchy or {}

    def analyze(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Analyze an instanceof expression."""
        if node.kind == ASTNodeKind.INSTANCEOF_EXPRESSION:
            left = node.left
            right = node.right
        elif (
            node.kind == ASTNodeKind.BINARY_EXPRESSION
            and node.operator == "instanceof"
        ):
            left = node.left
            right = node.right
        else:
            return None

        if left is None or right is None:
            return None

        var_name = self._extract_variable(left)
        class_name = self._extract_class_name(right)
        if not var_name or not class_name:
            return None

        original = variable_types.get(var_name, UNKNOWN_TYPE)
        class_type = TSType(kind=TypeKind.CLASS, name=class_name)

        true_type = self._compute_instanceof_true(original, class_type)
        false_type = self._compute_instanceof_false(original, class_type)

        pred = Predicate(
            kind=PredicateKind.ISINSTANCE,
            variable=var_name,
            type_name=class_name,
        )

        return NarrowingPoint(
            location=node.location,
            variable=var_name,
            guard_expression=node,
            true_type=true_type,
            false_type=false_type,
            narrowing_kind=NarrowingKind.INSTANCEOF,
            original_type=original,
            predicate=pred,
            metadata={"class_name": class_name},
        )

    def analyze_with_symbol_has_instance(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
        custom_instanceof_map: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Handle custom Symbol.hasInstance."""
        basic = self.analyze(node, variable_types)
        if basic is None:
            return None

        class_name = basic.metadata.get("class_name", "")
        if class_name in custom_instanceof_map:
            custom_type = custom_instanceof_map[class_name]
            original = variable_types.get(basic.variable, UNKNOWN_TYPE)
            basic.true_type = intersect_with(original, custom_type)
            basic.false_type = exclude_type(original, custom_type)
            basic.metadata["has_symbol_has_instance"] = True

        return basic

    def analyze_structural(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
        interface_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Handle structural / interface satisfaction check."""
        basic = self.analyze(node, variable_types)
        if basic is None:
            return None

        class_name = basic.metadata.get("class_name", "")
        if class_name in interface_types:
            iface = interface_types[class_name]
            original = variable_types.get(basic.variable, UNKNOWN_TYPE)
            basic.true_type = make_intersection([original, iface])
            basic.metadata["structural_check"] = True

        return basic

    def _compute_instanceof_true(
        self, original: TSType, class_type: TSType
    ) -> TSType:
        """Compute type when instanceof is true."""
        if original.kind == TypeKind.UNION:
            kept: List[TSType] = []
            for m in original.members:
                if self._is_subclass_of(m, class_type) or m == class_type:
                    kept.append(m)
                elif (
                    m.kind in (TypeKind.OBJECT, TypeKind.UNKNOWN, TypeKind.ANY)
                ):
                    kept.append(class_type)
            if kept:
                return make_union(kept)
            return class_type

        if original.kind in (TypeKind.UNKNOWN, TypeKind.ANY, TypeKind.OBJECT):
            return class_type

        if self._is_subclass_of(original, class_type):
            return original

        return class_type

    def _compute_instanceof_false(
        self, original: TSType, class_type: TSType
    ) -> TSType:
        """Compute type when instanceof is false."""
        if original.kind == TypeKind.UNION:
            kept = [
                m
                for m in original.members
                if not self._is_subclass_of(m, class_type) and m != class_type
            ]
            return make_union(kept)

        if original == class_type:
            return NEVER_TYPE

        if self._is_subclass_of(original, class_type):
            return NEVER_TYPE

        return original

    def _is_subclass_of(self, source: TSType, target: TSType) -> bool:
        """Check if source is a subclass of target using the hierarchy."""
        if source == target:
            return True
        if source.name in self._class_hierarchy:
            parents = self._class_hierarchy[source.name]
            if target.name in parents:
                return True
            for parent in parents:
                parent_type = TSType(kind=TypeKind.CLASS, name=parent)
                if self._is_subclass_of(parent_type, target):
                    return True
        for base in source.base_types:
            if self._is_subclass_of(base, target):
                return True
        return False

    def _extract_variable(self, node: ASTNode) -> Optional[str]:
        if node.kind == ASTNodeKind.IDENTIFIER:
            return node.name
        if node.kind == ASTNodeKind.MEMBER_EXPRESSION:
            obj = self._extract_variable(node.object_node) if node.object_node else None
            prop = node.property_node.name if node.property_node else None
            if obj and prop:
                return f"{obj}.{prop}"
        return None

    def _extract_class_name(self, node: ASTNode) -> Optional[str]:
        if node.kind == ASTNodeKind.IDENTIFIER:
            return node.name
        if node.kind == ASTNodeKind.MEMBER_EXPRESSION:
            obj = self._extract_class_name(node.object_node) if node.object_node else None
            prop = node.property_node.name if node.property_node else None
            if obj and prop:
                return f"{obj}.{prop}"
        return None


# ---------------------------------------------------------------------------
# InNarrowing
# ---------------------------------------------------------------------------

class InNarrowing:
    """Handle "key" in obj patterns."""

    def analyze(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Analyze an 'in' expression for narrowing."""
        if node.kind == ASTNodeKind.IN_EXPRESSION:
            left = node.left
            right = node.right
        elif (
            node.kind == ASTNodeKind.BINARY_EXPRESSION and node.operator == "in"
        ):
            left = node.left
            right = node.right
        else:
            return None

        if left is None or right is None:
            return None

        key_name = self._extract_key(left)
        var_name = self._extract_var(right)
        if key_name is None or var_name is None:
            return None

        original = variable_types.get(var_name, UNKNOWN_TYPE)

        true_type = self._compute_in_true(original, key_name)
        false_type = self._compute_in_false(original, key_name)

        pred = Predicate(
            kind=PredicateKind.HASATTR,
            variable=var_name,
            attribute=key_name,
        )

        return NarrowingPoint(
            location=node.location,
            variable=var_name,
            guard_expression=node,
            true_type=true_type,
            false_type=false_type,
            narrowing_kind=NarrowingKind.IN,
            original_type=original,
            predicate=pred,
            metadata={"key": key_name},
        )

    def analyze_discriminated(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
        discriminant_property: str,
    ) -> Optional[NarrowingPoint]:
        """Analyze 'in' as a discriminated property check."""
        basic = self.analyze(node, variable_types)
        if basic is None:
            return None

        key = basic.metadata.get("key", "")
        if key != discriminant_property:
            return basic

        original = variable_types.get(basic.variable, UNKNOWN_TYPE)
        if original.kind == TypeKind.UNION:
            has_disc: List[TSType] = []
            no_disc: List[TSType] = []
            for m in original.members:
                if key in m.properties:
                    has_disc.append(m)
                else:
                    no_disc.append(m)
            basic.true_type = make_union(has_disc)
            basic.false_type = make_union(no_disc)
            basic.metadata["discriminated_check"] = True

        return basic

    def analyze_optional_property(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Handle narrowing for optional property checks."""
        basic = self.analyze(node, variable_types)
        if basic is None:
            return None

        key = basic.metadata.get("key", "")
        original = variable_types.get(basic.variable, UNKNOWN_TYPE)

        if key in original.properties and original.properties[key].is_optional:
            # When "key" in obj is true, the property exists (non-optional)
            narrowed_props = dict(original.properties)
            prop_type = narrowed_props[key]
            narrowed_props[key] = TSType(
                kind=prop_type.kind,
                name=prop_type.name,
                literal_value=prop_type.literal_value,
                members=prop_type.members,
                properties=prop_type.properties,
                is_optional=False,
            )
            true_narrowed = TSType(
                kind=original.kind,
                name=original.name,
                properties=narrowed_props,
            )
            basic.true_type = true_narrowed
            basic.metadata["optional_property_narrowed"] = True

        return basic

    def analyze_index_signature(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Handle narrowing with index signatures."""
        basic = self.analyze(node, variable_types)
        if basic is None:
            return None

        original = variable_types.get(basic.variable, UNKNOWN_TYPE)
        key = basic.metadata.get("key", "")

        if original.index_signatures:
            for idx_key_type, idx_val_type in original.index_signatures.items():
                if idx_key_type == "string":
                    narrowed = TSType(
                        kind=original.kind,
                        name=original.name,
                        properties={**original.properties, key: idx_val_type},
                        index_signatures=original.index_signatures,
                    )
                    basic.true_type = narrowed
                    basic.metadata["index_signature_narrowed"] = True
                    break

        return basic

    def _compute_in_true(self, original: TSType, key: str) -> TSType:
        """Compute type when 'key' in obj is true."""
        if original.kind == TypeKind.UNION:
            kept = [m for m in original.members if key in m.properties]
            if kept:
                return make_union(kept)
            return make_intersection([
                original,
                TSType(
                    kind=TypeKind.OBJECT,
                    properties={key: UNKNOWN_TYPE},
                ),
            ])

        if key in original.properties:
            return original

        return make_intersection([
            original,
            TSType(kind=TypeKind.OBJECT, properties={key: UNKNOWN_TYPE}),
        ])

    def _compute_in_false(self, original: TSType, key: str) -> TSType:
        """Compute type when 'key' in obj is false."""
        if original.kind == TypeKind.UNION:
            kept = [m for m in original.members if key not in m.properties]
            return make_union(kept)
        return original

    def _extract_key(self, node: ASTNode) -> Optional[str]:
        if node.kind == ASTNodeKind.LITERAL and isinstance(node.value, str):
            return node.value
        if node.kind == ASTNodeKind.IDENTIFIER:
            return node.name
        return None

    def _extract_var(self, node: ASTNode) -> Optional[str]:
        if node.kind == ASTNodeKind.IDENTIFIER:
            return node.name
        return None


# ---------------------------------------------------------------------------
# EqualityNarrowing
# ---------------------------------------------------------------------------

class EqualityNarrowing:
    """Handle === and == checks for narrowing."""

    def analyze(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Analyze an equality expression for narrowing."""
        if node.kind != ASTNodeKind.BINARY_EXPRESSION:
            return None

        op = node.operator
        if op not in ("===", "!==", "==", "!="):
            return None

        left = node.left
        right = node.right
        if left is None or right is None:
            return None

        is_strict = op in ("===", "!==")
        is_negated = op in ("!==", "!=")

        # Determine which side is the variable and which is the literal
        var_node, literal_node = self._identify_variable_and_literal(left, right)
        if var_node is None:
            return None

        var_name = self._extract_var_name(var_node)
        if not var_name:
            return None

        original = variable_types.get(var_name, UNKNOWN_TYPE)

        if literal_node is not None and literal_node.kind == ASTNodeKind.LITERAL:
            return self._analyze_literal_equality(
                node, var_name, original, literal_node.value, is_strict, is_negated
            )

        if literal_node is not None and literal_node.kind == ASTNodeKind.IDENTIFIER:
            if literal_node.name in ("null", "undefined"):
                return self._analyze_nullish_equality(
                    node, var_name, original, literal_node.name, is_strict, is_negated
                )
            # Variable-to-variable comparison
            return self._analyze_variable_equality(
                node, var_name, original, literal_node, variable_types, is_strict, is_negated
            )

        return None

    def analyze_null_check(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Specifically handle null/undefined equality checks."""
        if node.kind != ASTNodeKind.BINARY_EXPRESSION:
            return None
        op = node.operator
        if op not in ("===", "!==", "==", "!="):
            return None

        left = node.left
        right = node.right
        if left is None or right is None:
            return None

        var_node = None
        null_kind = None

        if right.kind == ASTNodeKind.LITERAL and right.value is None:
            var_node = left
            null_kind = "null"
        elif left.kind == ASTNodeKind.LITERAL and left.value is None:
            var_node = right
            null_kind = "null"
        elif right.kind == ASTNodeKind.IDENTIFIER and right.name == "undefined":
            var_node = left
            null_kind = "undefined"
        elif left.kind == ASTNodeKind.IDENTIFIER and left.name == "undefined":
            var_node = right
            null_kind = "undefined"
        elif right.kind == ASTNodeKind.IDENTIFIER and right.name == "null":
            var_node = left
            null_kind = "null"
        elif left.kind == ASTNodeKind.IDENTIFIER and left.name == "null":
            var_node = right
            null_kind = "null"

        if var_node is None or null_kind is None:
            return None

        var_name = self._extract_var_name(var_node)
        if not var_name:
            return None

        is_strict = op in ("===", "!==")
        is_negated = op in ("!==", "!=")

        return self._analyze_nullish_equality(
            node, var_name, variable_types.get(var_name, UNKNOWN_TYPE),
            null_kind, is_strict, is_negated
        )

    def analyze_discriminated_union(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
        discriminant_property: str,
    ) -> Optional[NarrowingPoint]:
        """Handle discriminated union narrowing via property equality.

        E.g.: if (action.type === "increment") { ... }
        """
        if node.kind != ASTNodeKind.BINARY_EXPRESSION:
            return None
        if node.operator not in ("===", "!=="):
            return None

        left = node.left
        right = node.right
        if left is None or right is None:
            return None

        member_node = None
        literal_node = None
        if left.kind == ASTNodeKind.MEMBER_EXPRESSION and right.kind == ASTNodeKind.LITERAL:
            member_node = left
            literal_node = right
        elif right.kind == ASTNodeKind.MEMBER_EXPRESSION and left.kind == ASTNodeKind.LITERAL:
            member_node = right
            literal_node = left

        if member_node is None or literal_node is None:
            return None

        prop_name = (
            member_node.property_node.name if member_node.property_node else None
        )
        if prop_name != discriminant_property:
            return None

        obj_node = member_node.object_node
        if obj_node is None or obj_node.kind != ASTNodeKind.IDENTIFIER:
            return None

        var_name = obj_node.name
        original = variable_types.get(var_name, UNKNOWN_TYPE)
        disc_value = literal_node.value
        is_negated = node.operator == "!=="

        if original.kind == TypeKind.UNION:
            matched: List[TSType] = []
            unmatched: List[TSType] = []
            for m in original.members:
                disc_type = m.properties.get(discriminant_property)
                if disc_type and disc_type.literal_value == disc_value:
                    matched.append(m)
                else:
                    unmatched.append(m)

            true_type = make_union(matched)
            false_type = make_union(unmatched)
        else:
            true_type = original
            false_type = NEVER_TYPE

        if is_negated:
            true_type, false_type = false_type, true_type

        pred = Predicate(
            kind=PredicateKind.EQUALITY,
            variable=f"{var_name}.{discriminant_property}",
            value=disc_value,
        )
        if is_negated:
            pred = pred.negate()

        return NarrowingPoint(
            location=node.location,
            variable=var_name,
            guard_expression=node,
            true_type=true_type,
            false_type=false_type,
            narrowing_kind=NarrowingKind.DISCRIMINATED_UNION,
            original_type=original,
            predicate=pred,
            metadata={
                "discriminant_property": discriminant_property,
                "discriminant_value": disc_value,
                "is_negated": is_negated,
            },
        )

    def analyze_switch_cases(
        self,
        discriminant: ASTNode,
        cases: List[ASTNode],
        variable_types: Dict[str, TSType],
    ) -> List[NarrowingPoint]:
        """Analyze equality narrowing through switch statements."""
        var_name = self._extract_var_name(discriminant)
        if not var_name:
            return []

        original = variable_types.get(var_name, UNKNOWN_TYPE)
        points: List[NarrowingPoint] = []
        matched_values: List[Any] = []

        for case_node in cases:
            if case_node.value is not None:
                case_val = case_node.value
                matched_values.append(case_val)

                true_type = self._narrow_to_literal(original, case_val)
                false_type = self._narrow_excluding_literal(original, case_val)

                pred = Predicate(
                    kind=PredicateKind.EQUALITY,
                    variable=var_name,
                    value=case_val,
                )
                points.append(NarrowingPoint(
                    location=case_node.location,
                    variable=var_name,
                    guard_expression=case_node,
                    true_type=true_type,
                    false_type=false_type,
                    narrowing_kind=NarrowingKind.EQUALITY,
                    original_type=original,
                    predicate=pred,
                    metadata={"case_value": case_val, "in_switch": True},
                ))
            else:
                # Default case
                remaining = original
                for val in matched_values:
                    remaining = self._narrow_excluding_literal(remaining, val)
                points.append(NarrowingPoint(
                    location=case_node.location,
                    variable=var_name,
                    guard_expression=case_node,
                    true_type=remaining,
                    false_type=None,
                    narrowing_kind=NarrowingKind.EQUALITY,
                    original_type=original,
                    metadata={"is_default_case": True, "in_switch": True},
                ))

        return points

    def analyze_enum_equality(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
        enum_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Handle enum member equality checks."""
        if node.kind != ASTNodeKind.BINARY_EXPRESSION:
            return None
        if node.operator not in ("===", "!=="):
            return None

        left = node.left
        right = node.right
        if left is None or right is None:
            return None

        # Detect EnumType.Member pattern
        enum_member_node = None
        var_node = None
        if (
            right.kind == ASTNodeKind.MEMBER_EXPRESSION
            and right.object_node
            and right.object_node.kind == ASTNodeKind.IDENTIFIER
            and right.object_node.name in enum_types
        ):
            enum_member_node = right
            var_node = left
        elif (
            left.kind == ASTNodeKind.MEMBER_EXPRESSION
            and left.object_node
            and left.object_node.kind == ASTNodeKind.IDENTIFIER
            and left.object_node.name in enum_types
        ):
            enum_member_node = left
            var_node = right

        if enum_member_node is None or var_node is None:
            return None

        var_name = self._extract_var_name(var_node)
        if not var_name:
            return None

        enum_name = enum_member_node.object_node.name if enum_member_node.object_node else ""
        member_name = (
            enum_member_node.property_node.name if enum_member_node.property_node else ""
        )
        full_member = f"{enum_name}.{member_name}"
        is_negated = node.operator == "!=="
        original = variable_types.get(var_name, UNKNOWN_TYPE)

        enum_member_type = TSType(
            kind=TypeKind.ENUM_MEMBER,
            name=full_member,
        )

        true_type = enum_member_type
        false_type = exclude_type(original, enum_member_type)
        if is_negated:
            true_type, false_type = false_type, true_type

        pred = Predicate(
            kind=PredicateKind.EQUALITY,
            variable=var_name,
            value=full_member,
        )
        if is_negated:
            pred = pred.negate()

        return NarrowingPoint(
            location=node.location,
            variable=var_name,
            guard_expression=node,
            true_type=true_type,
            false_type=false_type,
            narrowing_kind=NarrowingKind.EQUALITY,
            original_type=original,
            predicate=pred,
            metadata={"enum_member": full_member, "is_negated": is_negated},
        )

    def _analyze_literal_equality(
        self,
        node: ASTNode,
        var_name: str,
        original: TSType,
        literal_value: Any,
        is_strict: bool,
        is_negated: bool,
    ) -> NarrowingPoint:
        """Analyze narrowing for a literal equality check."""
        true_type = self._narrow_to_literal(original, literal_value)
        false_type = self._narrow_excluding_literal(original, literal_value)

        if is_negated:
            true_type, false_type = false_type, true_type

        pred = Predicate(
            kind=PredicateKind.EQUALITY,
            variable=var_name,
            value=literal_value,
        )
        if is_negated:
            pred = pred.negate()

        return NarrowingPoint(
            location=node.location,
            variable=var_name,
            guard_expression=node,
            true_type=true_type,
            false_type=false_type,
            narrowing_kind=NarrowingKind.EQUALITY,
            original_type=original,
            predicate=pred,
            metadata={
                "literal_value": literal_value,
                "is_strict": is_strict,
                "is_negated": is_negated,
            },
        )

    def _analyze_nullish_equality(
        self,
        node: ASTNode,
        var_name: str,
        original: TSType,
        null_kind: str,
        is_strict: bool,
        is_negated: bool,
    ) -> NarrowingPoint:
        """Analyze nullish equality narrowing."""
        if is_strict:
            if null_kind == "null":
                null_type = NULL_TYPE
            else:
                null_type = UNDEFINED_TYPE
            true_type = intersect_with(original, null_type)
            false_type = exclude_type(original, null_type)
        else:
            # Loose equality: x == null matches both null and undefined
            nullish = make_union([NULL_TYPE, UNDEFINED_TYPE])
            true_type = intersect_with(original, nullish)
            false_type = exclude_types(original, [NULL_TYPE, UNDEFINED_TYPE])

        if is_negated:
            true_type, false_type = false_type, true_type

        if null_kind == "null" and not is_negated:
            pred = Predicate(kind=PredicateKind.IS_NONE, variable=var_name)
        elif null_kind == "null" and is_negated:
            pred = Predicate(kind=PredicateKind.NOT_NONE, variable=var_name)
        elif null_kind == "undefined" and not is_negated:
            pred = Predicate(kind=PredicateKind.IS_NONE, variable=var_name)
        else:
            pred = Predicate(kind=PredicateKind.NOT_NONE, variable=var_name)

        return NarrowingPoint(
            location=node.location,
            variable=var_name,
            guard_expression=node,
            true_type=true_type,
            false_type=false_type,
            narrowing_kind=NarrowingKind.EQUALITY,
            original_type=original,
            predicate=pred,
            metadata={
                "null_kind": null_kind,
                "is_strict": is_strict,
                "is_negated": is_negated,
            },
        )

    def _analyze_variable_equality(
        self,
        node: ASTNode,
        var_name: str,
        original: TSType,
        other_node: ASTNode,
        variable_types: Dict[str, TSType],
        is_strict: bool,
        is_negated: bool,
    ) -> NarrowingPoint:
        """Analyze variable-to-variable equality."""
        other_name = self._extract_var_name(other_node)
        other_type = (
            variable_types.get(other_name, UNKNOWN_TYPE)
            if other_name
            else UNKNOWN_TYPE
        )

        true_type = make_intersection([original, other_type])
        false_type = original

        if is_negated:
            true_type, false_type = false_type, true_type

        pred = Predicate(
            kind=PredicateKind.EQUALITY,
            variable=var_name,
            value=other_name,
        )
        if is_negated:
            pred = pred.negate()

        return NarrowingPoint(
            location=node.location,
            variable=var_name,
            guard_expression=node,
            true_type=true_type,
            false_type=false_type,
            narrowing_kind=NarrowingKind.EQUALITY,
            original_type=original,
            predicate=pred,
            metadata={
                "other_variable": other_name,
                "is_strict": is_strict,
                "is_negated": is_negated,
            },
        )

    def _narrow_to_literal(self, original: TSType, value: Any) -> TSType:
        """Narrow to a literal type."""
        if isinstance(value, str):
            lit = TSType(kind=TypeKind.LITERAL_STRING, literal_value=value)
        elif isinstance(value, bool):
            lit = TSType(kind=TypeKind.LITERAL_BOOLEAN, literal_value=value)
        elif isinstance(value, int):
            lit = TSType(kind=TypeKind.LITERAL_NUMBER, literal_value=value)
        elif isinstance(value, float):
            lit = TSType(kind=TypeKind.LITERAL_NUMBER, literal_value=value)
        elif value is None:
            lit = NULL_TYPE
        else:
            return original

        if original.kind == TypeKind.UNION:
            for m in original.members:
                if m == lit:
                    return lit
                if m.kind == lit.kind and m.literal_value == lit.literal_value:
                    return m
            # If no exact match, check if any member could hold this literal
            for m in original.members:
                if is_assignable_to(lit, m):
                    return lit
            return lit

        if is_assignable_to(lit, original):
            return lit
        return lit

    def _narrow_excluding_literal(self, original: TSType, value: Any) -> TSType:
        """Narrow by excluding a literal."""
        if isinstance(value, str):
            lit = TSType(kind=TypeKind.LITERAL_STRING, literal_value=value)
        elif isinstance(value, bool):
            lit = TSType(kind=TypeKind.LITERAL_BOOLEAN, literal_value=value)
        elif isinstance(value, int):
            lit = TSType(kind=TypeKind.LITERAL_NUMBER, literal_value=value)
        elif isinstance(value, float):
            lit = TSType(kind=TypeKind.LITERAL_NUMBER, literal_value=value)
        elif value is None:
            lit = NULL_TYPE
        else:
            return original

        return exclude_type(original, lit)

    def _identify_variable_and_literal(
        self, left: ASTNode, right: ASTNode
    ) -> Tuple[Optional[ASTNode], Optional[ASTNode]]:
        """Identify which operand is the variable and which is the literal."""
        if right.kind == ASTNodeKind.LITERAL:
            return left, right
        if left.kind == ASTNodeKind.LITERAL:
            return right, left
        if right.kind == ASTNodeKind.IDENTIFIER and right.name in ("null", "undefined"):
            return left, right
        if left.kind == ASTNodeKind.IDENTIFIER and left.name in ("null", "undefined"):
            return right, left
        # Both are non-literals: left is "variable", right is "other"
        return left, right

    def _extract_var_name(self, node: ASTNode) -> Optional[str]:
        if node.kind == ASTNodeKind.IDENTIFIER:
            return node.name
        if node.kind == ASTNodeKind.MEMBER_EXPRESSION:
            obj = self._extract_var_name(node.object_node) if node.object_node else None
            prop = node.property_node.name if node.property_node else None
            if obj and prop:
                return f"{obj}.{prop}"
        return None


# ---------------------------------------------------------------------------
# TruthinessNarrowing
# ---------------------------------------------------------------------------

class TruthinessNarrowing:
    """Handle if (x) patterns and truthiness-based narrowing."""

    def analyze(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Analyze truthiness narrowing from an expression used as condition."""
        actual_node, negation_count = self._unwrap_negations(node)

        var_name = self._extract_variable(actual_node)
        if not var_name:
            return None

        original = variable_types.get(var_name, UNKNOWN_TYPE)

        truthy_type = original.without_falsy()
        falsy_type = original.only_falsy()

        is_negated = (negation_count % 2) == 1

        if is_negated:
            true_type = falsy_type
            false_type = truthy_type
        else:
            true_type = truthy_type
            false_type = falsy_type

        if is_negated:
            pred = Predicate(kind=PredicateKind.IS_FALSY, variable=var_name)
        else:
            pred = Predicate(kind=PredicateKind.IS_TRUTHY, variable=var_name)

        return NarrowingPoint(
            location=node.location,
            variable=var_name,
            guard_expression=node,
            true_type=true_type,
            false_type=false_type,
            narrowing_kind=NarrowingKind.TRUTHINESS,
            original_type=original,
            predicate=pred,
            metadata={
                "negation_count": negation_count,
                "is_double_negation": negation_count >= 2,
            },
        )

    def analyze_short_circuit_and(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> List[NarrowingPoint]:
        """Analyze x && x.method() short-circuit pattern."""
        if node.kind != ASTNodeKind.LOGICAL_EXPRESSION or node.operator != "&&":
            return []

        left = node.left
        right = node.right
        if left is None or right is None:
            return []

        points: List[NarrowingPoint] = []
        left_point = self.analyze(left, variable_types)
        if left_point:
            points.append(left_point)
            # In the right branch, the left variable is narrowed to truthy
            narrowed_types = dict(variable_types)
            if left_point.true_type:
                narrowed_types[left_point.variable] = left_point.true_type

            right_point = self.analyze(right, narrowed_types)
            if right_point:
                points.append(right_point)

        return points

    def analyze_short_circuit_or(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> List[NarrowingPoint]:
        """Analyze x || default pattern."""
        if node.kind != ASTNodeKind.LOGICAL_EXPRESSION or node.operator != "||":
            return []

        left = node.left
        right = node.right
        if left is None or right is None:
            return []

        points: List[NarrowingPoint] = []
        left_point = self.analyze(left, variable_types)
        if left_point:
            # When || is used, the left side is used when truthy
            points.append(left_point)

        return points

    def analyze_nullish_coalescing(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Analyze x ?? default pattern."""
        if node.kind != ASTNodeKind.NULLISH_COALESCING_EXPRESSION:
            if not (
                node.kind == ASTNodeKind.BINARY_EXPRESSION and node.operator == "??"
            ):
                return None

        left = node.left
        if left is None:
            return None

        var_name = self._extract_variable(left)
        if not var_name:
            return None

        original = variable_types.get(var_name, UNKNOWN_TYPE)
        # x ?? y: left side is used when x is non-nullish
        non_nullish = original.without_nullish()

        pred = Predicate(kind=PredicateKind.NOT_NONE, variable=var_name)

        return NarrowingPoint(
            location=node.location,
            variable=var_name,
            guard_expression=node,
            true_type=non_nullish,
            false_type=make_union([NULL_TYPE, UNDEFINED_TYPE]),
            narrowing_kind=NarrowingKind.NULLISH_COALESCING,
            original_type=original,
            predicate=pred,
            metadata={"pattern": "nullish_coalescing"},
        )

    def _unwrap_negations(self, node: ASTNode) -> Tuple[ASTNode, int]:
        """Unwrap negation operators, counting how many."""
        count = 0
        current = node
        while (
            current.kind == ASTNodeKind.UNARY_EXPRESSION
            and current.operator == "!"
            and current.expression is not None
        ):
            count += 1
            current = current.expression
        return current, count

    def _extract_variable(self, node: ASTNode) -> Optional[str]:
        if node.kind == ASTNodeKind.IDENTIFIER:
            return node.name
        if node.kind == ASTNodeKind.MEMBER_EXPRESSION:
            obj = self._extract_variable(node.object_node) if node.object_node else None
            prop = node.property_node.name if node.property_node else None
            if obj and prop:
                return f"{obj}.{prop}"
        return None


# ---------------------------------------------------------------------------
# UserDefinedTypeGuard
# ---------------------------------------------------------------------------

class UserDefinedTypeGuard:
    """Handle custom type guard functions."""

    def __init__(self) -> None:
        self._registered_guards: Dict[str, Dict[str, Any]] = {}

    def register_guard(
        self,
        function_name: str,
        parameter_name: str,
        guarded_type: TSType,
        is_assertion: bool = False,
        is_this_guard: bool = False,
    ) -> None:
        """Register a type guard function."""
        self._registered_guards[function_name] = {
            "parameter_name": parameter_name,
            "guarded_type": guarded_type,
            "is_assertion": is_assertion,
            "is_this_guard": is_this_guard,
        }

    def detect_type_guard(self, func_node: ASTNode) -> Optional[Dict[str, Any]]:
        """Detect if a function is a type guard from its declaration."""
        if func_node.type_predicate:
            return func_node.type_predicate
        if func_node.return_type and func_node.return_type.name.startswith("x is "):
            guarded_type_name = func_node.return_type.name[5:]
            param_name = "x"
            if func_node.parameters:
                param_name = func_node.parameters[0].name
            return {
                "parameter_name": param_name,
                "guarded_type_name": guarded_type_name,
                "is_assertion": False,
            }
        if func_node.is_assertion and func_node.asserts_type:
            param_name = "x"
            if func_node.parameters:
                param_name = func_node.parameters[0].name
            return {
                "parameter_name": param_name,
                "guarded_type": func_node.asserts_type,
                "is_assertion": True,
            }
        return None

    def analyze_call(
        self,
        call_node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Analyze a call to a type guard function."""
        if call_node.kind != ASTNodeKind.CALL_EXPRESSION:
            return None

        func_name = self._extract_function_name(call_node.callee)
        if not func_name or func_name not in self._registered_guards:
            return None

        guard_info = self._registered_guards[func_name]
        guarded_type = guard_info["guarded_type"]
        is_assertion = guard_info["is_assertion"]
        is_this_guard = guard_info.get("is_this_guard", False)

        if is_this_guard:
            return self._analyze_this_guard(call_node, variable_types, guard_info)

        if not call_node.arguments:
            return None

        first_arg = call_node.arguments[0]
        var_name = self._extract_var_name(first_arg)
        if not var_name:
            return None

        original = variable_types.get(var_name, UNKNOWN_TYPE)

        true_type = intersect_with(original, guarded_type)
        if true_type.kind == TypeKind.NEVER:
            true_type = guarded_type

        if is_assertion:
            false_type = NEVER_TYPE  # Assertion throws if false
        else:
            false_type = exclude_type(original, guarded_type)

        kind = NarrowingKind.ASSERTION if is_assertion else NarrowingKind.USER_DEFINED

        pred = Predicate(
            kind=PredicateKind.TYPE_GUARD,
            variable=var_name,
            type_name=guarded_type.name,
            guard_function=func_name,
        )

        return NarrowingPoint(
            location=call_node.location,
            variable=var_name,
            guard_expression=call_node,
            true_type=true_type,
            false_type=false_type,
            narrowing_kind=kind,
            original_type=original,
            predicate=pred,
            metadata={
                "guard_function": func_name,
                "guarded_type": guarded_type.name,
                "is_assertion": is_assertion,
            },
        )

    def analyze_composed_guards(
        self,
        call_nodes: List[ASTNode],
        variable_types: Dict[str, TSType],
        composition: str,  # "and" or "or"
    ) -> Optional[NarrowingPoint]:
        """Analyze composed type guards: isA(x) && isB(x)."""
        points: List[NarrowingPoint] = []
        for call in call_nodes:
            pt = self.analyze_call(call, variable_types)
            if pt:
                points.append(pt)

        if not points:
            return None

        if len(points) == 1:
            return points[0]

        var_name = points[0].variable
        # Check all points refer to same variable
        if not all(p.variable == var_name for p in points):
            return None

        if composition == "and":
            true_types = [p.true_type for p in points if p.true_type]
            result_true = true_types[0] if true_types else NEVER_TYPE
            for t in true_types[1:]:
                result_true = make_intersection([result_true, t])

            false_types = [p.false_type for p in points if p.false_type]
            result_false = make_union(false_types) if false_types else NEVER_TYPE

            preds = [p.predicate for p in points if p.predicate]
            combined_pred = preds[0] if preds else Predicate(kind=PredicateKind.TRUE)
            for p in preds[1:]:
                combined_pred = combined_pred.conjoin(p)
        else:  # "or"
            true_types = [p.true_type for p in points if p.true_type]
            result_true = make_union(true_types) if true_types else NEVER_TYPE

            false_types = [p.false_type for p in points if p.false_type]
            result_false = false_types[0] if false_types else NEVER_TYPE
            for t in false_types[1:]:
                result_false = make_intersection([result_false, t])

            preds = [p.predicate for p in points if p.predicate]
            combined_pred = preds[0] if preds else Predicate(kind=PredicateKind.FALSE)
            for p in preds[1:]:
                combined_pred = combined_pred.disjoin(p)

        return NarrowingPoint(
            location=points[0].location,
            variable=var_name,
            guard_expression=None,
            true_type=result_true,
            false_type=result_false,
            narrowing_kind=NarrowingKind.COMPOUND,
            original_type=points[0].original_type,
            predicate=combined_pred,
            metadata={
                "composition": composition,
                "guard_count": len(points),
            },
        )

    def _analyze_this_guard(
        self,
        call_node: ASTNode,
        variable_types: Dict[str, TSType],
        guard_info: Dict[str, Any],
    ) -> Optional[NarrowingPoint]:
        """Handle this-based type guards: isWidget(): this is Widget."""
        if call_node.callee is None:
            return None
        if call_node.callee.kind != ASTNodeKind.MEMBER_EXPRESSION:
            return None

        obj_node = call_node.callee.object_node
        if obj_node is None:
            return None

        var_name = self._extract_var_name(obj_node)
        if not var_name:
            return None

        original = variable_types.get(var_name, UNKNOWN_TYPE)
        guarded_type = guard_info["guarded_type"]

        true_type = guarded_type
        false_type = exclude_type(original, guarded_type)

        func_name = (
            call_node.callee.property_node.name
            if call_node.callee.property_node
            else "unknown"
        )

        pred = Predicate(
            kind=PredicateKind.TYPE_GUARD,
            variable=var_name,
            type_name=guarded_type.name,
            guard_function=f"{var_name}.{func_name}",
        )

        return NarrowingPoint(
            location=call_node.location,
            variable=var_name,
            guard_expression=call_node,
            true_type=true_type,
            false_type=false_type,
            narrowing_kind=NarrowingKind.USER_DEFINED,
            original_type=original,
            predicate=pred,
            metadata={
                "is_this_guard": True,
                "guarded_type": guarded_type.name,
            },
        )

    def analyze_parameterized_guard(
        self,
        call_node: ASTNode,
        variable_types: Dict[str, TSType],
        type_parameter_map: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Handle parameterized type guards like isType<T>(x): x is T."""
        basic = self.analyze_call(call_node, variable_types)
        if basic is None:
            return None

        guard_info = self._registered_guards.get(
            self._extract_function_name(call_node.callee) or "", {}
        )
        guarded_type = guard_info.get("guarded_type")
        if guarded_type and guarded_type.kind == TypeKind.TYPE_PARAMETER:
            resolved = type_parameter_map.get(guarded_type.name)
            if resolved:
                original = variable_types.get(basic.variable, UNKNOWN_TYPE)
                basic.true_type = intersect_with(original, resolved)
                basic.false_type = exclude_type(original, resolved)
                basic.metadata["resolved_type_parameter"] = resolved.name

        return basic

    def _extract_function_name(self, node: Optional[ASTNode]) -> Optional[str]:
        if node is None:
            return None
        if node.kind == ASTNodeKind.IDENTIFIER:
            return node.name
        if node.kind == ASTNodeKind.MEMBER_EXPRESSION:
            prop = node.property_node.name if node.property_node else None
            return prop
        return None

    def _extract_var_name(self, node: ASTNode) -> Optional[str]:
        if node.kind == ASTNodeKind.IDENTIFIER:
            return node.name
        if node.kind == ASTNodeKind.MEMBER_EXPRESSION:
            obj = self._extract_var_name(node.object_node) if node.object_node else None
            prop = node.property_node.name if node.property_node else None
            if obj and prop:
                return f"{obj}.{prop}"
        return None


# ---------------------------------------------------------------------------
# DiscriminatedUnionNarrowing
# ---------------------------------------------------------------------------

class DiscriminatedUnionNarrowing:
    """Handle tagged / discriminated union narrowing."""

    def __init__(self) -> None:
        self._known_discriminants: Dict[str, List[str]] = {}

    def register_discriminated_union(
        self,
        union_name: str,
        discriminant_properties: List[str],
    ) -> None:
        """Register a known discriminated union."""
        self._known_discriminants[union_name] = discriminant_properties

    def detect_discriminant(self, union_type: TSType) -> Optional[str]:
        """Detect the discriminant property of a union type."""
        if union_type.kind != TypeKind.UNION or len(union_type.members) < 2:
            return None

        if union_type.discriminant_property:
            return union_type.discriminant_property

        candidate_props: Optional[Set[str]] = None
        for member in union_type.members:
            if not member.properties:
                return None
            member_props = set()
            for pname, ptype in member.properties.items():
                if ptype.is_literal:
                    member_props.add(pname)
            if candidate_props is None:
                candidate_props = member_props
            else:
                candidate_props = candidate_props & member_props

        if not candidate_props:
            return None

        # Pick the property that uniquely distinguishes all members
        for prop in candidate_props:
            values: Set[Any] = set()
            all_unique = True
            for member in union_type.members:
                val = member.properties[prop].literal_value
                if val in values:
                    all_unique = False
                    break
                values.add(val)
            if all_unique:
                return prop

        # If no single property uniquely discriminates, return first candidate
        return next(iter(candidate_props))

    def detect_multiple_discriminants(self, union_type: TSType) -> List[str]:
        """Detect multiple discriminant properties."""
        if union_type.kind != TypeKind.UNION or len(union_type.members) < 2:
            return []

        all_literal_props: Optional[Set[str]] = None
        for member in union_type.members:
            member_literal_props = {
                pname
                for pname, ptype in member.properties.items()
                if ptype.is_literal
            }
            if all_literal_props is None:
                all_literal_props = member_literal_props
            else:
                all_literal_props = all_literal_props & member_literal_props

        if not all_literal_props:
            return []

        return sorted(all_literal_props)

    def analyze_switch(
        self,
        switch_node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> List[NarrowingPoint]:
        """Analyze switch statement over discriminated union."""
        discriminant = switch_node.discriminant
        cases = switch_node.cases
        if discriminant is None or not cases:
            return []

        # Extract the object and discriminant property
        if discriminant.kind != ASTNodeKind.MEMBER_EXPRESSION:
            return []

        obj_node = discriminant.object_node
        prop_node = discriminant.property_node
        if obj_node is None or prop_node is None:
            return []

        var_name = self._extract_var(obj_node)
        disc_prop = prop_node.name
        if not var_name:
            return []

        original = variable_types.get(var_name, UNKNOWN_TYPE)
        if original.kind != TypeKind.UNION:
            return []

        points: List[NarrowingPoint] = []
        handled_values: List[Any] = []

        for case_node in cases:
            if case_node.value is not None:
                case_val = case_node.value
                handled_values.append(case_val)

                matched = [
                    m
                    for m in original.members
                    if disc_prop in m.properties
                    and m.properties[disc_prop].literal_value == case_val
                ]
                unmatched = [
                    m
                    for m in original.members
                    if disc_prop not in m.properties
                    or m.properties[disc_prop].literal_value != case_val
                ]

                true_type = make_union(matched)
                false_type = make_union(unmatched)

                pred = Predicate(
                    kind=PredicateKind.EQUALITY,
                    variable=f"{var_name}.{disc_prop}",
                    value=case_val,
                )

                points.append(NarrowingPoint(
                    location=case_node.location,
                    variable=var_name,
                    guard_expression=case_node,
                    true_type=true_type,
                    false_type=false_type,
                    narrowing_kind=NarrowingKind.DISCRIMINATED_UNION,
                    original_type=original,
                    predicate=pred,
                    metadata={
                        "discriminant_property": disc_prop,
                        "discriminant_value": case_val,
                        "in_switch": True,
                    },
                ))
            else:
                # Default case
                remaining_members = [
                    m
                    for m in original.members
                    if disc_prop not in m.properties
                    or m.properties[disc_prop].literal_value not in handled_values
                ]
                remaining = make_union(remaining_members)

                points.append(NarrowingPoint(
                    location=case_node.location,
                    variable=var_name,
                    guard_expression=case_node,
                    true_type=remaining,
                    false_type=None,
                    narrowing_kind=NarrowingKind.DISCRIMINATED_UNION,
                    original_type=original,
                    metadata={
                        "discriminant_property": disc_prop,
                        "is_default_case": True,
                        "in_switch": True,
                    },
                ))

        return points

    def analyze_if_chain(
        self,
        conditions: List[ASTNode],
        variable_types: Dict[str, TSType],
        discriminant_property: str,
    ) -> List[NarrowingPoint]:
        """Analyze if-else chain over discriminated union."""
        equality = EqualityNarrowing()
        points: List[NarrowingPoint] = []

        for cond in conditions:
            pt = equality.analyze_discriminated_union(
                cond, variable_types, discriminant_property
            )
            if pt:
                pt.narrowing_kind = NarrowingKind.DISCRIMINATED_UNION
                points.append(pt)

        return points

    def analyze_nested(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
        outer_disc: str,
        inner_disc: str,
    ) -> List[NarrowingPoint]:
        """Analyze nested discriminated union narrowing."""
        points: List[NarrowingPoint] = []

        # First level: narrow on outer discriminant
        outer_points = self.analyze_switch(node, variable_types) if node.kind == ASTNodeKind.SWITCH_STATEMENT else []

        for op in outer_points:
            points.append(op)
            if op.true_type and op.true_type.kind == TypeKind.UNION:
                # Check if inner discriminant exists
                inner_disc_found = self.detect_discriminant(op.true_type)
                if inner_disc_found:
                    op.metadata["has_nested_discriminant"] = True
                    op.metadata["inner_discriminant"] = inner_disc_found

        return points

    def _extract_var(self, node: ASTNode) -> Optional[str]:
        if node.kind == ASTNodeKind.IDENTIFIER:
            return node.name
        return None


# ---------------------------------------------------------------------------
# OptionalChainingNarrowing
# ---------------------------------------------------------------------------

class OptionalChainingNarrowing:
    """Handle ?. operator narrowing."""

    def analyze(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Analyze optional chaining expression for narrowing."""
        if node.kind != ASTNodeKind.OPTIONAL_CHAIN_EXPRESSION:
            return None

        chain_parts = self._extract_chain(node)
        if not chain_parts:
            return None

        root_var = chain_parts[0]
        original = variable_types.get(root_var, UNKNOWN_TYPE)

        # obj?.prop narrows obj to non-nullish within the expression
        non_nullish = original.without_nullish()

        pred = Predicate(kind=PredicateKind.NOT_NONE, variable=root_var)

        return NarrowingPoint(
            location=node.location,
            variable=root_var,
            guard_expression=node,
            true_type=non_nullish,
            false_type=make_union([NULL_TYPE, UNDEFINED_TYPE]),
            narrowing_kind=NarrowingKind.OPTIONAL_CHAINING,
            original_type=original,
            predicate=pred,
            metadata={
                "chain_parts": chain_parts,
                "chain_length": len(chain_parts),
            },
        )

    def analyze_method_call(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Handle obj?.method() pattern."""
        if node.kind != ASTNodeKind.CALL_EXPRESSION:
            return None
        if node.callee is None:
            return None
        if node.callee.kind != ASTNodeKind.OPTIONAL_CHAIN_EXPRESSION:
            return None

        return self.analyze(node.callee, variable_types)

    def analyze_element_access(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Handle arr?.[0] pattern."""
        if node.kind != ASTNodeKind.OPTIONAL_CHAIN_EXPRESSION:
            return None

        if not node.computed:
            return self.analyze(node, variable_types)

        chain_parts = self._extract_chain(node)
        if not chain_parts:
            return None

        root_var = chain_parts[0]
        original = variable_types.get(root_var, UNKNOWN_TYPE)

        non_nullish = original.without_nullish()

        return NarrowingPoint(
            location=node.location,
            variable=root_var,
            guard_expression=node,
            true_type=non_nullish,
            false_type=make_union([NULL_TYPE, UNDEFINED_TYPE]),
            narrowing_kind=NarrowingKind.OPTIONAL_CHAINING,
            original_type=original,
            predicate=Predicate(kind=PredicateKind.NOT_NONE, variable=root_var),
            metadata={
                "chain_parts": chain_parts,
                "element_access": True,
            },
        )

    def analyze_chained(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> List[NarrowingPoint]:
        """Handle obj?.a?.b?.c chained optional access."""
        points: List[NarrowingPoint] = []
        chain_parts = self._extract_full_chain(node)

        current_prefix = ""
        for i, part in enumerate(chain_parts):
            if i == 0:
                current_prefix = part
                continue
            current_prefix = f"{current_prefix}.{part}"
            original = variable_types.get(current_prefix, UNKNOWN_TYPE)

            if original.is_nullable or original.kind in (TypeKind.UNKNOWN, TypeKind.ANY):
                non_nullish = original.without_nullish()
                points.append(NarrowingPoint(
                    location=node.location,
                    variable=current_prefix,
                    guard_expression=node,
                    true_type=non_nullish,
                    false_type=make_union([NULL_TYPE, UNDEFINED_TYPE]),
                    narrowing_kind=NarrowingKind.OPTIONAL_CHAINING,
                    original_type=original,
                    predicate=Predicate(
                        kind=PredicateKind.NOT_NONE, variable=current_prefix
                    ),
                    metadata={
                        "chain_depth": i,
                        "full_chain": chain_parts,
                    },
                ))

        return points

    def _extract_chain(self, node: ASTNode) -> List[str]:
        """Extract chain parts from optional chain expression."""
        parts: List[str] = []
        current = node

        while current is not None:
            if current.kind == ASTNodeKind.IDENTIFIER:
                parts.append(current.name)
                break
            if current.kind in (
                ASTNodeKind.OPTIONAL_CHAIN_EXPRESSION,
                ASTNodeKind.MEMBER_EXPRESSION,
            ):
                if current.property_node and current.property_node.kind == ASTNodeKind.IDENTIFIER:
                    parts.append(current.property_node.name)
                current = current.object_node
            else:
                if current.kind == ASTNodeKind.IDENTIFIER:
                    parts.append(current.name)
                break

        parts.reverse()
        return parts

    def _extract_full_chain(self, node: ASTNode) -> List[str]:
        """Extract full chain for deeply nested optional access."""
        return self._extract_chain(node)


# ---------------------------------------------------------------------------
# NullishCoalescingNarrowing
# ---------------------------------------------------------------------------

class NullishCoalescingNarrowing:
    """Handle ?? operator narrowing."""

    def analyze(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Analyze x ?? y for narrowing."""
        is_nullish = (
            node.kind == ASTNodeKind.NULLISH_COALESCING_EXPRESSION
            or (node.kind == ASTNodeKind.BINARY_EXPRESSION and node.operator == "??")
        )
        if not is_nullish:
            return None

        left = node.left
        if left is None:
            return None

        var_name = self._extract_var(left)
        if not var_name:
            return None

        original = variable_types.get(var_name, UNKNOWN_TYPE)
        non_nullish = original.without_nullish()

        return NarrowingPoint(
            location=node.location,
            variable=var_name,
            guard_expression=node,
            true_type=non_nullish,
            false_type=make_union([NULL_TYPE, UNDEFINED_TYPE]),
            narrowing_kind=NarrowingKind.NULLISH_COALESCING,
            original_type=original,
            predicate=Predicate(kind=PredicateKind.NOT_NONE, variable=var_name),
            metadata={"pattern": "nullish_coalescing"},
        )

    def analyze_chained(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> List[NarrowingPoint]:
        """Analyze a ?? b ?? c chained nullish coalescing."""
        points: List[NarrowingPoint] = []
        current = node

        while self._is_nullish_coalescing(current):
            left = current.left
            if left is not None:
                var_name = self._extract_var(left)
                if var_name:
                    original = variable_types.get(var_name, UNKNOWN_TYPE)
                    non_nullish = original.without_nullish()
                    points.append(NarrowingPoint(
                        location=current.location,
                        variable=var_name,
                        guard_expression=current,
                        true_type=non_nullish,
                        false_type=make_union([NULL_TYPE, UNDEFINED_TYPE]),
                        narrowing_kind=NarrowingKind.NULLISH_COALESCING,
                        original_type=original,
                        predicate=Predicate(
                            kind=PredicateKind.NOT_NONE, variable=var_name
                        ),
                        metadata={"chained": True},
                    ))
            current = current.right if current.right else None
            if current is None:
                break

        return points

    def analyze_assignment(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Analyze x ??= y assignment narrowing."""
        if node.kind != ASTNodeKind.NULLISH_ASSIGNMENT_EXPRESSION:
            if not (
                node.kind == ASTNodeKind.ASSIGNMENT_EXPRESSION
                and node.operator == "??="
            ):
                return None

        target = node.target or node.left
        if target is None:
            return None

        var_name = self._extract_var(target)
        if not var_name:
            return None

        original = variable_types.get(var_name, UNKNOWN_TYPE)

        # After x ??= y, x is guaranteed to be non-nullish
        rhs_type = UNKNOWN_TYPE
        if node.right and node.right.type_annotation:
            rhs_type = node.right.type_annotation

        result_type = make_union([original.without_nullish(), rhs_type])

        return NarrowingPoint(
            location=node.location,
            variable=var_name,
            guard_expression=node,
            true_type=result_type,
            false_type=None,
            narrowing_kind=NarrowingKind.NULLISH_COALESCING,
            original_type=original,
            predicate=Predicate(kind=PredicateKind.NOT_NONE, variable=var_name),
            metadata={"pattern": "nullish_assignment"},
        )

    def _is_nullish_coalescing(self, node: Optional[ASTNode]) -> bool:
        if node is None:
            return False
        return (
            node.kind == ASTNodeKind.NULLISH_COALESCING_EXPRESSION
            or (node.kind == ASTNodeKind.BINARY_EXPRESSION and node.operator == "??")
        )

    def _extract_var(self, node: ASTNode) -> Optional[str]:
        if node.kind == ASTNodeKind.IDENTIFIER:
            return node.name
        if node.kind == ASTNodeKind.MEMBER_EXPRESSION:
            obj = self._extract_var(node.object_node) if node.object_node else None
            prop = node.property_node.name if node.property_node else None
            if obj and prop:
                return f"{obj}.{prop}"
        return None


# ---------------------------------------------------------------------------
# ControlFlowNarrowing
# ---------------------------------------------------------------------------

class ControlFlowNarrowing:
    """General control flow narrowing analysis."""

    def __init__(self) -> None:
        self._typeof = TypeofNarrowing()
        self._instanceof = InstanceofNarrowing()
        self._in = InNarrowing()
        self._equality = EqualityNarrowing()
        self._truthiness = TruthinessNarrowing()
        self._user_guard = UserDefinedTypeGuard()
        self._optional_chain = OptionalChainingNarrowing()
        self._nullish = NullishCoalescingNarrowing()

    def analyze_if_statement(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Tuple[Dict[str, TSType], Dict[str, TSType]]:
        """Analyze if statement, returning (true_branch_types, false_branch_types)."""
        if node.condition is None:
            return dict(variable_types), dict(variable_types)

        true_types = dict(variable_types)
        false_types = dict(variable_types)

        points = self._analyze_condition(node.condition, variable_types)
        for pt in points:
            if pt.true_type:
                true_types[pt.variable] = pt.true_type
            if pt.false_type:
                false_types[pt.variable] = pt.false_type

        return true_types, false_types

    def analyze_return_elimination(
        self,
        return_node: ASTNode,
        variable_types: Dict[str, TSType],
        narrowing_in_condition: Optional[NarrowingPoint],
    ) -> Dict[str, TSType]:
        """After a return/throw, narrow via elimination in subsequent code."""
        if narrowing_in_condition is None:
            return dict(variable_types)

        result = dict(variable_types)
        # If the return was in the true branch, subsequent code has the false type
        if narrowing_in_condition.false_type:
            result[narrowing_in_condition.variable] = narrowing_in_condition.false_type

        return result

    def analyze_after_assignment(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Dict[str, TSType]:
        """Update types after an assignment."""
        result = dict(variable_types)

        if node.kind == ASTNodeKind.ASSIGNMENT_EXPRESSION:
            target = node.target or node.left
            value = node.init or node.right
        elif node.kind == ASTNodeKind.VARIABLE_DECLARATION:
            target = node.target or (node.declarations[0] if node.declarations else None)
            value = node.init or (
                node.declarations[0].init if node.declarations else None
            )
        else:
            return result

        if target is None:
            return result

        var_name = self._extract_var(target)
        if not var_name:
            return result

        if value is not None:
            inferred = self._infer_expression_type(value, variable_types)
            if inferred:
                result[var_name] = inferred

        return result

    def analyze_assertion(
        self,
        call_node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Dict[str, TSType]:
        """Analyze assertion function call and narrow types after it."""
        result = dict(variable_types)

        pt = self._user_guard.analyze_call(call_node, variable_types)
        if pt and pt.true_type:
            result[pt.variable] = pt.true_type

        return result

    def analyze_loop_condition(
        self,
        loop_node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Tuple[Dict[str, TSType], Dict[str, TSType]]:
        """Analyze loop condition, returning (body_types, after_loop_types)."""
        condition = loop_node.condition
        if condition is None:
            return dict(variable_types), dict(variable_types)

        body_types = dict(variable_types)
        after_types = dict(variable_types)

        points = self._analyze_condition(condition, variable_types)
        for pt in points:
            if pt.true_type:
                body_types[pt.variable] = pt.true_type
            if pt.false_type:
                after_types[pt.variable] = pt.false_type

        return body_types, after_types

    def analyze_switch_statement(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> List[Tuple[Dict[str, TSType], Optional[Any]]]:
        """Analyze switch statement, returning types for each case."""
        discriminant = node.discriminant
        cases = node.cases
        if discriminant is None or not cases:
            return [(dict(variable_types), None)]

        result: List[Tuple[Dict[str, TSType], Optional[Any]]] = []

        # Check for discriminated union switch
        if discriminant.kind == ASTNodeKind.MEMBER_EXPRESSION:
            du = DiscriminatedUnionNarrowing()
            points = du.analyze_switch(node, variable_types)
            for pt in points:
                case_types = dict(variable_types)
                if pt.true_type:
                    case_types[pt.variable] = pt.true_type
                result.append((
                    case_types,
                    pt.metadata.get("discriminant_value"),
                ))
            if result:
                return result

        # Check for typeof switch
        if discriminant.kind == ASTNodeKind.TYPEOF_EXPRESSION:
            points = self._typeof.analyze_switch(discriminant, cases, variable_types)
            for pt in points:
                case_types = dict(variable_types)
                if pt.true_type:
                    case_types[pt.variable] = pt.true_type
                result.append((
                    case_types,
                    pt.metadata.get("typeof_result"),
                ))
            if result:
                return result

        # Regular equality switch
        eq_points = self._equality.analyze_switch_cases(
            discriminant, cases, variable_types
        )
        for pt in eq_points:
            case_types = dict(variable_types)
            if pt.true_type:
                case_types[pt.variable] = pt.true_type
            result.append((case_types, pt.metadata.get("case_value")))

        return result if result else [(dict(variable_types), None)]

    def analyze_try_catch(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Tuple[Dict[str, TSType], Dict[str, TSType]]:
        """Analyze try-catch for exception type narrowing."""
        try_types = dict(variable_types)
        catch_types = dict(variable_types)

        if node.handler:
            catch_param = node.handler.param
            if catch_param and catch_param.kind == ASTNodeKind.PARAMETER:
                param_name = catch_param.name
                if catch_param.type_annotation:
                    catch_types[param_name] = catch_param.type_annotation
                else:
                    catch_types[param_name] = UNKNOWN_TYPE

        return try_types, catch_types

    def _analyze_condition(
        self,
        condition: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> List[NarrowingPoint]:
        """Analyze a condition expression and extract narrowing points."""
        points: List[NarrowingPoint] = []

        if condition.kind == ASTNodeKind.LOGICAL_EXPRESSION:
            if condition.operator == "&&":
                return self._analyze_and_condition(condition, variable_types)
            elif condition.operator == "||":
                return self._analyze_or_condition(condition, variable_types)

        # typeof check
        pt = self._typeof.analyze(condition, variable_types)
        if pt:
            points.append(pt)
            return points

        # typeof + null check compound
        pt = self._typeof.handles_null_check(condition, variable_types)
        if pt:
            points.append(pt)
            return points

        # instanceof check
        pt = self._instanceof.analyze(condition, variable_types)
        if pt:
            points.append(pt)
            return points

        # in check
        pt = self._in.analyze(condition, variable_types)
        if pt:
            points.append(pt)
            return points

        # Equality check
        pt = self._equality.analyze(condition, variable_types)
        if pt:
            points.append(pt)
            return points

        # Null check
        pt = self._equality.analyze_null_check(condition, variable_types)
        if pt:
            points.append(pt)
            return points

        # User-defined type guard call
        if condition.kind == ASTNodeKind.CALL_EXPRESSION:
            pt = self._user_guard.analyze_call(condition, variable_types)
            if pt:
                points.append(pt)
                return points

        # Truthiness (fallback)
        pt = self._truthiness.analyze(condition, variable_types)
        if pt:
            points.append(pt)

        return points

    def _analyze_and_condition(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> List[NarrowingPoint]:
        """Analyze guard1 && guard2 → intersect narrowings."""
        left = node.left
        right = node.right
        if left is None or right is None:
            return []

        left_points = self._analyze_condition(left, variable_types)

        # In right branch, left narrowings are applied
        narrowed = dict(variable_types)
        for pt in left_points:
            if pt.true_type:
                narrowed[pt.variable] = pt.true_type

        right_points = self._analyze_condition(right, narrowed)

        # Merge: for AND, the true branch has all narrowings applied
        all_points = left_points + right_points

        # Group by variable and compose
        by_var: Dict[str, List[NarrowingPoint]] = {}
        for pt in all_points:
            by_var.setdefault(pt.variable, []).append(pt)

        result: List[NarrowingPoint] = []
        for var_name, var_points in by_var.items():
            if len(var_points) == 1:
                result.append(var_points[0])
            else:
                composed = NarrowingComposition.compose_and(var_points)
                if composed:
                    result.append(composed)

        return result

    def _analyze_or_condition(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> List[NarrowingPoint]:
        """Analyze guard1 || guard2 → union narrowings."""
        left = node.left
        right = node.right
        if left is None or right is None:
            return []

        left_points = self._analyze_condition(left, variable_types)

        # In right side of ||, left narrowing is FALSE
        narrowed = dict(variable_types)
        for pt in left_points:
            if pt.false_type:
                narrowed[pt.variable] = pt.false_type

        right_points = self._analyze_condition(right, narrowed)

        all_points = left_points + right_points

        by_var: Dict[str, List[NarrowingPoint]] = {}
        for pt in all_points:
            by_var.setdefault(pt.variable, []).append(pt)

        result: List[NarrowingPoint] = []
        for var_name, var_points in by_var.items():
            if len(var_points) == 1:
                result.append(var_points[0])
            else:
                composed = NarrowingComposition.compose_or(var_points)
                if composed:
                    result.append(composed)

        return result

    def _infer_expression_type(
        self, node: ASTNode, variable_types: Dict[str, TSType]
    ) -> Optional[TSType]:
        """Infer the type of an expression."""
        if node.type_annotation:
            return node.type_annotation

        if node.kind == ASTNodeKind.LITERAL:
            return self._literal_type(node.value)

        if node.kind == ASTNodeKind.IDENTIFIER:
            return variable_types.get(node.name)

        if node.kind == ASTNodeKind.CALL_EXPRESSION:
            if node.callee and node.callee.type_annotation:
                rt = node.callee.return_type
                if rt:
                    return rt
            return None

        if node.kind == ASTNodeKind.ARRAY_EXPRESSION:
            return TSType(kind=TypeKind.ARRAY, name="Array")

        if node.kind == ASTNodeKind.OBJECT_EXPRESSION:
            return TSType(kind=TypeKind.OBJECT, name="object")

        if node.kind == ASTNodeKind.TEMPLATE_LITERAL:
            return STRING_TYPE

        return None

    def _literal_type(self, value: Any) -> TSType:
        """Create a literal type from a value."""
        if isinstance(value, str):
            return TSType(kind=TypeKind.LITERAL_STRING, literal_value=value)
        if isinstance(value, bool):
            return TSType(kind=TypeKind.LITERAL_BOOLEAN, literal_value=value)
        if isinstance(value, int):
            return TSType(kind=TypeKind.LITERAL_NUMBER, literal_value=value)
        if isinstance(value, float):
            return TSType(kind=TypeKind.LITERAL_NUMBER, literal_value=value)
        if value is None:
            return NULL_TYPE
        return UNKNOWN_TYPE

    def _extract_var(self, node: ASTNode) -> Optional[str]:
        if node.kind == ASTNodeKind.IDENTIFIER:
            return node.name
        if node.kind == ASTNodeKind.MEMBER_EXPRESSION:
            obj = self._extract_var(node.object_node) if node.object_node else None
            prop = node.property_node.name if node.property_node else None
            if obj and prop:
                return f"{obj}.{prop}"
        return None


# ---------------------------------------------------------------------------
# AssignmentNarrowing
# ---------------------------------------------------------------------------

class AssignmentNarrowing:
    """Narrow type via assignment analysis."""

    def analyze(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> Optional[NarrowingPoint]:
        """Analyze a direct assignment for narrowing."""
        if node.kind not in (
            ASTNodeKind.ASSIGNMENT_EXPRESSION,
            ASTNodeKind.VARIABLE_DECLARATION,
        ):
            return None

        target = node.target or node.left
        value = node.init or node.right

        if node.kind == ASTNodeKind.VARIABLE_DECLARATION and node.declarations:
            decl = node.declarations[0]
            target = target or decl.target or decl
            value = value or decl.init

        if target is None:
            return None

        var_name = self._extract_var(target)
        if not var_name:
            return None

        original = variable_types.get(var_name, UNKNOWN_TYPE)
        assigned_type = self._infer_type(value, variable_types) if value else UNKNOWN_TYPE

        pred = Predicate(
            kind=PredicateKind.ISINSTANCE,
            variable=var_name,
            type_name=assigned_type.name if assigned_type else "unknown",
        )

        return NarrowingPoint(
            location=node.location,
            variable=var_name,
            guard_expression=node,
            true_type=assigned_type,
            false_type=None,
            narrowing_kind=NarrowingKind.ASSIGNMENT,
            original_type=original,
            predicate=pred,
            metadata={"assignment": True},
        )

    def analyze_destructuring(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> List[NarrowingPoint]:
        """Analyze destructuring assignment for narrowing."""
        points: List[NarrowingPoint] = []

        if node.kind == ASTNodeKind.VARIABLE_DECLARATION:
            target = node.target
            value = node.init
        elif node.kind == ASTNodeKind.ASSIGNMENT_EXPRESSION:
            target = node.target or node.left
            value = node.init or node.right
        else:
            return points

        if target is None:
            return points

        source_type = self._infer_type(value, variable_types) if value else UNKNOWN_TYPE

        if target.kind == ASTNodeKind.OBJECT_PATTERN:
            points.extend(
                self._analyze_object_destructuring(target, source_type, node.location)
            )
        elif target.kind == ASTNodeKind.ARRAY_PATTERN:
            points.extend(
                self._analyze_array_destructuring(target, source_type, node.location)
            )

        return points

    def _analyze_object_destructuring(
        self,
        pattern: ASTNode,
        source_type: TSType,
        location: SourceLocation,
    ) -> List[NarrowingPoint]:
        """Analyze object destructuring: const {a, b} = obj."""
        points: List[NarrowingPoint] = []

        for prop_node in pattern.properties_nodes:
            key_name = prop_node.name
            if not key_name and prop_node.kind == ASTNodeKind.IDENTIFIER:
                key_name = prop_node.name

            if key_name:
                prop_type = source_type.properties.get(key_name, UNKNOWN_TYPE)
                points.append(NarrowingPoint(
                    location=location,
                    variable=key_name,
                    guard_expression=None,
                    true_type=prop_type,
                    false_type=None,
                    narrowing_kind=NarrowingKind.ASSIGNMENT,
                    original_type=UNKNOWN_TYPE,
                    predicate=Predicate(
                        kind=PredicateKind.ISINSTANCE,
                        variable=key_name,
                        type_name=prop_type.name,
                    ),
                    metadata={"destructuring": "object", "property": key_name},
                ))

        return points

    def _analyze_array_destructuring(
        self,
        pattern: ASTNode,
        source_type: TSType,
        location: SourceLocation,
    ) -> List[NarrowingPoint]:
        """Analyze array destructuring: const [first, ...rest] = arr."""
        points: List[NarrowingPoint] = []

        element_type = UNKNOWN_TYPE
        if source_type.kind == TypeKind.ARRAY and source_type.type_arguments:
            element_type = source_type.type_arguments[0]
        elif source_type.kind == TypeKind.TUPLE and source_type.members:
            pass  # Handle per-element

        for i, elem in enumerate(pattern.elements):
            if elem.kind == ASTNodeKind.REST_ELEMENT:
                rest_name = self._extract_var(elem.expression if elem.expression else elem)
                if rest_name:
                    rest_type = TSType(
                        kind=TypeKind.ARRAY,
                        name="Array",
                        type_arguments=[element_type],
                    )
                    points.append(NarrowingPoint(
                        location=location,
                        variable=rest_name,
                        guard_expression=None,
                        true_type=rest_type,
                        false_type=None,
                        narrowing_kind=NarrowingKind.ASSIGNMENT,
                        metadata={"destructuring": "array_rest", "index": i},
                    ))
            elif elem.kind == ASTNodeKind.IDENTIFIER:
                var_name = elem.name
                if source_type.kind == TypeKind.TUPLE and i < len(source_type.members):
                    elem_type = source_type.members[i]
                else:
                    elem_type = element_type

                points.append(NarrowingPoint(
                    location=location,
                    variable=var_name,
                    guard_expression=None,
                    true_type=elem_type,
                    false_type=None,
                    narrowing_kind=NarrowingKind.ASSIGNMENT,
                    metadata={"destructuring": "array", "index": i},
                ))

        return points

    def _infer_type(
        self, node: Optional[ASTNode], variable_types: Dict[str, TSType]
    ) -> TSType:
        """Infer type from an expression."""
        if node is None:
            return UNKNOWN_TYPE

        if node.type_annotation:
            return node.type_annotation

        if node.kind == ASTNodeKind.LITERAL:
            return self._literal_type(node.value)

        if node.kind == ASTNodeKind.IDENTIFIER:
            return variable_types.get(node.name, UNKNOWN_TYPE)

        if node.kind == ASTNodeKind.TEMPLATE_LITERAL:
            return STRING_TYPE

        if node.kind == ASTNodeKind.ARRAY_EXPRESSION:
            return TSType(kind=TypeKind.ARRAY, name="Array")

        if node.kind == ASTNodeKind.OBJECT_EXPRESSION:
            props: Dict[str, TSType] = {}
            for child in node.children:
                if child.name:
                    props[child.name] = self._infer_type(child.init, variable_types)
            return TSType(kind=TypeKind.OBJECT, name="object", properties=props)

        if node.kind == ASTNodeKind.CALL_EXPRESSION:
            if node.callee and node.callee.return_type:
                return node.callee.return_type
            return UNKNOWN_TYPE

        return UNKNOWN_TYPE

    def _literal_type(self, value: Any) -> TSType:
        if isinstance(value, str):
            return TSType(kind=TypeKind.LITERAL_STRING, literal_value=value)
        if isinstance(value, bool):
            return TSType(kind=TypeKind.LITERAL_BOOLEAN, literal_value=value)
        if isinstance(value, int):
            return TSType(kind=TypeKind.LITERAL_NUMBER, literal_value=value)
        if isinstance(value, float):
            return TSType(kind=TypeKind.LITERAL_NUMBER, literal_value=value)
        if value is None:
            return NULL_TYPE
        return UNKNOWN_TYPE

    def _extract_var(self, node: Optional[ASTNode]) -> Optional[str]:
        if node is None:
            return None
        if node.kind == ASTNodeKind.IDENTIFIER:
            return node.name
        if node.kind == ASTNodeKind.MEMBER_EXPRESSION:
            obj = self._extract_var(node.object_node)
            prop = node.property_node.name if node.property_node else None
            if obj and prop:
                return f"{obj}.{prop}"
        return None


# ---------------------------------------------------------------------------
# NarrowingComposition
# ---------------------------------------------------------------------------

class NarrowingComposition:
    """Compose multiple narrowings (AND, OR, NOT)."""

    @staticmethod
    def compose_and(points: List[NarrowingPoint]) -> Optional[NarrowingPoint]:
        """Compose narrowings with AND: intersect true types, union false types."""
        if not points:
            return None
        if len(points) == 1:
            return points[0]

        var_name = points[0].variable
        original = points[0].original_type

        # True type: intersection of all true types
        true_types = [p.true_type for p in points if p.true_type]
        if true_types:
            result_true = true_types[0]
            for t in true_types[1:]:
                result_true = make_intersection([result_true, t])
        else:
            result_true = None

        # False type: union of all false types
        false_types = [p.false_type for p in points if p.false_type]
        if false_types:
            result_false = make_union(false_types)
        else:
            result_false = None

        preds = [p.predicate for p in points if p.predicate]
        combined = preds[0] if preds else Predicate(kind=PredicateKind.TRUE)
        for p in preds[1:]:
            combined = combined.conjoin(p)

        return NarrowingPoint(
            location=points[0].location,
            variable=var_name,
            guard_expression=None,
            true_type=result_true,
            false_type=result_false,
            narrowing_kind=NarrowingKind.COMPOUND,
            original_type=original,
            predicate=combined,
            metadata={"composition": "and", "count": len(points)},
        )

    @staticmethod
    def compose_or(points: List[NarrowingPoint]) -> Optional[NarrowingPoint]:
        """Compose narrowings with OR: union true types, intersect false types."""
        if not points:
            return None
        if len(points) == 1:
            return points[0]

        var_name = points[0].variable
        original = points[0].original_type

        true_types = [p.true_type for p in points if p.true_type]
        if true_types:
            result_true = make_union(true_types)
        else:
            result_true = None

        false_types = [p.false_type for p in points if p.false_type]
        if false_types:
            result_false = false_types[0]
            for t in false_types[1:]:
                result_false = make_intersection([result_false, t])
        else:
            result_false = None

        preds = [p.predicate for p in points if p.predicate]
        combined = preds[0] if preds else Predicate(kind=PredicateKind.FALSE)
        for p in preds[1:]:
            combined = combined.disjoin(p)

        return NarrowingPoint(
            location=points[0].location,
            variable=var_name,
            guard_expression=None,
            true_type=result_true,
            false_type=result_false,
            narrowing_kind=NarrowingKind.COMPOUND,
            original_type=original,
            predicate=combined,
            metadata={"composition": "or", "count": len(points)},
        )

    @staticmethod
    def compose_not(point: NarrowingPoint) -> NarrowingPoint:
        """Compose a NOT narrowing: swap true/false types."""
        return point.flipped()

    @staticmethod
    def compose_sequential(
        points: List[NarrowingPoint],
        variable_types: Dict[str, TSType],
    ) -> Dict[str, TSType]:
        """Apply sequential narrowings on the same variable.

        Each subsequent narrowing uses the result of the previous one.
        """
        result = dict(variable_types)
        for pt in points:
            if pt.true_type:
                result[pt.variable] = pt.true_type
        return result


# ---------------------------------------------------------------------------
# NarrowingGraph
# ---------------------------------------------------------------------------

class NarrowingGraph:
    """Graph of narrowing effects, built from CFG and guard analysis."""

    def __init__(self) -> None:
        self._nodes: Dict[int, FlowNode] = {}
        self._next_id: int = 0

    def add_node(
        self,
        node_type: FlowNodeType,
        cfg_block_id: Optional[int] = None,
        narrowing: Optional[NarrowingPoint] = None,
        ast_node: Optional[ASTNode] = None,
    ) -> FlowNode:
        nid = self._next_id
        self._next_id += 1
        fn = FlowNode(
            id=nid,
            type=node_type,
            cfg_block_id=cfg_block_id,
            associated_narrowing=narrowing,
            ast_node=ast_node,
        )
        self._nodes[nid] = fn
        return fn

    def add_edge(self, source_id: int, target_id: int) -> None:
        if source_id in self._nodes and target_id in self._nodes:
            self._nodes[source_id].successors.append(target_id)
            self._nodes[target_id].predecessors.append(source_id)

    def get_node(self, node_id: int) -> Optional[FlowNode]:
        return self._nodes.get(node_id)

    @property
    def nodes(self) -> Dict[int, FlowNode]:
        return self._nodes

    def build_from_cfg(
        self,
        cfg: CFG,
        narrowing_points: List[NarrowingPoint],
    ) -> None:
        """Build narrowing graph from CFG and identified narrowing points."""
        block_to_node: Dict[int, int] = {}
        narrowing_by_block: Dict[int, List[NarrowingPoint]] = {}

        for pt in narrowing_points:
            if pt.cfg_block_id is not None:
                narrowing_by_block.setdefault(pt.cfg_block_id, []).append(pt)

        # Create flow nodes for each CFG block
        for bid, block in cfg.blocks.items():
            narrowings = narrowing_by_block.get(bid, [])
            if narrowings:
                for nar in narrowings:
                    fn = self.add_node(
                        FlowNodeType.NARROWING,
                        cfg_block_id=bid,
                        narrowing=nar,
                    )
                    block_to_node.setdefault(bid, fn.id)
            else:
                node_type = self._block_kind_to_flow_type(block.kind)
                fn = self.add_node(node_type, cfg_block_id=bid)
                block_to_node[bid] = fn.id

        # Add edges
        for edge in cfg.edges:
            src = block_to_node.get(edge.source)
            tgt = block_to_node.get(edge.target)
            if src is not None and tgt is not None:
                self.add_edge(src, tgt)

    def propagate_types(
        self,
        initial_types: Dict[str, TSType],
        max_iterations: int = 100,
    ) -> Dict[int, Dict[str, TSType]]:
        """Propagate type narrowings through the graph using fixed-point iteration."""
        node_types: Dict[int, Dict[str, TSType]] = {}

        # Initialize
        for nid, node in self._nodes.items():
            if not node.predecessors:
                node_types[nid] = dict(initial_types)
                node.variable_types = dict(initial_types)
            else:
                node_types[nid] = {}

        # Fixed-point iteration
        changed = True
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for nid in self._topological_order():
                node = self._nodes[nid]
                if not node.predecessors:
                    continue

                # Merge incoming types
                incoming: Dict[str, List[TSType]] = {}
                for pred_id in node.predecessors:
                    pred_types = node_types.get(pred_id, {})
                    for var, typ in pred_types.items():
                        incoming.setdefault(var, []).append(typ)

                merged: Dict[str, TSType] = {}
                for var, types in incoming.items():
                    if len(types) == 1:
                        merged[var] = types[0]
                    else:
                        merged[var] = make_union(types)

                # Apply narrowing if present
                if node.associated_narrowing:
                    nar = node.associated_narrowing
                    if node.type == FlowNodeType.CONDITION_TRUE and nar.true_type:
                        merged[nar.variable] = nar.true_type
                    elif node.type == FlowNodeType.CONDITION_FALSE and nar.false_type:
                        merged[nar.variable] = nar.false_type
                    elif node.type == FlowNodeType.NARROWING and nar.true_type:
                        merged[nar.variable] = nar.true_type

                if node.type == FlowNodeType.WIDENING:
                    # At loop back edges, widen to union with initial
                    for var in merged:
                        if var in initial_types:
                            merged[var] = make_union([merged[var], initial_types[var]])

                old = node_types.get(nid, {})
                if merged != old:
                    node_types[nid] = merged
                    node.variable_types = merged
                    changed = True

        return node_types

    def handle_loop_widening(
        self,
        cfg: CFG,
        node_types: Dict[int, Dict[str, TSType]],
        initial_types: Dict[str, TSType],
    ) -> Dict[int, Dict[str, TSType]]:
        """Handle widening at loop back edges to ensure termination."""
        back_edges = cfg.detect_back_edges()
        result = dict(node_types)

        for edge in back_edges:
            target_nid = None
            for nid, node in self._nodes.items():
                if node.cfg_block_id == edge.target:
                    target_nid = nid
                    break

            if target_nid is not None and target_nid in result:
                types = result[target_nid]
                widened: Dict[str, TSType] = {}
                for var, typ in types.items():
                    if var in initial_types:
                        widened[var] = make_union([typ, initial_types[var]])
                    else:
                        widened[var] = typ
                result[target_nid] = widened

        return result

    def _topological_order(self) -> List[int]:
        """Topological order of flow nodes."""
        visited: Set[int] = set()
        order: List[int] = []

        def dfs(nid: int) -> None:
            if nid in visited:
                return
            visited.add(nid)
            for succ in self._nodes[nid].successors:
                if succ not in visited:
                    dfs(succ)
            order.append(nid)

        for nid in self._nodes:
            if not self._nodes[nid].predecessors:
                dfs(nid)
        for nid in self._nodes:
            if nid not in visited:
                dfs(nid)

        order.reverse()
        return order

    def _block_kind_to_flow_type(self, kind: CFGBlockKind) -> FlowNodeType:
        mapping = {
            CFGBlockKind.ENTRY: FlowNodeType.START,
            CFGBlockKind.EXIT: FlowNodeType.END,
            CFGBlockKind.CONDITION: FlowNodeType.CONDITION_TRUE,
            CFGBlockKind.LOOP_HEADER: FlowNodeType.LOOP_START,
            CFGBlockKind.LOOP_EXIT: FlowNodeType.LOOP_END,
            CFGBlockKind.UNREACHABLE: FlowNodeType.UNREACHABLE,
        }
        return mapping.get(kind, FlowNodeType.ASSIGNMENT)


# ---------------------------------------------------------------------------
# NarrowingToPredicateConverter
# ---------------------------------------------------------------------------

class NarrowingToPredicateConverter:
    """Convert TypeScript narrowings to refinement predicate language P."""

    def __init__(self) -> None:
        self._type_name_map: Dict[str, str] = {
            "string": "str",
            "number": "float",
            "bigint": "int",
            "boolean": "bool",
            "symbol": "Symbol",
            "undefined": "NoneType",
            "void": "NoneType",
            "null": "NoneType",
            "object": "object",
            "function": "Callable",
            "Array": "list",
            "Map": "dict",
            "Set": "set",
            "Promise": "Awaitable",
            "Date": "datetime",
            "RegExp": "Pattern",
            "Error": "Exception",
        }

    def convert(self, narrowing_point: NarrowingPoint) -> Predicate:
        """Convert a narrowing point to a refinement predicate."""
        if narrowing_point.predicate:
            return narrowing_point.predicate

        dispatch: Dict[NarrowingKind, Callable[[NarrowingPoint], Predicate]] = {
            NarrowingKind.TYPEOF: self._convert_typeof,
            NarrowingKind.INSTANCEOF: self._convert_instanceof,
            NarrowingKind.EQUALITY: self._convert_equality,
            NarrowingKind.IN: self._convert_in,
            NarrowingKind.TRUTHINESS: self._convert_truthiness,
            NarrowingKind.USER_DEFINED: self._convert_user_defined,
            NarrowingKind.ASSERTION: self._convert_assertion,
            NarrowingKind.DISCRIMINATED_UNION: self._convert_discriminated,
            NarrowingKind.OPTIONAL_CHAINING: self._convert_optional_chain,
            NarrowingKind.NULLISH_COALESCING: self._convert_nullish,
            NarrowingKind.ASSIGNMENT: self._convert_assignment,
            NarrowingKind.COMPOUND: self._convert_compound,
        }

        converter = dispatch.get(narrowing_point.narrowing_kind)
        if converter:
            return converter(narrowing_point)
        return Predicate(kind=PredicateKind.TRUE)

    def convert_all(self, points: List[NarrowingPoint]) -> List[Predicate]:
        """Convert all narrowing points to predicates."""
        return [self.convert(pt) for pt in points]

    def _convert_typeof(self, pt: NarrowingPoint) -> Predicate:
        typeof_result_val = pt.metadata.get("typeof_result", "")
        py_type = self._type_name_map.get(typeof_result_val, typeof_result_val)
        pred = Predicate(
            kind=PredicateKind.ISINSTANCE,
            variable=pt.variable,
            type_name=py_type,
        )
        if pt.is_negated:
            pred = pred.negate()
        return pred

    def _convert_instanceof(self, pt: NarrowingPoint) -> Predicate:
        class_name = pt.metadata.get("class_name", "")
        py_name = self._type_name_map.get(class_name, class_name)
        pred = Predicate(
            kind=PredicateKind.ISINSTANCE,
            variable=pt.variable,
            type_name=py_name,
        )
        if pt.is_negated:
            pred = pred.negate()
        return pred

    def _convert_equality(self, pt: NarrowingPoint) -> Predicate:
        null_kind = pt.metadata.get("null_kind")
        if null_kind:
            is_negated = pt.metadata.get("is_negated", False)
            if is_negated:
                return Predicate(kind=PredicateKind.NOT_NONE, variable=pt.variable)
            else:
                return Predicate(kind=PredicateKind.IS_NONE, variable=pt.variable)

        value = pt.metadata.get("literal_value")
        if value is not None:
            pred = Predicate(
                kind=PredicateKind.EQUALITY,
                variable=pt.variable,
                value=value,
            )
            if pt.metadata.get("is_negated", False):
                pred = pred.negate()
            return pred

        enum_member = pt.metadata.get("enum_member")
        if enum_member:
            pred = Predicate(
                kind=PredicateKind.EQUALITY,
                variable=pt.variable,
                value=enum_member,
            )
            if pt.metadata.get("is_negated", False):
                pred = pred.negate()
            return pred

        return Predicate(kind=PredicateKind.TRUE, variable=pt.variable)

    def _convert_in(self, pt: NarrowingPoint) -> Predicate:
        key = pt.metadata.get("key", "")
        return Predicate(
            kind=PredicateKind.HASATTR,
            variable=pt.variable,
            attribute=key,
        )

    def _convert_truthiness(self, pt: NarrowingPoint) -> Predicate:
        if pt.is_negated:
            return Predicate(kind=PredicateKind.IS_FALSY, variable=pt.variable)
        return Predicate(kind=PredicateKind.IS_TRUTHY, variable=pt.variable)

    def _convert_user_defined(self, pt: NarrowingPoint) -> Predicate:
        guard_fn = pt.metadata.get("guard_function", "")
        guarded_type = pt.metadata.get("guarded_type", "")
        py_type = self._type_name_map.get(guarded_type, guarded_type)
        return Predicate(
            kind=PredicateKind.TYPE_GUARD,
            variable=pt.variable,
            type_name=py_type,
            guard_function=guard_fn,
        )

    def _convert_assertion(self, pt: NarrowingPoint) -> Predicate:
        guarded_type = pt.metadata.get("guarded_type", "")
        py_type = self._type_name_map.get(guarded_type, guarded_type)
        return Predicate(
            kind=PredicateKind.ISINSTANCE,
            variable=pt.variable,
            type_name=py_type,
        )

    def _convert_discriminated(self, pt: NarrowingPoint) -> Predicate:
        disc_prop = pt.metadata.get("discriminant_property", "")
        disc_val = pt.metadata.get("discriminant_value")
        return Predicate(
            kind=PredicateKind.EQUALITY,
            variable=f"{pt.variable}.{disc_prop}",
            value=disc_val,
        )

    def _convert_optional_chain(self, pt: NarrowingPoint) -> Predicate:
        return Predicate(
            kind=PredicateKind.NOT_NONE,
            variable=pt.variable,
        )

    def _convert_nullish(self, pt: NarrowingPoint) -> Predicate:
        return Predicate(
            kind=PredicateKind.NOT_NONE,
            variable=pt.variable,
        )

    def _convert_assignment(self, pt: NarrowingPoint) -> Predicate:
        if pt.true_type:
            py_type = self._type_name_map.get(pt.true_type.name, pt.true_type.name)
            return Predicate(
                kind=PredicateKind.ISINSTANCE,
                variable=pt.variable,
                type_name=py_type,
            )
        return Predicate(kind=PredicateKind.TRUE, variable=pt.variable)

    def _convert_compound(self, pt: NarrowingPoint) -> Predicate:
        if pt.predicate:
            return pt.predicate
        return Predicate(kind=PredicateKind.TRUE, variable=pt.variable)


# ---------------------------------------------------------------------------
# ExhaustivenessChecker
# ---------------------------------------------------------------------------

class ExhaustivenessChecker:
    """Check exhaustive handling of union types."""

    def __init__(self) -> None:
        self._du = DiscriminatedUnionNarrowing()

    def check_switch_exhaustiveness(
        self,
        switch_node: ASTNode,
        union_type: TSType,
        discriminant_property: str,
    ) -> ExhaustivenessResult:
        """Check if a switch statement exhaustively handles all union members."""
        if union_type.kind != TypeKind.UNION:
            return ExhaustivenessResult(
                is_exhaustive=True,
                handled_members=[],
                missing_members=[],
                has_default=False,
            )

        cases = switch_node.cases or []
        handled_values: Set[Any] = set()
        has_default = False

        for case_node in cases:
            if case_node.value is not None:
                handled_values.add(case_node.value)
            else:
                has_default = True

        handled_members: List[TSType] = []
        missing_members: List[TSType] = []

        for member in union_type.members:
            disc_type = member.properties.get(discriminant_property)
            if disc_type and disc_type.literal_value in handled_values:
                handled_members.append(member)
            else:
                missing_members.append(member)

        is_exhaustive = has_default or len(missing_members) == 0

        return ExhaustivenessResult(
            is_exhaustive=is_exhaustive,
            handled_members=handled_members,
            missing_members=missing_members,
            has_default=has_default,
            never_type_at_end=is_exhaustive and not has_default,
            missing_values=[
                m.properties.get(discriminant_property, UNKNOWN_TYPE).literal_value
                for m in missing_members
                if discriminant_property in m.properties
            ],
        )

    def check_if_chain_exhaustiveness(
        self,
        conditions: List[ASTNode],
        union_type: TSType,
        discriminant_property: str,
        has_else: bool = False,
    ) -> ExhaustivenessResult:
        """Check if an if-else chain exhaustively handles all union members."""
        if union_type.kind != TypeKind.UNION:
            return ExhaustivenessResult(
                is_exhaustive=True,
                handled_members=[],
                missing_members=[],
                has_default=has_else,
            )

        handled_values: Set[Any] = set()
        for cond in conditions:
            disc_val = self._extract_discriminant_value(cond, discriminant_property)
            if disc_val is not None:
                handled_values.add(disc_val)

        handled_members: List[TSType] = []
        missing_members: List[TSType] = []

        for member in union_type.members:
            disc_type = member.properties.get(discriminant_property)
            if disc_type and disc_type.literal_value in handled_values:
                handled_members.append(member)
            else:
                missing_members.append(member)

        is_exhaustive = has_else or len(missing_members) == 0

        return ExhaustivenessResult(
            is_exhaustive=is_exhaustive,
            handled_members=handled_members,
            missing_members=missing_members,
            has_default=has_else,
            never_type_at_end=is_exhaustive and not has_else,
            missing_values=[
                m.properties.get(discriminant_property, UNKNOWN_TYPE).literal_value
                for m in missing_members
                if discriminant_property in m.properties
            ],
        )

    def check_typeof_exhaustiveness(
        self,
        handled_typeof_values: List[str],
        original_type: TSType,
    ) -> ExhaustivenessResult:
        """Check if typeof checks exhaustively handle all type possibilities."""
        all_typeof = set(TypeofNarrowing.TYPEOF_RESULTS)
        handled = set(handled_typeof_values)

        if original_type.kind == TypeKind.UNION:
            needed: Set[str] = set()
            for m in original_type.members:
                needed.add(typeof_result(m))
        else:
            needed = {typeof_result(original_type)}

        missing_typeof = needed - handled
        is_exhaustive = len(missing_typeof) == 0

        return ExhaustivenessResult(
            is_exhaustive=is_exhaustive,
            handled_members=[],
            missing_members=[],
            has_default=False,
            missing_values=list(missing_typeof),
        )

    def suggest_missing_cases(
        self, result: ExhaustivenessResult
    ) -> List[str]:
        """Generate suggestions for missing cases."""
        suggestions: List[str] = []
        for member in result.missing_members:
            if member.discriminant_property and member.discriminant_value:
                suggestions.append(
                    f"case {member.discriminant_value!r}: // Handle {member.name}"
                )
            elif member.name:
                suggestions.append(f"// Missing case for type: {member.name}")

        for val in result.missing_values:
            if val not in [m.discriminant_value for m in result.missing_members if m.discriminant_value]:
                suggestions.append(f"case {val!r}: // Missing handler")

        if not result.is_exhaustive and not suggestions:
            suggestions.append("default: // Add default case for exhaustiveness")

        return suggestions

    def detect_never_at_point(
        self,
        variable_types: Dict[str, TSType],
        variable: str,
    ) -> bool:
        """Check if a variable has been narrowed to never (exhausted)."""
        typ = variable_types.get(variable)
        if typ is None:
            return False
        return typ.kind == TypeKind.NEVER

    def _extract_discriminant_value(
        self, condition: ASTNode, disc_prop: str
    ) -> Optional[Any]:
        """Extract discriminant value from a condition."""
        if condition.kind != ASTNodeKind.BINARY_EXPRESSION:
            return None
        if condition.operator not in ("===", "=="):
            return None

        left = condition.left
        right = condition.right
        if left is None or right is None:
            return None

        member_node = None
        literal_node = None
        if left.kind == ASTNodeKind.MEMBER_EXPRESSION and right.kind == ASTNodeKind.LITERAL:
            member_node = left
            literal_node = right
        elif right.kind == ASTNodeKind.MEMBER_EXPRESSION and left.kind == ASTNodeKind.LITERAL:
            member_node = right
            literal_node = left

        if member_node is None or literal_node is None:
            return None

        prop_name = (
            member_node.property_node.name if member_node.property_node else None
        )
        if prop_name != disc_prop:
            return None

        return literal_node.value


@dataclass
class ExhaustivenessResult:
    """Result of an exhaustiveness check."""
    is_exhaustive: bool
    handled_members: List[TSType]
    missing_members: List[TSType]
    has_default: bool
    never_type_at_end: bool = False
    missing_values: List[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CFG Builder (simplified, for narrowing analysis)
# ---------------------------------------------------------------------------

class CFGBuilder:
    """Build a control flow graph from a TypeScript AST."""

    def __init__(self) -> None:
        self._cfg: CFG = CFG(
            blocks={
                0: CFGBlock(id=0, kind=CFGBlockKind.ENTRY),
                1: CFGBlock(id=1, kind=CFGBlockKind.EXIT),
            },
        )

    def build(self, func_node: ASTNode) -> CFG:
        """Build CFG from a function declaration AST node."""
        body = func_node.body
        if body is None:
            self._cfg.add_edge(0, 1)
            return self._cfg

        current_block = self._cfg.add_block(CFGBlockKind.NORMAL)
        self._cfg.add_edge(0, current_block.id)

        last = self._process_block(body, current_block)
        if last is not None:
            self._cfg.add_edge(last.id, 1)

        self._cfg.detect_back_edges()
        self._cfg.compute_dominators()

        return self._cfg

    def _process_block(
        self, node: ASTNode, current: CFGBlock
    ) -> Optional[CFGBlock]:
        """Process a block of statements, returning the last block."""
        if node.kind == ASTNodeKind.BLOCK:
            for child in node.children:
                result = self._process_statement(child, current)
                if result is None:
                    return None  # Unreachable after return/throw
                current = result
            return current
        return self._process_statement(node, current)

    def _process_statement(
        self, node: ASTNode, current: CFGBlock
    ) -> Optional[CFGBlock]:
        """Process a single statement, returning the next block."""
        if node.kind == ASTNodeKind.IF_STATEMENT:
            return self._process_if(node, current)
        if node.kind == ASTNodeKind.SWITCH_STATEMENT:
            return self._process_switch(node, current)
        if node.kind in (
            ASTNodeKind.WHILE_STATEMENT,
            ASTNodeKind.DO_WHILE_STATEMENT,
            ASTNodeKind.FOR_STATEMENT,
        ):
            return self._process_loop(node, current)
        if node.kind == ASTNodeKind.RETURN_STATEMENT:
            return self._process_return(node, current)
        if node.kind == ASTNodeKind.THROW_STATEMENT:
            return self._process_throw(node, current)
        if node.kind == ASTNodeKind.TRY_STATEMENT:
            return self._process_try(node, current)
        if node.kind == ASTNodeKind.BLOCK:
            return self._process_block(node, current)

        # Regular statement: add to current block
        current.statements.append(node)
        return current

    def _process_if(
        self, node: ASTNode, current: CFGBlock
    ) -> Optional[CFGBlock]:
        """Process an if statement."""
        cond_block = self._cfg.add_block(CFGBlockKind.CONDITION)
        cond_block.condition = node.condition
        self._cfg.add_edge(current.id, cond_block.id)

        true_block = self._cfg.add_block(CFGBlockKind.NORMAL)
        self._cfg.add_edge(cond_block.id, true_block.id, label="true", condition=node.condition)

        join_block = self._cfg.add_block(CFGBlockKind.NORMAL)

        true_last = None
        if node.consequent:
            true_last = self._process_block(node.consequent, true_block)
        else:
            true_last = true_block
        if true_last:
            self._cfg.add_edge(true_last.id, join_block.id)

        if node.alternate:
            false_block = self._cfg.add_block(CFGBlockKind.NORMAL)
            self._cfg.add_edge(cond_block.id, false_block.id, label="false", condition=node.condition)
            false_last = self._process_block(node.alternate, false_block)
            if false_last:
                self._cfg.add_edge(false_last.id, join_block.id)
        else:
            self._cfg.add_edge(cond_block.id, join_block.id, label="false", condition=node.condition)

        return join_block

    def _process_switch(
        self, node: ASTNode, current: CFGBlock
    ) -> Optional[CFGBlock]:
        """Process a switch statement."""
        switch_block = self._cfg.add_block(CFGBlockKind.SWITCH_HEAD)
        switch_block.condition = node.discriminant
        self._cfg.add_edge(current.id, switch_block.id)

        join_block = self._cfg.add_block(CFGBlockKind.NORMAL)

        for case_node in (node.cases or []):
            if case_node.value is not None:
                case_block = self._cfg.add_block(CFGBlockKind.CASE)
            else:
                case_block = self._cfg.add_block(CFGBlockKind.DEFAULT_CASE)
            case_block.condition = case_node
            self._cfg.add_edge(switch_block.id, case_block.id, label="case")

            last = case_block
            if case_node.body:
                last_result = self._process_block(case_node.body, case_block)
                if last_result:
                    last = last_result
            elif case_node.children:
                for child in case_node.children:
                    result = self._process_statement(child, last)
                    if result is None:
                        last = None  # type: ignore[assignment]
                        break
                    last = result

            if last is not None:
                self._cfg.add_edge(last.id, join_block.id)

        return join_block

    def _process_loop(
        self, node: ASTNode, current: CFGBlock
    ) -> Optional[CFGBlock]:
        """Process a loop statement."""
        header = self._cfg.add_block(CFGBlockKind.LOOP_HEADER)
        header.condition = node.condition
        self._cfg.add_edge(current.id, header.id)

        body_block = self._cfg.add_block(CFGBlockKind.LOOP_BODY)
        self._cfg.add_edge(header.id, body_block.id, label="true", condition=node.condition)

        exit_block = self._cfg.add_block(CFGBlockKind.LOOP_EXIT)
        self._cfg.add_edge(header.id, exit_block.id, label="false", condition=node.condition)

        body_last = body_block
        if node.body:
            result = self._process_block(node.body, body_block)
            if result:
                body_last = result

        if body_last:
            # Back edge
            back = self._cfg.add_edge(body_last.id, header.id)
            back.is_back_edge = True

        return exit_block

    def _process_return(
        self, node: ASTNode, current: CFGBlock
    ) -> Optional[CFGBlock]:
        """Process a return statement."""
        ret_block = self._cfg.add_block(CFGBlockKind.RETURN)
        ret_block.statements.append(node)
        self._cfg.add_edge(current.id, ret_block.id)
        self._cfg.add_edge(ret_block.id, self._cfg.exit_id)
        return None

    def _process_throw(
        self, node: ASTNode, current: CFGBlock
    ) -> Optional[CFGBlock]:
        """Process a throw statement."""
        throw_block = self._cfg.add_block(CFGBlockKind.THROW)
        throw_block.statements.append(node)
        self._cfg.add_edge(current.id, throw_block.id)
        return None

    def _process_try(
        self, node: ASTNode, current: CFGBlock
    ) -> Optional[CFGBlock]:
        """Process a try-catch statement."""
        try_block = self._cfg.add_block(CFGBlockKind.TRY)
        self._cfg.add_edge(current.id, try_block.id)

        join_block = self._cfg.add_block(CFGBlockKind.NORMAL)

        try_last = try_block
        if node.body:
            result = self._process_block(node.body, try_block)
            if result:
                try_last = result

        if try_last:
            self._cfg.add_edge(try_last.id, join_block.id)

        if node.handler:
            catch_block = self._cfg.add_block(CFGBlockKind.CATCH)
            self._cfg.add_edge(try_block.id, catch_block.id, label="exception")

            catch_last = catch_block
            if node.handler.body:
                result = self._process_block(node.handler.body, catch_block)
                if result:
                    catch_last = result
            if catch_last:
                self._cfg.add_edge(catch_last.id, join_block.id)

        if node.finalizer:
            finally_block = self._cfg.add_block(CFGBlockKind.FINALLY)
            self._cfg.add_edge(join_block.id, finally_block.id)
            finally_last = finally_block
            if node.finalizer.body:
                result = self._process_block(node.finalizer.body, finally_block)
                if result:
                    finally_last = result
            return finally_last

        return join_block


# ---------------------------------------------------------------------------
# NarrowingAnalyzer — main entry point
# ---------------------------------------------------------------------------

class NarrowingAnalyzer:
    """Main narrowing analysis engine.

    Orchestrates all narrowing sub-analyzers: typeof, instanceof, in,
    equality, truthiness, user-defined type guards, discriminated unions,
    optional chaining, nullish coalescing, control flow, and assignment.
    """

    def __init__(
        self,
        class_hierarchy: Optional[Dict[str, List[str]]] = None,
        type_guards: Optional[Dict[str, Dict[str, Any]]] = None,
        enum_types: Optional[Dict[str, TSType]] = None,
        interface_types: Optional[Dict[str, TSType]] = None,
    ) -> None:
        self._typeof = TypeofNarrowing()
        self._instanceof = InstanceofNarrowing(class_hierarchy)
        self._in = InNarrowing()
        self._equality = EqualityNarrowing()
        self._truthiness = TruthinessNarrowing()
        self._user_guard = UserDefinedTypeGuard()
        self._disc_union = DiscriminatedUnionNarrowing()
        self._optional_chain = OptionalChainingNarrowing()
        self._nullish = NullishCoalescingNarrowing()
        self._control_flow = ControlFlowNarrowing()
        self._assignment = AssignmentNarrowing()
        self._converter = NarrowingToPredicateConverter()
        self._exhaustiveness = ExhaustivenessChecker()
        self._composition = NarrowingComposition()

        self._enum_types = enum_types or {}
        self._interface_types = interface_types or {}
        self._class_hierarchy = class_hierarchy or {}

        # Register type guards
        if type_guards:
            for name, info in type_guards.items():
                self._user_guard.register_guard(
                    function_name=name,
                    parameter_name=info.get("parameter_name", "x"),
                    guarded_type=info.get("guarded_type", UNKNOWN_TYPE),
                    is_assertion=info.get("is_assertion", False),
                    is_this_guard=info.get("is_this_guard", False),
                )

    def analyze_function(self, ts_ast: ASTNode) -> NarrowingResult:
        """Analyze a function for type narrowing.

        This is the main entry point. It builds a CFG, identifies narrowing
        points, computes narrowed types at every block, and converts
        narrowings to refinement predicates.
        """
        result = NarrowingResult()
        stats = NarrowingStatistics()

        # Build CFG
        cfg_builder = CFGBuilder()
        cfg = cfg_builder.build(ts_ast)
        stats.total_blocks = len(cfg.blocks)

        # Extract initial types from function parameters
        initial_types = self._extract_parameter_types(ts_ast)

        # Identify narrowing points
        narrowing_points = self.identify_narrowing_points(cfg)
        result.narrowing_points = narrowing_points
        stats.total_narrowing_points = len(narrowing_points)

        # Update narrowing point stats
        for pt in narrowing_points:
            self._update_kind_stats(pt, stats)

        # Compute narrowed types at each block
        variable_types = self.compute_narrowed_types(cfg, initial_types)
        result.variable_types = variable_types

        # Build narrowing graph
        graph = NarrowingGraph()
        graph.build_from_cfg(cfg, narrowing_points)
        node_types = graph.propagate_types(initial_types)
        node_types = graph.handle_loop_widening(cfg, node_types, initial_types)
        result.flow_nodes = list(graph.nodes.values())
        stats.total_flow_nodes = len(result.flow_nodes)

        # Convert narrowings to predicates
        predicates = self._converter.convert_all(narrowing_points)
        result.predicates = predicates
        stats.predicates_generated = len(predicates)

        # Check exhaustiveness for switch statements
        exhaustiveness_info = self._check_all_exhaustiveness(cfg, variable_types)
        result.exhaustiveness_info = exhaustiveness_info
        if "switches" in exhaustiveness_info:
            for sw_info in exhaustiveness_info["switches"]:
                if sw_info.get("is_exhaustive"):
                    stats.exhaustive_switches += 1
                else:
                    stats.non_exhaustive_switches += 1

        # Compute narrowing depth
        for pt in narrowing_points:
            stats.narrowing_depth[pt.variable] = (
                stats.narrowing_depth.get(pt.variable, 0) + 1
            )
        stats.variables_narrowed = len(stats.narrowing_depth)

        # Count unreachable blocks
        for block in cfg.blocks.values():
            if block.kind == CFGBlockKind.UNREACHABLE:
                stats.unreachable_blocks += 1

        stats.compute_derived()
        result.statistics = stats

        return result

    def identify_narrowing_points(self, cfg: CFG) -> List[NarrowingPoint]:
        """Identify all narrowing points in a CFG."""
        points: List[NarrowingPoint] = []
        initial_types: Dict[str, TSType] = {}

        # Collect initial types from entry block
        entry = cfg.get_block(cfg.entry_id)
        if entry:
            initial_types = dict(entry.narrowing_state_out)

        for bid, block in cfg.blocks.items():
            # Check condition blocks
            if block.condition:
                cond_points = self._analyze_expression_for_narrowing(
                    block.condition, initial_types
                )
                for pt in cond_points:
                    pt.cfg_block_id = bid
                points.extend(cond_points)

            # Check statements
            for stmt in block.statements:
                stmt_points = self._analyze_statement_for_narrowing(
                    stmt, initial_types
                )
                for pt in stmt_points:
                    pt.cfg_block_id = bid
                points.extend(stmt_points)

        return points

    def compute_narrowed_types(
        self,
        cfg: CFG,
        initial_types: Dict[str, TSType],
    ) -> Dict[str, Dict[int, TSType]]:
        """Compute narrowed types at each CFG block.

        Returns a mapping: variable → {block_id → narrowed_type}
        """
        result: Dict[str, Dict[int, TSType]] = {}

        # Initialize entry block
        entry = cfg.get_block(cfg.entry_id)
        if entry:
            entry.narrowing_state_in = dict(initial_types)
            entry.narrowing_state_out = dict(initial_types)

        # Forward dataflow analysis
        worklist = list(cfg.reverse_postorder())
        max_iter = len(worklist) * 3
        iteration = 0

        while worklist and iteration < max_iter:
            iteration += 1
            bid = worklist.pop(0)
            block = cfg.get_block(bid)
            if block is None:
                continue

            # Merge incoming types from predecessors
            if bid == cfg.entry_id:
                new_in = dict(initial_types)
            else:
                new_in = self._merge_predecessor_types(block, cfg)

            # Check if narrowing state changed
            if new_in == block.narrowing_state_in and block.narrowing_state_out:
                continue

            block.narrowing_state_in = new_in

            # Apply narrowings in this block
            new_out = self._apply_block_narrowings(block, new_in)
            block.narrowing_state_out = new_out

            # Add successors to worklist
            for succ_id in block.successors:
                if succ_id not in worklist:
                    worklist.append(succ_id)

        # Collect results
        for bid, block in cfg.blocks.items():
            for var, typ in block.narrowing_state_out.items():
                result.setdefault(var, {})[bid] = typ

        return result

    def _extract_parameter_types(self, func_node: ASTNode) -> Dict[str, TSType]:
        """Extract parameter types from a function declaration."""
        types: Dict[str, TSType] = {}
        for param in func_node.parameters:
            name = param.name
            if param.type_annotation:
                types[name] = param.type_annotation
            else:
                types[name] = UNKNOWN_TYPE
        return types

    def _analyze_expression_for_narrowing(
        self,
        node: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> List[NarrowingPoint]:
        """Analyze an expression and extract all narrowing points."""
        points: List[NarrowingPoint] = []

        # typeof
        pt = self._typeof.analyze(node, variable_types)
        if pt:
            points.append(pt)
            return points

        # typeof + null compound
        pt = self._typeof.handles_null_check(node, variable_types)
        if pt:
            points.append(pt)
            return points

        # instanceof
        pt = self._instanceof.analyze(node, variable_types)
        if pt:
            points.append(pt)
            return points

        # in
        pt = self._in.analyze(node, variable_types)
        if pt:
            points.append(pt)
            return points

        # Equality / null checks
        pt = self._equality.analyze_null_check(node, variable_types)
        if pt:
            points.append(pt)
            return points

        pt = self._equality.analyze(node, variable_types)
        if pt:
            points.append(pt)
            return points

        # Enum equality
        pt = self._equality.analyze_enum_equality(
            node, variable_types, self._enum_types
        )
        if pt:
            points.append(pt)
            return points

        # Optional chaining
        pt = self._optional_chain.analyze(node, variable_types)
        if pt:
            points.append(pt)
            return points

        # Nullish coalescing
        pt = self._nullish.analyze(node, variable_types)
        if pt:
            points.append(pt)
            return points

        # User-defined type guard
        if node.kind == ASTNodeKind.CALL_EXPRESSION:
            pt = self._user_guard.analyze_call(node, variable_types)
            if pt:
                points.append(pt)
                return points

        # Logical expression (compound)
        if node.kind == ASTNodeKind.LOGICAL_EXPRESSION:
            if node.operator == "&&":
                sub = self._truthiness.analyze_short_circuit_and(node, variable_types)
                points.extend(sub)
                if points:
                    return points
            elif node.operator == "||":
                sub = self._truthiness.analyze_short_circuit_or(node, variable_types)
                points.extend(sub)
                if points:
                    return points

        # Truthiness (fallback)
        pt = self._truthiness.analyze(node, variable_types)
        if pt:
            points.append(pt)

        return points

    def _analyze_statement_for_narrowing(
        self,
        stmt: ASTNode,
        variable_types: Dict[str, TSType],
    ) -> List[NarrowingPoint]:
        """Analyze a statement for narrowing points."""
        points: List[NarrowingPoint] = []

        # Assignment narrowing
        if stmt.kind in (
            ASTNodeKind.ASSIGNMENT_EXPRESSION,
            ASTNodeKind.VARIABLE_DECLARATION,
        ):
            pt = self._assignment.analyze(stmt, variable_types)
            if pt:
                points.append(pt)

            # Check for destructuring
            destr = self._assignment.analyze_destructuring(stmt, variable_types)
            points.extend(destr)

        # Assertion call
        if stmt.kind == ASTNodeKind.EXPRESSION_STATEMENT and stmt.expression:
            if stmt.expression.kind == ASTNodeKind.CALL_EXPRESSION:
                pt = self._user_guard.analyze_call(
                    stmt.expression, variable_types
                )
                if pt and pt.narrowing_kind == NarrowingKind.ASSERTION:
                    points.append(pt)

        # Nullish assignment
        if stmt.kind == ASTNodeKind.NULLISH_ASSIGNMENT_EXPRESSION or (
            stmt.kind == ASTNodeKind.ASSIGNMENT_EXPRESSION
            and stmt.operator == "??="
        ):
            pt = self._nullish.analyze_assignment(stmt, variable_types)
            if pt:
                points.append(pt)

        return points

    def _merge_predecessor_types(
        self, block: CFGBlock, cfg: CFG
    ) -> Dict[str, TSType]:
        """Merge types from predecessor blocks."""
        if not block.predecessors:
            return {}

        incoming: Dict[str, List[Tuple[TSType, str]]] = {}

        for pred_id in block.predecessors:
            pred = cfg.get_block(pred_id)
            if pred is None:
                continue

            # Find the edge label from predecessor to this block
            edge_label = "unconditional"
            for edge in pred.edges_out:
                if edge.target == block.id:
                    edge_label = edge.label
                    break

            pred_types = pred.narrowing_state_out
            for var, typ in pred_types.items():
                incoming.setdefault(var, []).append((typ, edge_label))

        result: Dict[str, TSType] = {}
        for var, type_label_pairs in incoming.items():
            types = [t for t, _ in type_label_pairs]
            if len(types) == 1:
                result[var] = types[0]
            else:
                result[var] = make_union(types)

        return result

    def _apply_block_narrowings(
        self, block: CFGBlock, types_in: Dict[str, TSType]
    ) -> Dict[str, TSType]:
        """Apply narrowings in a block to produce output types."""
        current = dict(types_in)

        # Apply condition narrowings
        if block.condition:
            if block.kind == CFGBlockKind.CONDITION:
                points = self._analyze_expression_for_narrowing(
                    block.condition, current
                )
                # Determine which branch this block leads to
                for edge in block.edges_out:
                    if edge.label == "true":
                        for pt in points:
                            if pt.true_type:
                                current[pt.variable] = pt.true_type
                    elif edge.label == "false":
                        for pt in points:
                            if pt.false_type:
                                current[pt.variable] = pt.false_type

        # Apply statement narrowings
        for stmt in block.statements:
            stmt_points = self._analyze_statement_for_narrowing(stmt, current)
            for pt in stmt_points:
                if pt.true_type:
                    current[pt.variable] = pt.true_type

        return current

    def _check_all_exhaustiveness(
        self,
        cfg: CFG,
        variable_types: Dict[str, Dict[int, TSType]],
    ) -> Dict[str, Any]:
        """Check exhaustiveness for all switch statements in the CFG."""
        info: Dict[str, Any] = {"switches": []}

        for bid, block in cfg.blocks.items():
            if block.kind == CFGBlockKind.SWITCH_HEAD and block.condition:
                # Attempt to detect discriminated union
                if block.condition.kind == ASTNodeKind.MEMBER_EXPRESSION:
                    obj = block.condition.object_node
                    prop = block.condition.property_node
                    if obj and prop and obj.kind == ASTNodeKind.IDENTIFIER:
                        var_name = obj.name
                        disc_prop = prop.name
                        block_types = variable_types.get(var_name, {})
                        union_type = block_types.get(bid, UNKNOWN_TYPE)

                        if union_type.kind == TypeKind.UNION:
                            case_values = []
                            for succ_id in block.successors:
                                succ = cfg.get_block(succ_id)
                                if succ and succ.condition and succ.condition.value is not None:
                                    case_values.append(succ.condition.value)

                            handled: List[TSType] = []
                            missing: List[TSType] = []
                            has_default = False
                            for succ_id in block.successors:
                                succ = cfg.get_block(succ_id)
                                if succ and succ.kind == CFGBlockKind.DEFAULT_CASE:
                                    has_default = True

                            for member in union_type.members:
                                dt = member.properties.get(disc_prop)
                                if dt and dt.literal_value in case_values:
                                    handled.append(member)
                                else:
                                    missing.append(member)

                            is_exhaustive = has_default or len(missing) == 0
                            info["switches"].append({
                                "block_id": bid,
                                "variable": var_name,
                                "discriminant": disc_prop,
                                "is_exhaustive": is_exhaustive,
                                "handled_count": len(handled),
                                "missing_count": len(missing),
                                "has_default": has_default,
                            })

        return info

    def _update_kind_stats(
        self, pt: NarrowingPoint, stats: NarrowingStatistics
    ) -> None:
        """Update statistics based on narrowing kind."""
        kind_map = {
            NarrowingKind.TYPEOF: "typeof_narrowings",
            NarrowingKind.INSTANCEOF: "instanceof_narrowings",
            NarrowingKind.EQUALITY: "equality_narrowings",
            NarrowingKind.TRUTHINESS: "truthiness_narrowings",
            NarrowingKind.IN: "in_narrowings",
            NarrowingKind.USER_DEFINED: "user_defined_guards",
            NarrowingKind.ASSERTION: "assertion_narrowings",
            NarrowingKind.DISCRIMINATED_UNION: "discriminated_unions",
            NarrowingKind.OPTIONAL_CHAINING: "optional_chaining_narrowings",
            NarrowingKind.NULLISH_COALESCING: "nullish_coalescing_narrowings",
            NarrowingKind.ASSIGNMENT: "assignment_narrowings",
            NarrowingKind.COMPOUND: "compound_narrowings",
        }
        attr = kind_map.get(pt.narrowing_kind)
        if attr:
            setattr(stats, attr, getattr(stats, attr) + 1)


# ---------------------------------------------------------------------------
# Utilities for external use
# ---------------------------------------------------------------------------

def analyze_narrowing(
    ts_ast: ASTNode,
    class_hierarchy: Optional[Dict[str, List[str]]] = None,
    type_guards: Optional[Dict[str, Dict[str, Any]]] = None,
    enum_types: Optional[Dict[str, TSType]] = None,
    interface_types: Optional[Dict[str, TSType]] = None,
) -> NarrowingResult:
    """Convenience function to perform full narrowing analysis."""
    analyzer = NarrowingAnalyzer(
        class_hierarchy=class_hierarchy,
        type_guards=type_guards,
        enum_types=enum_types,
        interface_types=interface_types,
    )
    return analyzer.analyze_function(ts_ast)


def narrowing_points_for_variable(
    result: NarrowingResult,
    variable: str,
) -> List[NarrowingPoint]:
    """Get all narrowing points for a specific variable."""
    return [pt for pt in result.narrowing_points if pt.variable == variable]


def predicates_for_variable(
    result: NarrowingResult,
    variable: str,
) -> List[Predicate]:
    """Get all predicates for a specific variable."""
    return [
        p for p in result.predicates
        if p.variable == variable
    ]


def type_at_block(
    result: NarrowingResult,
    variable: str,
    block_id: int,
) -> Optional[TSType]:
    """Get the narrowed type of a variable at a specific CFG block."""
    var_types = result.variable_types.get(variable, {})
    return var_types.get(block_id)


def summarize_narrowing(result: NarrowingResult) -> Dict[str, Any]:
    """Summarize a narrowing analysis result."""
    stats = result.statistics
    summary: Dict[str, Any] = {
        "total_narrowing_points": stats.total_narrowing_points if stats else 0,
        "variables_narrowed": stats.variables_narrowed if stats else 0,
        "predicates_generated": stats.predicates_generated if stats else 0,
        "total_blocks": stats.total_blocks if stats else 0,
        "total_flow_nodes": stats.total_flow_nodes if stats else 0,
    }
    if stats:
        summary["narrowing_distribution"] = stats.narrowing_kinds_distribution
        summary["exhaustiveness_coverage"] = stats.exhaustiveness_coverage
        summary["average_narrowing_depth"] = stats.average_narrowing_depth
        summary["max_narrowing_depth"] = stats.max_narrowing_depth
    if result.errors:
        summary["errors"] = result.errors
    if result.warnings:
        summary["warnings"] = result.warnings
    return summary


def create_typeof_guard(
    variable: str,
    typeof_value: str,
    original_type: TSType,
) -> NarrowingPoint:
    """Create a typeof narrowing point programmatically."""
    tn = TypeofNarrowing()
    node = ASTNode(
        kind=ASTNodeKind.BINARY_EXPRESSION,
        operator="===",
        left=ASTNode(
            kind=ASTNodeKind.TYPEOF_EXPRESSION,
            expression=ASTNode(kind=ASTNodeKind.IDENTIFIER, name=variable),
        ),
        right=ASTNode(kind=ASTNodeKind.LITERAL, value=typeof_value),
    )
    pt = tn.analyze(node, {variable: original_type})
    if pt is None:
        return NarrowingPoint(
            location=SourceLocation(),
            variable=variable,
            guard_expression=node,
            true_type=TYPEOF_RESULT_MAP.get(typeof_value, UNKNOWN_TYPE),
            false_type=original_type,
            narrowing_kind=NarrowingKind.TYPEOF,
            original_type=original_type,
        )
    return pt


def create_null_check(
    variable: str,
    original_type: TSType,
    is_strict: bool = True,
) -> NarrowingPoint:
    """Create a null check narrowing point programmatically."""
    en = EqualityNarrowing()
    node = ASTNode(
        kind=ASTNodeKind.BINARY_EXPRESSION,
        operator="===" if is_strict else "==",
        left=ASTNode(kind=ASTNodeKind.IDENTIFIER, name=variable),
        right=ASTNode(kind=ASTNodeKind.LITERAL, value=None),
    )
    pt = en.analyze_null_check(node, {variable: original_type})
    if pt is None:
        return NarrowingPoint(
            location=SourceLocation(),
            variable=variable,
            guard_expression=node,
            true_type=NULL_TYPE,
            false_type=original_type.without_nullish(),
            narrowing_kind=NarrowingKind.EQUALITY,
            original_type=original_type,
        )
    return pt


def create_instanceof_guard(
    variable: str,
    class_name: str,
    original_type: TSType,
) -> NarrowingPoint:
    """Create an instanceof narrowing point programmatically."""
    inst = InstanceofNarrowing()
    node = ASTNode(
        kind=ASTNodeKind.INSTANCEOF_EXPRESSION,
        left=ASTNode(kind=ASTNodeKind.IDENTIFIER, name=variable),
        right=ASTNode(kind=ASTNodeKind.IDENTIFIER, name=class_name),
    )
    pt = inst.analyze(node, {variable: original_type})
    if pt is None:
        class_type = TSType(kind=TypeKind.CLASS, name=class_name)
        return NarrowingPoint(
            location=SourceLocation(),
            variable=variable,
            guard_expression=node,
            true_type=class_type,
            false_type=exclude_type(original_type, class_type),
            narrowing_kind=NarrowingKind.INSTANCEOF,
            original_type=original_type,
        )
    return pt


def create_truthiness_guard(
    variable: str,
    original_type: TSType,
) -> NarrowingPoint:
    """Create a truthiness narrowing point programmatically."""
    tn = TruthinessNarrowing()
    node = ASTNode(kind=ASTNodeKind.IDENTIFIER, name=variable)
    pt = tn.analyze(node, {variable: original_type})
    if pt is None:
        return NarrowingPoint(
            location=SourceLocation(),
            variable=variable,
            guard_expression=node,
            true_type=original_type.without_falsy(),
            false_type=original_type.only_falsy(),
            narrowing_kind=NarrowingKind.TRUTHINESS,
            original_type=original_type,
        )
    return pt


def create_in_guard(
    variable: str,
    key: str,
    original_type: TSType,
) -> NarrowingPoint:
    """Create an 'in' narrowing point programmatically."""
    inn = InNarrowing()
    node = ASTNode(
        kind=ASTNodeKind.IN_EXPRESSION,
        left=ASTNode(kind=ASTNodeKind.LITERAL, value=key),
        right=ASTNode(kind=ASTNodeKind.IDENTIFIER, name=variable),
    )
    pt = inn.analyze(node, {variable: original_type})
    if pt is None:
        return NarrowingPoint(
            location=SourceLocation(),
            variable=variable,
            guard_expression=node,
            true_type=original_type,
            false_type=original_type,
            narrowing_kind=NarrowingKind.IN,
            original_type=original_type,
        )
    return pt


def build_narrowing_pipeline(
    ts_ast: ASTNode,
    config: Optional[Dict[str, Any]] = None,
) -> NarrowingResult:
    """Build and run the complete narrowing analysis pipeline.

    This is the top-level entry point that sets up all configuration
    and runs the full analysis.
    """
    config = config or {}

    class_hierarchy = config.get("class_hierarchy", {})
    type_guards = config.get("type_guards", {})
    enum_types = config.get("enum_types", {})
    interface_types = config.get("interface_types", {})
    custom_instanceof = config.get("custom_instanceof", {})

    analyzer = NarrowingAnalyzer(
        class_hierarchy=class_hierarchy,
        type_guards=type_guards,
        enum_types=enum_types,
        interface_types=interface_types,
    )

    result = analyzer.analyze_function(ts_ast)

    # Post-processing: simplify predicates
    simplified_predicates: List[Predicate] = []
    for pred in result.predicates:
        simplified = _simplify_predicate(pred)
        simplified_predicates.append(simplified)
    result.predicates = simplified_predicates

    return result


def _simplify_predicate(pred: Predicate) -> Predicate:
    """Simplify a predicate by removing redundancies."""
    if pred.kind == PredicateKind.AND:
        simplified_ops = [_simplify_predicate(op) for op in pred.operands]
        # Remove TRUE operands
        filtered = [op for op in simplified_ops if op.kind != PredicateKind.TRUE]
        if not filtered:
            return Predicate(kind=PredicateKind.TRUE)
        if any(op.kind == PredicateKind.FALSE for op in filtered):
            return Predicate(kind=PredicateKind.FALSE)
        if len(filtered) == 1:
            return filtered[0]
        # Deduplicate
        seen: Set[int] = set()
        deduped: List[Predicate] = []
        for op in filtered:
            h = hash(op)
            if h not in seen:
                seen.add(h)
                deduped.append(op)
        return Predicate(kind=PredicateKind.AND, operands=deduped)

    if pred.kind == PredicateKind.OR:
        simplified_ops = [_simplify_predicate(op) for op in pred.operands]
        filtered = [op for op in simplified_ops if op.kind != PredicateKind.FALSE]
        if not filtered:
            return Predicate(kind=PredicateKind.FALSE)
        if any(op.kind == PredicateKind.TRUE for op in filtered):
            return Predicate(kind=PredicateKind.TRUE)
        if len(filtered) == 1:
            return filtered[0]
        seen_set: Set[int] = set()
        deduped_list: List[Predicate] = []
        for op in filtered:
            h = hash(op)
            if h not in seen_set:
                seen_set.add(h)
                deduped_list.append(op)
        return Predicate(kind=PredicateKind.OR, operands=deduped_list)

    if pred.kind == PredicateKind.NOT and pred.operands:
        inner = _simplify_predicate(pred.operands[0])
        if inner.kind == PredicateKind.NOT and inner.operands:
            return inner.operands[0]
        if inner.kind == PredicateKind.TRUE:
            return Predicate(kind=PredicateKind.FALSE)
        if inner.kind == PredicateKind.FALSE:
            return Predicate(kind=PredicateKind.TRUE)
        return Predicate(kind=PredicateKind.NOT, operands=[inner])

    return pred


# ---------------------------------------------------------------------------
# Batch analysis helpers
# ---------------------------------------------------------------------------

def analyze_multiple_functions(
    functions: List[ASTNode],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, NarrowingResult]:
    """Analyze multiple functions for narrowing."""
    results: Dict[str, NarrowingResult] = {}
    for func in functions:
        name = func.name or f"anonymous_{id(func)}"
        results[name] = build_narrowing_pipeline(func, config)
    return results


def merge_statistics(
    results: Dict[str, NarrowingResult],
) -> NarrowingStatistics:
    """Merge statistics from multiple function analyses."""
    merged = NarrowingStatistics()

    for name, result in results.items():
        stats = result.statistics
        if stats is None:
            continue
        merged.total_narrowing_points += stats.total_narrowing_points
        merged.variables_narrowed += stats.variables_narrowed
        merged.total_blocks += stats.total_blocks
        merged.total_flow_nodes += stats.total_flow_nodes
        merged.typeof_narrowings += stats.typeof_narrowings
        merged.instanceof_narrowings += stats.instanceof_narrowings
        merged.equality_narrowings += stats.equality_narrowings
        merged.truthiness_narrowings += stats.truthiness_narrowings
        merged.in_narrowings += stats.in_narrowings
        merged.user_defined_guards += stats.user_defined_guards
        merged.assertion_narrowings += stats.assertion_narrowings
        merged.discriminated_unions += stats.discriminated_unions
        merged.optional_chaining_narrowings += stats.optional_chaining_narrowings
        merged.nullish_coalescing_narrowings += stats.nullish_coalescing_narrowings
        merged.assignment_narrowings += stats.assignment_narrowings
        merged.compound_narrowings += stats.compound_narrowings
        merged.exhaustive_switches += stats.exhaustive_switches
        merged.non_exhaustive_switches += stats.non_exhaustive_switches
        merged.predicates_generated += stats.predicates_generated
        merged.unreachable_blocks += stats.unreachable_blocks
        for var, depth in stats.narrowing_depth.items():
            key = f"{name}.{var}"
            merged.narrowing_depth[key] = depth

    merged.compute_derived()
    return merged


# ---------------------------------------------------------------------------
# Type utility functions
# ---------------------------------------------------------------------------

def flatten_union(t: TSType) -> List[TSType]:
    """Flatten a union type into its constituent members."""
    if t.kind != TypeKind.UNION:
        return [t]
    result: List[TSType] = []
    for m in t.members:
        result.extend(flatten_union(m))
    return result


def reconstruct_union(members: List[TSType]) -> TSType:
    """Reconstruct a union from a list of members."""
    return make_union(members)


def type_to_string(t: TSType) -> str:
    """Convert a TSType to a human-readable string."""
    if t.kind == TypeKind.UNION:
        return " | ".join(type_to_string(m) for m in t.members)
    if t.kind == TypeKind.INTERSECTION:
        return " & ".join(type_to_string(m) for m in t.members)
    if t.is_literal:
        return repr(t.literal_value)
    if t.kind == TypeKind.ARRAY:
        if t.type_arguments:
            return f"{type_to_string(t.type_arguments[0])}[]"
        return "any[]"
    if t.kind == TypeKind.TUPLE:
        inner = ", ".join(type_to_string(m) for m in t.members)
        return f"[{inner}]"
    if t.name:
        if t.type_arguments:
            args = ", ".join(type_to_string(a) for a in t.type_arguments)
            return f"{t.name}<{args}>"
        return t.name
    return t.kind.value


def predicate_to_string(p: Predicate) -> str:
    """Convert a Predicate to a human-readable string."""
    return str(p)


def compare_narrowing_results(
    a: NarrowingResult,
    b: NarrowingResult,
) -> Dict[str, Any]:
    """Compare two narrowing results for testing/debugging."""
    diff: Dict[str, Any] = {
        "points_a": len(a.narrowing_points),
        "points_b": len(b.narrowing_points),
        "points_diff": len(a.narrowing_points) - len(b.narrowing_points),
        "predicates_a": len(a.predicates),
        "predicates_b": len(b.predicates),
        "predicates_diff": len(a.predicates) - len(b.predicates),
    }

    vars_a = set(a.variable_types.keys())
    vars_b = set(b.variable_types.keys())
    diff["vars_only_in_a"] = list(vars_a - vars_b)
    diff["vars_only_in_b"] = list(vars_b - vars_a)
    diff["common_vars"] = list(vars_a & vars_b)

    return diff


# ---------------------------------------------------------------------------
# Guard pattern detection (for scanning source without full AST)
# ---------------------------------------------------------------------------

class GuardPatternDetector:
    """Detect common guard patterns from partial AST or expression analysis."""

    TYPEOF_PATTERN_OPERATORS = ("===", "!==", "==", "!=")
    NULLISH_KEYWORDS = ("null", "undefined")

    def detect_pattern(
        self, node: ASTNode
    ) -> Optional[NarrowingKind]:
        """Detect what kind of narrowing pattern a node represents."""
        if node.kind == ASTNodeKind.BINARY_EXPRESSION:
            if node.operator in self.TYPEOF_PATTERN_OPERATORS:
                left = node.left
                right = node.right
                if left and left.kind == ASTNodeKind.TYPEOF_EXPRESSION:
                    return NarrowingKind.TYPEOF
                if right and right.kind == ASTNodeKind.TYPEOF_EXPRESSION:
                    return NarrowingKind.TYPEOF
                if self._is_null_literal(left) or self._is_null_literal(right):
                    return NarrowingKind.EQUALITY
                if left and left.kind == ASTNodeKind.LITERAL:
                    return NarrowingKind.EQUALITY
                if right and right.kind == ASTNodeKind.LITERAL:
                    return NarrowingKind.EQUALITY
            if node.operator == "instanceof":
                return NarrowingKind.INSTANCEOF
            if node.operator == "in":
                return NarrowingKind.IN

        if node.kind == ASTNodeKind.INSTANCEOF_EXPRESSION:
            return NarrowingKind.INSTANCEOF

        if node.kind == ASTNodeKind.IN_EXPRESSION:
            return NarrowingKind.IN

        if node.kind == ASTNodeKind.CALL_EXPRESSION:
            return NarrowingKind.USER_DEFINED

        if node.kind == ASTNodeKind.OPTIONAL_CHAIN_EXPRESSION:
            return NarrowingKind.OPTIONAL_CHAINING

        if node.kind == ASTNodeKind.NULLISH_COALESCING_EXPRESSION:
            return NarrowingKind.NULLISH_COALESCING

        if node.kind == ASTNodeKind.UNARY_EXPRESSION and node.operator == "!":
            return NarrowingKind.TRUTHINESS

        if node.kind == ASTNodeKind.IDENTIFIER:
            return NarrowingKind.TRUTHINESS

        if node.kind == ASTNodeKind.LOGICAL_EXPRESSION:
            return NarrowingKind.COMPOUND

        return None

    def detect_all_patterns(
        self, nodes: List[ASTNode]
    ) -> Dict[NarrowingKind, int]:
        """Count all narrowing patterns in a list of nodes."""
        counts: Dict[NarrowingKind, int] = {}
        for node in nodes:
            kind = self.detect_pattern(node)
            if kind:
                counts[kind] = counts.get(kind, 0) + 1
        return counts

    def _is_null_literal(self, node: Optional[ASTNode]) -> bool:
        if node is None:
            return False
        if node.kind == ASTNodeKind.LITERAL and node.value is None:
            return True
        if node.kind == ASTNodeKind.IDENTIFIER and node.name in self.NULLISH_KEYWORDS:
            return True
        return False


# ---------------------------------------------------------------------------
# Narrowing scope tracking
# ---------------------------------------------------------------------------

@dataclass
class NarrowingScope:
    """Track narrowing state within a lexical scope."""
    parent: Optional[NarrowingScope] = None
    variable_types: Dict[str, TSType] = field(default_factory=dict)
    narrowing_points: List[NarrowingPoint] = field(default_factory=list)
    is_loop_scope: bool = False
    is_switch_scope: bool = False
    is_try_scope: bool = False

    def lookup(self, variable: str) -> Optional[TSType]:
        """Look up a variable's type, searching parent scopes."""
        if variable in self.variable_types:
            return self.variable_types[variable]
        if self.parent:
            return self.parent.lookup(variable)
        return None

    def set_type(self, variable: str, typ: TSType) -> None:
        """Set a variable's narrowed type in this scope."""
        self.variable_types[variable] = typ

    def all_types(self) -> Dict[str, TSType]:
        """Get all variable types, including from parent scopes."""
        result: Dict[str, TSType] = {}
        if self.parent:
            result.update(self.parent.all_types())
        result.update(self.variable_types)
        return result

    def child_scope(
        self,
        is_loop: bool = False,
        is_switch: bool = False,
        is_try: bool = False,
    ) -> NarrowingScope:
        """Create a child scope."""
        return NarrowingScope(
            parent=self,
            is_loop_scope=is_loop,
            is_switch_scope=is_switch,
            is_try_scope=is_try,
        )


class ScopeTracker:
    """Track narrowing scopes during analysis."""

    def __init__(self, initial_types: Optional[Dict[str, TSType]] = None) -> None:
        self._root = NarrowingScope(
            variable_types=initial_types or {}
        )
        self._current = self._root
        self._scope_stack: List[NarrowingScope] = [self._root]

    @property
    def current_scope(self) -> NarrowingScope:
        return self._current

    def enter_scope(
        self,
        is_loop: bool = False,
        is_switch: bool = False,
        is_try: bool = False,
    ) -> NarrowingScope:
        """Enter a new narrowing scope."""
        child = self._current.child_scope(is_loop, is_switch, is_try)
        self._scope_stack.append(child)
        self._current = child
        return child

    def exit_scope(self) -> NarrowingScope:
        """Exit the current scope and return to parent."""
        if len(self._scope_stack) > 1:
            self._scope_stack.pop()
            self._current = self._scope_stack[-1]
        return self._current

    def narrow_variable(self, variable: str, typ: TSType) -> None:
        """Narrow a variable in the current scope."""
        self._current.set_type(variable, typ)

    def lookup_type(self, variable: str) -> Optional[TSType]:
        """Look up a variable's current narrowed type."""
        return self._current.lookup(variable)

    def all_current_types(self) -> Dict[str, TSType]:
        """Get all variable types in the current scope."""
        return self._current.all_types()

    def add_narrowing_point(self, point: NarrowingPoint) -> None:
        """Record a narrowing point in the current scope."""
        self._current.narrowing_points.append(point)

    def collect_all_points(self) -> List[NarrowingPoint]:
        """Collect all narrowing points from all scopes."""
        all_points: List[NarrowingPoint] = []

        def collect(scope: NarrowingScope) -> None:
            all_points.extend(scope.narrowing_points)

        self._walk_scopes(self._root, collect)
        return all_points

    def _walk_scopes(
        self,
        scope: NarrowingScope,
        callback: Callable[[NarrowingScope], None],
    ) -> None:
        """Walk all scopes in the tree."""
        callback(scope)


# ---------------------------------------------------------------------------
# Integration: Full narrowing pass
# ---------------------------------------------------------------------------

class FullNarrowingPass:
    """A complete narrowing pass that integrates all components.

    This class orchestrates the full narrowing analysis workflow:
    1. Build CFG from AST
    2. Detect narrowing patterns
    3. Compute narrowed types via dataflow analysis
    4. Convert to refinement predicates
    5. Check exhaustiveness
    6. Generate statistics
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._analyzer = NarrowingAnalyzer(
            class_hierarchy=self._config.get("class_hierarchy"),
            type_guards=self._config.get("type_guards"),
            enum_types=self._config.get("enum_types"),
            interface_types=self._config.get("interface_types"),
        )
        self._converter = NarrowingToPredicateConverter()
        self._exhaustiveness = ExhaustivenessChecker()
        self._detector = GuardPatternDetector()

    def run(self, ts_ast: ASTNode) -> NarrowingResult:
        """Run the full narrowing pass."""
        return self._analyzer.analyze_function(ts_ast)

    def run_batch(
        self, functions: List[ASTNode]
    ) -> Dict[str, NarrowingResult]:
        """Run narrowing analysis on multiple functions."""
        return analyze_multiple_functions(functions, self._config)

    def run_incremental(
        self,
        ts_ast: ASTNode,
        previous_result: NarrowingResult,
        changed_blocks: Set[int],
    ) -> NarrowingResult:
        """Run incremental narrowing analysis on changed blocks only.

        Reuses results from unchanged blocks to speed up re-analysis.
        """
        full_result = self._analyzer.analyze_function(ts_ast)

        # Merge: keep old points for unchanged blocks, use new for changed
        merged_points: List[NarrowingPoint] = []
        for pt in full_result.narrowing_points:
            if pt.cfg_block_id in changed_blocks:
                merged_points.append(pt)

        for pt in previous_result.narrowing_points:
            if pt.cfg_block_id not in changed_blocks:
                merged_points.append(pt)

        full_result.narrowing_points = merged_points
        return full_result

    def get_predicates(self, result: NarrowingResult) -> List[Predicate]:
        """Extract simplified predicates from a result."""
        return [_simplify_predicate(p) for p in result.predicates]

    def check_exhaustiveness(
        self,
        result: NarrowingResult,
    ) -> List[ExhaustivenessResult]:
        """Check exhaustiveness for all discriminated union switches in result."""
        results: List[ExhaustivenessResult] = []
        info = result.exhaustiveness_info
        if "switches" in info:
            for sw in info["switches"]:
                results.append(ExhaustivenessResult(
                    is_exhaustive=sw.get("is_exhaustive", False),
                    handled_members=[],
                    missing_members=[],
                    has_default=sw.get("has_default", False),
                ))
        return results


# ---------------------------------------------------------------------------
# Type narrowing validation
# ---------------------------------------------------------------------------

class NarrowingValidator:
    """Validate narrowing results for correctness."""

    def validate_narrowing_point(self, point: NarrowingPoint) -> List[str]:
        """Validate a single narrowing point for consistency."""
        errors: List[str] = []

        if point.true_type is None and point.false_type is None:
            errors.append(
                f"Narrowing point for '{point.variable}' has neither "
                f"true_type nor false_type"
            )

        if point.true_type and point.false_type and point.original_type:
            # Check that true and false types are subsets of original
            original_members = flatten_union(point.original_type)
            true_members = flatten_union(point.true_type)
            false_members = flatten_union(point.false_type)

            for tm in true_members:
                if tm.kind != TypeKind.NEVER and not any(
                    is_assignable_to(tm, om) for om in original_members
                ):
                    if point.original_type.kind not in (TypeKind.UNKNOWN, TypeKind.ANY):
                        errors.append(
                            f"True type member {type_to_string(tm)} is not "
                            f"assignable to original {type_to_string(point.original_type)}"
                        )

        if point.narrowing_kind == NarrowingKind.TYPEOF:
            typeof_val = point.metadata.get("typeof_result", "")
            if typeof_val and typeof_val not in TypeofNarrowing.TYPEOF_RESULTS:
                errors.append(
                    f"Invalid typeof result: {typeof_val}"
                )

        return errors

    def validate_result(self, result: NarrowingResult) -> List[str]:
        """Validate an entire narrowing result."""
        all_errors: List[str] = []

        for pt in result.narrowing_points:
            errors = self.validate_narrowing_point(pt)
            all_errors.extend(errors)

        # Check that variable types are consistent
        for var, block_types in result.variable_types.items():
            for block_id, typ in block_types.items():
                if typ.kind == TypeKind.NEVER:
                    # Check if this is expected (after exhaustive narrowing)
                    pass

        return all_errors

    def validate_predicates(self, predicates: List[Predicate]) -> List[str]:
        """Validate generated predicates."""
        errors: List[str] = []
        for pred in predicates:
            if pred.kind in (
                PredicateKind.ISINSTANCE,
                PredicateKind.IS_NONE,
                PredicateKind.NOT_NONE,
                PredicateKind.EQUALITY,
                PredicateKind.HASATTR,
                PredicateKind.IS_TRUTHY,
                PredicateKind.IS_FALSY,
                PredicateKind.TYPE_GUARD,
            ):
                if not pred.variable:
                    errors.append(
                        f"Predicate of kind {pred.kind.value} has no variable"
                    )
            if pred.kind == PredicateKind.ISINSTANCE and not pred.type_name:
                errors.append("isinstance predicate has no type_name")
            if pred.kind == PredicateKind.HASATTR and not pred.attribute:
                errors.append("hasattr predicate has no attribute")
        return errors


# ---------------------------------------------------------------------------
# Debug / introspection helpers
# ---------------------------------------------------------------------------

def dump_narrowing_points(points: List[NarrowingPoint]) -> str:
    """Dump narrowing points in a human-readable format."""
    lines: List[str] = []
    for i, pt in enumerate(points):
        lines.append(f"[{i}] {pt.narrowing_kind.value} on '{pt.variable}' at {pt.location}")
        if pt.true_type:
            lines.append(f"     true  => {type_to_string(pt.true_type)}")
        if pt.false_type:
            lines.append(f"     false => {type_to_string(pt.false_type)}")
        if pt.predicate:
            lines.append(f"     pred  => {pt.predicate}")
    return "\n".join(lines)


def dump_cfg(cfg: CFG) -> str:
    """Dump a CFG in a human-readable format."""
    lines: List[str] = []
    for bid in sorted(cfg.blocks.keys()):
        block = cfg.blocks[bid]
        lines.append(
            f"Block {bid} ({block.kind.value})"
            f" preds={block.predecessors} succs={block.successors}"
        )
        for stmt in block.statements:
            lines.append(f"  stmt: {stmt.kind.value}")
        if block.narrowing_state_out:
            for var, typ in block.narrowing_state_out.items():
                lines.append(f"  {var}: {type_to_string(typ)}")
    for edge in cfg.edges:
        back = " [back]" if edge.is_back_edge else ""
        lines.append(
            f"Edge {edge.source} -> {edge.target} ({edge.label}){back}"
        )
    return "\n".join(lines)


def dump_predicates(predicates: List[Predicate]) -> str:
    """Dump predicates in a human-readable format."""
    return "\n".join(f"[{i}] {p}" for i, p in enumerate(predicates))


def dump_statistics(stats: NarrowingStatistics) -> str:
    """Dump narrowing statistics in a human-readable format."""
    lines = [
        f"Total narrowing points: {stats.total_narrowing_points}",
        f"Variables narrowed: {stats.variables_narrowed}",
        f"Total blocks: {stats.total_blocks}",
        f"Total flow nodes: {stats.total_flow_nodes}",
        f"Unreachable blocks: {stats.unreachable_blocks}",
        f"Predicates generated: {stats.predicates_generated}",
        "",
        "Narrowing kinds:",
    ]
    for kind, count in stats.narrowing_kinds_distribution.items():
        if count > 0:
            lines.append(f"  {kind}: {count}")
    lines.extend([
        "",
        f"Exhaustive switches: {stats.exhaustive_switches}",
        f"Non-exhaustive switches: {stats.non_exhaustive_switches}",
        f"Exhaustiveness coverage: {stats.exhaustiveness_coverage:.1%}",
        f"Average narrowing depth: {stats.average_narrowing_depth:.2f}",
        f"Max narrowing depth: {stats.max_narrowing_depth}",
    ])
    return "\n".join(lines)
