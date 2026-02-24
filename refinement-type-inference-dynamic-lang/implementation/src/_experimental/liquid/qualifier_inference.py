"""
Automatic qualifier discovery for liquid type inference.

Provides qualifier mining from program AST guards, parameterized qualifier
templates, template instantiation, qualifier minimization, and a database
for indexing / serializing qualifier sets.
"""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Local type stubs – standalone, no cross-module imports
# ---------------------------------------------------------------------------

class _NodeKind(Enum):
    VAR = auto()
    CONST = auto()
    BINOP = auto()
    UNARYOP = auto()
    CALL = auto()
    IF = auto()
    WHILE = auto()
    ASSERT = auto()
    RETURN = auto()
    ASSIGN = auto()
    SEQ = auto()
    FUN = auto()
    ANNOT = auto()


@dataclass
class ASTNode:
    """Minimal AST node used for qualifier mining."""
    kind: _NodeKind
    name: Optional[str] = None
    op: Optional[str] = None
    value: Any = None
    children: List["ASTNode"] = field(default_factory=list)
    annotation: Optional[str] = None

    def walk(self) -> Iterable["ASTNode"]:
        """Depth-first traversal yielding every node."""
        yield self
        for child in self.children:
            yield from child.walk()


# ===================================================================
# 1. Qualifier
# ===================================================================

@dataclass(frozen=True)
class Qualifier:
    """
    A concrete qualifier: a predicate over a *subject* variable.

    Examples: ``x > 0``, ``x != null``, ``len(x) > k``.

    *expr_str* is the canonical string form (used for identity / hashing).
    *variables* lists the free variables that appear in the predicate.
    *predicate_shape* is a normalised pattern (e.g. ``_ > 0``) used for
    indexing.
    """
    expr_str: str
    variables: Tuple[str, ...]
    base_type: str = "any"
    predicate_shape: str = ""

    @staticmethod
    def from_parts(lhs: str, op: str, rhs: str, *, base_type: str = "any") -> "Qualifier":
        expr = f"{lhs} {op} {rhs}"
        shape = f"_ {op} {rhs}"
        variables = tuple(v for v in (lhs,) if not _is_literal(v))
        return Qualifier(
            expr_str=expr,
            variables=variables,
            base_type=base_type,
            predicate_shape=shape,
        )

    def mentions(self, var: str) -> bool:
        return var in self.variables

    def rename(self, old: str, new: str) -> "Qualifier":
        new_vars = tuple(new if v == old else v for v in self.variables)
        new_expr = self.expr_str.replace(old, new)
        return Qualifier(
            expr_str=new_expr,
            variables=new_vars,
            base_type=self.base_type,
            predicate_shape=self.predicate_shape,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expr": self.expr_str,
            "variables": list(self.variables),
            "base_type": self.base_type,
            "shape": self.predicate_shape,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Qualifier":
        return Qualifier(
            expr_str=d["expr"],
            variables=tuple(d.get("variables", [])),
            base_type=d.get("base_type", "any"),
            predicate_shape=d.get("shape", ""),
        )

    def __repr__(self) -> str:
        return self.expr_str


def _is_literal(s: str) -> bool:
    """Heuristic: is *s* a literal constant rather than a variable name?"""
    if s in ("true", "false", "null", "None"):
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


# ===================================================================
# 2. Qualifier template
# ===================================================================

@dataclass(frozen=True)
class QualifierTemplate:
    """
    A parameterized qualifier template with holes.

    ``_`` stands for the subject variable and ``?`` for an unknown constant
    that will be instantiated.  Examples: ``_ > ?``, ``_ != ?``, ``len(_) >= ?``.
    """
    pattern: str  # e.g. "_ > ?"
    operator: str  # e.g. ">"
    arity: int = 1  # number of ``?`` holes
    applicable_types: Tuple[str, ...] = ("int",)

    def instantiate(self, var: str, constants: Tuple[Any, ...]) -> Qualifier:
        """Produce a concrete :class:`Qualifier` by filling holes."""
        expr = self.pattern.replace("_", var, 1)
        for c in constants:
            expr = expr.replace("?", str(c), 1)
        shape = self.pattern
        for c in constants:
            shape = shape.replace("?", str(c), 1)
        variables = (var,)
        return Qualifier(
            expr_str=expr,
            variables=variables,
            base_type=self.applicable_types[0] if self.applicable_types else "any",
            predicate_shape=shape,
        )

    def __repr__(self) -> str:
        return f"Template({self.pattern})"


# A library of common templates
BUILTIN_TEMPLATES: List[QualifierTemplate] = [
    QualifierTemplate(pattern="_ > ?", operator=">", applicable_types=("int",)),
    QualifierTemplate(pattern="_ >= ?", operator=">=", applicable_types=("int",)),
    QualifierTemplate(pattern="_ < ?", operator="<", applicable_types=("int",)),
    QualifierTemplate(pattern="_ <= ?", operator="<=", applicable_types=("int",)),
    QualifierTemplate(pattern="_ == ?", operator="==", applicable_types=("int", "string", "bool")),
    QualifierTemplate(pattern="_ != ?", operator="!=", applicable_types=("int", "string", "bool")),
    QualifierTemplate(pattern="len(_) > ?", operator=">", applicable_types=("string", "list")),
    QualifierTemplate(pattern="len(_) >= ?", operator=">=", applicable_types=("string", "list")),
    QualifierTemplate(pattern="len(_) == ?", operator="==", applicable_types=("string", "list")),
    QualifierTemplate(pattern="_ != null", operator="!=", arity=0, applicable_types=("any",)),
    QualifierTemplate(pattern="_ > 0", operator=">", arity=0, applicable_types=("int",)),
    QualifierTemplate(pattern="_ >= 0", operator=">=", arity=0, applicable_types=("int",)),
]


# ===================================================================
# 3. Qualifier miner
# ===================================================================

class QualifierMiner:
    """
    Extracts qualifiers from program source by inspecting guards
    (if-conditions, assert-conditions, while-conditions) and annotations.
    """

    # comparison operators we recognise
    _COMPARISON_OPS: FrozenSet[str] = frozenset({"==", "!=", "<", "<=", ">", ">="})
    _BOOL_OPS: FrozenSet[str] = frozenset({"and", "or", "not"})

    def __init__(self) -> None:
        self._qualifiers: List[Qualifier] = []

    def qualifiers(self) -> List[Qualifier]:
        return list(self._qualifiers)

    # -- public API -------------------------------------------------------

    def mine_from_ast(self, root: ASTNode) -> List[Qualifier]:
        """Walk the full AST, extracting qualifiers from guard positions."""
        for node in root.walk():
            if node.kind == _NodeKind.IF and node.children:
                self._extract_from_expr(node.children[0])
            elif node.kind == _NodeKind.WHILE and node.children:
                self._extract_from_expr(node.children[0])
            elif node.kind == _NodeKind.ASSERT and node.children:
                self._extract_from_expr(node.children[0])
            elif node.kind == _NodeKind.ANNOT and node.annotation:
                self._extract_from_annotation_str(node.annotation)
        return list(self._qualifiers)

    def mine_from_guards(self, guards: List[ASTNode]) -> List[Qualifier]:
        """Extract qualifiers from a pre-collected list of guard expressions."""
        for g in guards:
            self._extract_from_expr(g)
        return list(self._qualifiers)

    def mine_from_annotations(self, annotations: List[str]) -> List[Qualifier]:
        """Parse annotation strings like ``x > 0`` into qualifiers."""
        for a in annotations:
            self._extract_from_annotation_str(a)
        return list(self._qualifiers)

    # -- internal extraction ----------------------------------------------

    def _extract_from_expr(self, node: ASTNode) -> None:
        """Recursively break apart a boolean expression into atomic qualifiers."""
        if node.kind == _NodeKind.BINOP and node.op in self._COMPARISON_OPS:
            self._record_binop(node)
        elif node.kind == _NodeKind.BINOP and node.op in self._BOOL_OPS:
            for child in node.children:
                self._extract_from_expr(child)
        elif node.kind == _NodeKind.UNARYOP and node.op == "not":
            if node.children:
                self._extract_from_expr(node.children[0])
        elif node.kind == _NodeKind.CALL:
            # e.g. len(x) > 0 might appear as BINOP(CALL(len, x), >, 0)
            pass
        elif node.kind == _NodeKind.VAR:
            # bare variable in guard position – treat as  ``x != false``
            if node.name:
                q = Qualifier.from_parts(node.name, "!=", "false", base_type="bool")
                self._add(q)

    def _record_binop(self, node: ASTNode) -> None:
        """Record a comparison ``lhs op rhs`` as a qualifier."""
        assert node.op is not None
        lhs_str = self._node_to_str(node.children[0]) if node.children else "?"
        rhs_str = self._node_to_str(node.children[1]) if len(node.children) > 1 else "?"

        q = Qualifier.from_parts(lhs_str, node.op, rhs_str)
        self._add(q)

        # also record the flipped version when meaningful
        flipped_op = _flip_op(node.op)
        if flipped_op:
            q2 = Qualifier.from_parts(rhs_str, flipped_op, lhs_str)
            self._add(q2)

    def _extract_from_annotation_str(self, text: str) -> None:
        """Attempt to parse simple ``var op const`` annotations."""
        text = text.strip()
        for op in sorted(self._COMPARISON_OPS, key=len, reverse=True):
            if f" {op} " in text:
                parts = text.split(f" {op} ", 1)
                if len(parts) == 2:
                    q = Qualifier.from_parts(parts[0].strip(), op, parts[1].strip())
                    self._add(q)
                    return

    def _node_to_str(self, node: ASTNode) -> str:
        """Best-effort conversion of an AST node to a string token."""
        if node.kind == _NodeKind.VAR and node.name:
            return node.name
        if node.kind == _NodeKind.CONST:
            return str(node.value)
        if node.kind == _NodeKind.CALL and node.name and node.children:
            inner = self._node_to_str(node.children[0])
            return f"{node.name}({inner})"
        return "?"

    def _add(self, q: Qualifier) -> None:
        if q not in self._qualifiers:
            self._qualifiers.append(q)


def _flip_op(op: str) -> Optional[str]:
    _map = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
    return _map.get(op)


# ===================================================================
# 4. Template instantiator
# ===================================================================

class TemplateInstantiator:
    """
    Given a set of :class:`QualifierTemplate` instances and program
    variables (with their types), generate concrete qualifier instances.
    """

    def __init__(
        self,
        templates: Optional[List[QualifierTemplate]] = None,
        constants: Optional[List[Any]] = None,
    ) -> None:
        self._templates = templates if templates is not None else list(BUILTIN_TEMPLATES)
        self._constants = constants if constants is not None else [0, 1, -1, 2]

    def instantiate(
        self,
        variables: Dict[str, str],
    ) -> List[Qualifier]:
        """
        Instantiate all applicable templates for the given *variables*.

        Parameters
        ----------
        variables : dict
            Mapping from variable name to its base type string
            (e.g. ``{"x": "int", "s": "string"}``).

        Returns
        -------
        list of Qualifier
        """
        results: List[Qualifier] = []
        seen: Set[str] = set()
        for var_name, var_type in variables.items():
            for tmpl in self._templates:
                if not self._type_applicable(var_type, tmpl.applicable_types):
                    continue
                if tmpl.arity == 0:
                    q = tmpl.instantiate(var_name, ())
                    if q.expr_str not in seen:
                        seen.add(q.expr_str)
                        results.append(q)
                else:
                    for combo in itertools.product(self._constants, repeat=tmpl.arity):
                        q = tmpl.instantiate(var_name, combo)
                        if q.expr_str not in seen:
                            seen.add(q.expr_str)
                            results.append(q)
        return results

    def instantiate_with_extra_constants(
        self,
        variables: Dict[str, str],
        extra_constants: List[Any],
    ) -> List[Qualifier]:
        """Like :meth:`instantiate` but with additional constant values."""
        old_constants = self._constants
        self._constants = list(set(old_constants + extra_constants))
        result = self.instantiate(variables)
        self._constants = old_constants
        return result

    @staticmethod
    def _type_applicable(var_type: str, applicable: Tuple[str, ...]) -> bool:
        if "any" in applicable:
            return True
        return var_type in applicable


# ===================================================================
# 5. Qualifier minimizer
# ===================================================================

class QualifierMinimizer:
    """
    Find the smallest set of qualifiers sufficient for verification.

    Two strategies:

    1. **Iterative removal** – try removing each qualifier one at a time,
       re-check validity, and keep the qualifier only if removal breaks it.
    2. **Greedy set cover** – model the problem as a set cover instance
       where each qualifier "covers" a set of constraints.
    """

    def __init__(self, checker: Optional[Callable[[List[Qualifier]], bool]] = None) -> None:
        self._checker = checker or (lambda qs: True)

    def set_checker(self, checker: Callable[[List[Qualifier]], bool]) -> None:
        self._checker = checker

    # -- iterative removal ------------------------------------------------

    def minimize_iterative(self, qualifiers: List[Qualifier]) -> List[Qualifier]:
        """Remove qualifiers one by one; keep only those that are necessary.

        The checker function should return True when the given qualifier set
        is sufficient.
        """
        current = list(qualifiers)
        if not self._checker(current):
            return current  # already invalid – cannot minimize

        i = 0
        while i < len(current):
            candidate = current[:i] + current[i + 1:]
            if self._checker(candidate):
                current = candidate
                # don't increment – the next qualifier slid into position i
            else:
                i += 1
        return current

    # -- greedy set cover -------------------------------------------------

    def minimize_greedy(
        self,
        qualifiers: List[Qualifier],
        constraints: List[str],
        covers: Dict[str, Set[str]],
    ) -> List[Qualifier]:
        """
        Greedy set-cover minimization.

        Parameters
        ----------
        qualifiers : list
            Full qualifier pool.
        constraints : list
            Constraint identifiers that must all be covered.
        covers : dict
            Maps qualifier ``expr_str`` → set of constraint ids it covers.

        Returns
        -------
        list of Qualifier
            A (greedily) minimal subset.
        """
        uncovered: Set[str] = set(constraints)
        selected: List[Qualifier] = []
        remaining = list(qualifiers)

        while uncovered and remaining:
            # pick qualifier covering the most uncovered constraints
            best: Optional[Qualifier] = None
            best_count = -1
            for q in remaining:
                cov = covers.get(q.expr_str, set()) & uncovered
                if len(cov) > best_count:
                    best_count = len(cov)
                    best = q
            if best is None or best_count == 0:
                break
            selected.append(best)
            uncovered -= covers.get(best.expr_str, set())
            remaining.remove(best)

        return selected

    # -- combined strategy ------------------------------------------------

    def minimize(
        self,
        qualifiers: List[Qualifier],
        constraints: Optional[List[str]] = None,
        covers: Optional[Dict[str, Set[str]]] = None,
    ) -> List[Qualifier]:
        """
        Minimize using the best available strategy.

        If *constraints* and *covers* are provided, try greedy set cover
        first and then refine with iterative removal.  Otherwise fall back
        to iterative removal alone.
        """
        if constraints is not None and covers is not None:
            stage1 = self.minimize_greedy(qualifiers, constraints, covers)
            return self.minimize_iterative(stage1)
        return self.minimize_iterative(qualifiers)


# ===================================================================
# 6. Qualifier database
# ===================================================================

class QualifierDatabase:
    """
    Stores and indexes qualifiers.

    Supports lookup by type, by variable, by predicate shape.
    Also supports merge, diff, and JSON serialization.
    """

    def __init__(self) -> None:
        self._by_expr: Dict[str, Qualifier] = {}
        self._by_type: Dict[str, List[Qualifier]] = {}
        self._by_variable: Dict[str, List[Qualifier]] = {}
        self._by_shape: Dict[str, List[Qualifier]] = {}

    # -- mutation ---------------------------------------------------------

    def add(self, q: Qualifier) -> bool:
        """Add a qualifier.  Returns True if it was new."""
        if q.expr_str in self._by_expr:
            return False
        self._by_expr[q.expr_str] = q
        self._by_type.setdefault(q.base_type, []).append(q)
        for v in q.variables:
            self._by_variable.setdefault(v, []).append(q)
        if q.predicate_shape:
            self._by_shape.setdefault(q.predicate_shape, []).append(q)
        return True

    def add_all(self, qs: Iterable[Qualifier]) -> int:
        """Add multiple qualifiers, returning the count of new ones."""
        return sum(1 for q in qs if self.add(q))

    def remove(self, expr_str: str) -> Optional[Qualifier]:
        """Remove a qualifier by its expression string."""
        q = self._by_expr.pop(expr_str, None)
        if q is None:
            return None
        self._remove_from_index(self._by_type, q.base_type, q)
        for v in q.variables:
            self._remove_from_index(self._by_variable, v, q)
        if q.predicate_shape:
            self._remove_from_index(self._by_shape, q.predicate_shape, q)
        return q

    @staticmethod
    def _remove_from_index(index: Dict[str, List[Qualifier]], key: str, q: Qualifier) -> None:
        lst = index.get(key)
        if lst is not None:
            try:
                lst.remove(q)
            except ValueError:
                pass
            if not lst:
                del index[key]

    # -- queries ----------------------------------------------------------

    def lookup(self, expr_str: str) -> Optional[Qualifier]:
        return self._by_expr.get(expr_str)

    def by_type(self, base_type: str) -> List[Qualifier]:
        return list(self._by_type.get(base_type, []))

    def by_variable(self, var: str) -> List[Qualifier]:
        return list(self._by_variable.get(var, []))

    def by_shape(self, shape: str) -> List[Qualifier]:
        return list(self._by_shape.get(shape, []))

    def all_qualifiers(self) -> List[Qualifier]:
        return list(self._by_expr.values())

    def all_shapes(self) -> Set[str]:
        return set(self._by_shape.keys())

    def all_variables(self) -> Set[str]:
        return set(self._by_variable.keys())

    def size(self) -> int:
        return len(self._by_expr)

    def contains(self, expr_str: str) -> bool:
        return expr_str in self._by_expr

    # -- set operations ---------------------------------------------------

    def merge(self, other: "QualifierDatabase") -> int:
        """Merge qualifiers from *other* into this database. Returns count of new ones."""
        return self.add_all(other.all_qualifiers())

    def diff(self, other: "QualifierDatabase") -> List[Qualifier]:
        """Return qualifiers in *self* but not in *other*."""
        return [q for q in self._by_expr.values() if not other.contains(q.expr_str)]

    def intersection(self, other: "QualifierDatabase") -> List[Qualifier]:
        """Return qualifiers present in both databases."""
        return [q for q in self._by_expr.values() if other.contains(q.expr_str)]

    # -- serialization ----------------------------------------------------

    def to_json(self) -> str:
        """Serialize the database to a JSON string."""
        data = [q.to_dict() for q in self._by_expr.values()]
        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(text: str) -> "QualifierDatabase":
        """Deserialize a database from a JSON string."""
        data = json.loads(text)
        db = QualifierDatabase()
        for item in data:
            db.add(Qualifier.from_dict(item))
        return db

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [q.to_dict() for q in self._by_expr.values()]

    # -- summary ----------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary of the database contents."""
        return {
            "total": self.size(),
            "by_type": {t: len(qs) for t, qs in self._by_type.items()},
            "shapes": len(self._by_shape),
            "variables": len(self._by_variable),
        }

    def __repr__(self) -> str:
        return f"QualifierDatabase(size={self.size()})"

    def __len__(self) -> int:
        return self.size()

    def __contains__(self, expr_str: str) -> bool:  # type: ignore[override]
        return self.contains(expr_str)
