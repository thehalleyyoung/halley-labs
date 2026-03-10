"""
usability_oracle.algebra.operad — Operadic composition for cognitive tasks.

An **operad** is an algebraic structure that captures *multi-input
composition*: an operation with ``n`` input slots can have sub-operations
plugged into each slot.  This is the natural language for describing
**task decomposition** — a complex UI task is broken into sub-tasks,
each of which may be further decomposed.

Mathematical Structure
----------------------
An operad ``O`` consists of:

* A collection of **operations** ``O(n)`` for each arity ``n ≥ 0``.
* A **composition** ``γ : O(n) × O(k₁) × … × O(kₙ) → O(k₁ + … + kₙ)``
  that plugs sub-operations into slots.
* An **identity** ``id ∈ O(1)``.
* **Associativity** and **unitality** axioms.

Colored Operad Extension
~~~~~~~~~~~~~~~~~~~~~~~~
A **colored operad** (a.k.a. *symmetric multicategory*) assigns a
*color* (type) to each input/output slot, enforcing that only
type-compatible operations can be composed.

Application
~~~~~~~~~~~
* **Task decomposition** maps to operadic composition: decomposing a
  ``click-and-fill`` task into ``navigate``, ``click``, ``type`` sub-tasks
  is an operadic composition.
* **Cost propagation** is an operad algebra: the evaluation functor maps
  operadic terms to cost elements.
* **Cost bounds** compose operadically, yielding sound bounds on the
  decomposed task.

References
----------
* Leinster, *Higher Operads, Higher Categories*, Cambridge, 2004.
* Yau, *Colored Operads*, AMS Graduate Studies, 2016.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np

from usability_oracle.algebra.models import CostElement
from usability_oracle.algebra.sequential import SequentialComposer
from usability_oracle.algebra.parallel import ParallelComposer

# ---------------------------------------------------------------------------
# Operation (element of the operad)
# ---------------------------------------------------------------------------


@dataclass
class Operation:
    """An element of the cognitive operad — a typed multi-input operation.

    Parameters
    ----------
    name : str
        Human-readable name (e.g., ``"click"``, ``"navigate"``).
    arity : int
        Number of input slots (sub-tasks to be filled).
    input_colors : tuple[str, ...]
        Types of each input slot.
    output_color : str
        Type of the output.
    cost : CostElement
        Base cognitive cost of this operation (before sub-task costs).
    """

    name: str
    arity: int
    input_colors: Tuple[str, ...] = ()
    output_color: str = "task"
    cost: CostElement = field(default_factory=CostElement.zero)

    def __post_init__(self) -> None:
        if len(self.input_colors) != self.arity:
            if self.arity > 0 and not self.input_colors:
                object.__setattr__(
                    self, "input_colors", ("task",) * self.arity
                )
            elif len(self.input_colors) != self.arity:
                raise ValueError(
                    f"input_colors length {len(self.input_colors)} != arity {self.arity}"
                )

    def is_identity(self) -> bool:
        """True if this is the operadic identity (arity 1, zero cost)."""
        return self.arity == 1 and abs(self.cost.mu) < 1e-15

    def __repr__(self) -> str:
        colors = ",".join(self.input_colors)
        return f"Op({self.name}:{colors}→{self.output_color}, μ={self.cost.mu:.4f})"


# ---------------------------------------------------------------------------
# Operadic term tree
# ---------------------------------------------------------------------------


@dataclass
class OperadTerm:
    """A tree representing a composite operadic term.

    Each node carries an :class:`Operation`; its children (sub-terms)
    correspond to the operation's input slots.

    Parameters
    ----------
    operation : Operation
        The operation at this node.
    children : list[OperadTerm]
        Sub-terms plugged into the input slots.
    """

    operation: Operation
    children: List["OperadTerm"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.children) != self.operation.arity:
            if self.operation.arity == 0 and not self.children:
                pass  # leaf
            elif len(self.children) != self.operation.arity:
                raise ValueError(
                    f"Operation {self.operation.name!r} has arity "
                    f"{self.operation.arity} but got {len(self.children)} children."
                )

    @property
    def arity(self) -> int:
        """Effective arity: number of unfilled leaf slots."""
        if self.operation.arity == 0:
            return 0
        return sum(c.arity for c in self.children) if self.children else self.operation.arity

    def depth(self) -> int:
        """Depth of the term tree."""
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def node_count(self) -> int:
        """Total number of nodes in the term tree."""
        return 1 + sum(c.node_count() for c in self.children)

    def leaves(self) -> List[Operation]:
        """Collect leaf operations (arity 0)."""
        if not self.children:
            return [self.operation]
        result: List[Operation] = []
        for c in self.children:
            result.extend(c.leaves())
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the term tree."""
        return {
            "operation": self.operation.name,
            "cost": self.operation.cost.to_dict(),
            "children": [c.to_dict() for c in self.children],
        }

    def __repr__(self) -> str:
        if not self.children:
            return self.operation.name
        args = ", ".join(repr(c) for c in self.children)
        return f"{self.operation.name}({args})"


# ---------------------------------------------------------------------------
# CognitiveOperad
# ---------------------------------------------------------------------------


class CognitiveOperad:
    r"""An operad of cognitive operations with cost propagation.

    Implements the composition law and the evaluation algebra that maps
    operadic terms to :class:`CostElement` values.

    Parameters
    ----------
    composition_mode : str
        How sub-task costs combine with the parent: ``"sequential"`` or
        ``"parallel"``.  Default ``"sequential"`` (sub-tasks done in order).
    coupling : float
        Coupling parameter for sequential composition.
    interference : float
        Interference parameter for parallel composition.
    """

    def __init__(
        self,
        composition_mode: str = "sequential",
        coupling: float = 0.0,
        interference: float = 0.0,
    ) -> None:
        self._mode = composition_mode
        self._seq = SequentialComposer()
        self._par = ParallelComposer()
        self._coupling = coupling
        self._interference = interference
        self._operations: Dict[str, Operation] = {}

    def register(self, op: Operation) -> None:
        """Register an operation in this operad."""
        self._operations[op.name] = op

    def get(self, name: str) -> Operation:
        """Retrieve a registered operation by name."""
        return self._operations[name]

    def identity(self, color: str = "task") -> Operation:
        """The operadic identity ``id ∈ O(1)``."""
        return Operation(name="id", arity=1, input_colors=(color,),
                         output_color=color, cost=CostElement.zero())

    # -- operadic composition ------------------------------------------------

    def compose(
        self,
        outer: Operation,
        inners: List[Operation],
    ) -> Operation:
        r"""Operadic composition: plug ``inners`` into ``outer``'s slots.

        Given ``outer ∈ O(n)`` and ``inner_i ∈ O(k_i)``, the result is
        in ``O(k_1 + … + k_n)`` with combined cost.

        Enforces **color matching**: ``inner_i.output_color`` must equal
        ``outer.input_colors[i]``.

        Parameters
        ----------
        outer : Operation
            The outer operation with ``n`` slots.
        inners : list[Operation]
            One inner operation per slot.

        Returns
        -------
        Operation
            The composed operation.

        Raises
        ------
        ValueError
            If arity or color mismatch.
        """
        if len(inners) != outer.arity:
            raise ValueError(
                f"Outer arity {outer.arity} != {len(inners)} inner operations."
            )

        # Color checking
        for i, inner in enumerate(inners):
            if inner.output_color != outer.input_colors[i]:
                raise ValueError(
                    f"Color mismatch at slot {i}: inner output "
                    f"{inner.output_color!r} != outer input "
                    f"{outer.input_colors[i]!r}."
                )

        # Combine input colors
        new_inputs: List[str] = []
        for inner in inners:
            new_inputs.extend(inner.input_colors)

        # Compute combined cost
        inner_costs = [inner.cost for inner in inners]
        combined_inner = self._combine_costs(inner_costs)
        total_cost = self._seq.compose(outer.cost, combined_inner, coupling=self._coupling)

        new_arity = sum(inner.arity for inner in inners)
        return Operation(
            name=f"{outer.name}[{','.join(i.name for i in inners)}]",
            arity=new_arity,
            input_colors=tuple(new_inputs),
            output_color=outer.output_color,
            cost=total_cost,
        )

    def compose_term(self, term: OperadTerm) -> Operation:
        r"""Recursively compose an operadic term tree to a single operation.

        This is the universal property of operadic composition:
        any term tree evaluates to a single composite operation.
        """
        if not term.children:
            return term.operation

        inner_ops = [self.compose_term(child) for child in term.children]
        return self.compose(term.operation, inner_ops)

    # -- verification of operadic axioms -------------------------------------

    def verify_associativity(
        self,
        outer: Operation,
        middles: List[Operation],
        inners: List[List[Operation]],
        tol: float = 1e-6,
    ) -> bool:
        r"""Verify associativity of operadic composition.

        Checks: ``γ(outer; γ(m₁; i₁…), …, γ(mₙ; iₙ…))
                 ≈ γ(γ(outer; m₁, …, mₙ); i₁…, …, iₙ…)``.
        """
        if len(middles) != outer.arity or len(inners) != len(middles):
            raise ValueError("Arity mismatch for associativity check.")

        # Path 1: compose middles with their inners first, then with outer
        composed_middles = [
            self.compose(middles[i], inners[i]) for i in range(len(middles))
        ]
        path1 = self.compose(outer, composed_middles)

        # Path 2: compose outer with middles first, then with all inners
        outer_mid = self.compose(outer, middles)
        all_inners = [op for group in inners for op in group]
        path2 = self.compose(outer_mid, all_inners)

        return (
            abs(path1.cost.mu - path2.cost.mu) < tol
            and abs(path1.cost.sigma_sq - path2.cost.sigma_sq) < tol
        )

    def verify_unitality(self, op: Operation, tol: float = 1e-10) -> bool:
        r"""Verify unitality: ``γ(op; id, …, id) ≈ op`` and ``γ(id; op) ≈ op``."""
        # Left unitality: compose with identities
        ids = [self.identity(c) for c in op.input_colors]
        composed = self.compose(op, ids)
        left_ok = abs(composed.cost.mu - op.cost.mu) < tol

        # Right unitality: plug op into identity
        id_op = self.identity(op.output_color)
        right_composed = self.compose(id_op, [op])
        right_ok = abs(right_composed.cost.mu - op.cost.mu) < tol

        return left_ok and right_ok

    # -- evaluation algebra (operad → CostElement) ---------------------------

    def evaluate(self, term: OperadTerm) -> CostElement:
        r"""Evaluate an operadic term to a :class:`CostElement`.

        This is the operad algebra morphism ``eval : Terms → CostElement``.
        Recursively evaluates sub-terms and composes costs.
        """
        if not term.children:
            return term.operation.cost

        child_costs = [self.evaluate(child) for child in term.children]
        combined_child = self._combine_costs(child_costs)
        return self._seq.compose(term.operation.cost, combined_child, coupling=self._coupling)

    def evaluate_bounds(
        self,
        term: OperadTerm,
        cost_bounds: Optional[Dict[str, Tuple[CostElement, CostElement]]] = None,
    ) -> Tuple[CostElement, CostElement]:
        r"""Evaluate operadic term with interval cost bounds.

        Parameters
        ----------
        term : OperadTerm
            The term to evaluate.
        cost_bounds : dict[str, (CostElement, CostElement)] | None
            Map from operation name to ``(lower, upper)`` cost bounds.
            If None, uses point estimates.

        Returns
        -------
        (lower, upper) : tuple[CostElement, CostElement]
            Lower and upper cost bounds for the term.
        """
        if cost_bounds is None:
            cost = self.evaluate(term)
            return cost, cost

        if not term.children:
            name = term.operation.name
            if name in cost_bounds:
                return cost_bounds[name]
            return term.operation.cost, term.operation.cost

        child_lowers = []
        child_uppers = []
        for child in term.children:
            lo, hi = self.evaluate_bounds(child, cost_bounds)
            child_lowers.append(lo)
            child_uppers.append(hi)

        combined_lower = self._combine_costs(child_lowers)
        combined_upper = self._combine_costs(child_uppers)

        name = term.operation.name
        if name in cost_bounds:
            op_lo, op_hi = cost_bounds[name]
        else:
            op_lo = op_hi = term.operation.cost

        lower = self._seq.compose(op_lo, combined_lower, coupling=self._coupling)
        upper = self._seq.compose(op_hi, combined_upper, coupling=self._coupling)
        return lower, upper

    # -- task decomposition as operadic structure ----------------------------

    def decompose(
        self,
        task: Operation,
        subtasks: List[Operation],
        mapping: Optional[List[int]] = None,
    ) -> OperadTerm:
        r"""Decompose a task into sub-tasks as an operadic term.

        Parameters
        ----------
        task : Operation
            The high-level task to decompose.
        subtasks : list[Operation]
            The sub-tasks. Must have ``len(subtasks) == task.arity``.
        mapping : list[int] | None
            Permutation of sub-task indices (for reordering).

        Returns
        -------
        OperadTerm
            A term tree representing the decomposition.
        """
        if mapping is not None:
            subtasks = [subtasks[i] for i in mapping]

        children = [OperadTerm(operation=st) for st in subtasks]
        return OperadTerm(operation=task, children=children)

    def hierarchical_decompose(
        self,
        task: Operation,
        decomposition_map: Dict[str, List[Operation]],
    ) -> OperadTerm:
        r"""Recursively decompose a task using a decomposition map.

        Parameters
        ----------
        task : Operation
            The root task.
        decomposition_map : dict[str, list[Operation]]
            Map from operation name to its sub-task list.

        Returns
        -------
        OperadTerm
            A fully expanded term tree.
        """
        if task.name not in decomposition_map or task.arity == 0:
            return OperadTerm(operation=task)

        subtasks = decomposition_map[task.name]
        children = [
            self.hierarchical_decompose(st, decomposition_map)
            for st in subtasks
        ]
        return OperadTerm(operation=task, children=children)

    # -- internal helpers ----------------------------------------------------

    def _combine_costs(self, costs: List[CostElement]) -> CostElement:
        """Combine a list of costs according to the composition mode."""
        if not costs:
            return CostElement.zero()
        if self._mode == "parallel":
            return self._par.compose_group(costs, interference=self._interference)
        # sequential
        return self._seq.compose_chain(costs, couplings=[self._coupling] * max(0, len(costs) - 1))
