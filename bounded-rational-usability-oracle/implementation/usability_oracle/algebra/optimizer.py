"""
usability_oracle.algebra.optimizer — Algebraic simplification of cost
expression trees.

The :class:`AlgebraicOptimizer` applies rewriting rules to a
:class:`CostExpression` tree to:

* **Flatten** nested associative operators: ``(a ⊕ b) ⊕ c → a ⊕ b ⊕ c``
* **Eliminate zero** identity elements: ``a ⊕ 0 → a``
* **Factor** common sub-expressions
* **Reorder** commutative operators for canonical form
* **Estimate** evaluation cost for expression scheduling
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from usability_oracle.algebra.models import (
    CostElement,
    CostExpression,
    Leaf,
    Sequential,
    Parallel,
    ContextMod,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_zero(expr: CostExpression) -> bool:
    """Check if an expression evaluates to the zero element."""
    if isinstance(expr, Leaf):
        e = expr.element
        return (
            abs(e.mu) < 1e-15
            and abs(e.sigma_sq) < 1e-15
            and abs(e.kappa) < 1e-15
            and abs(e.lambda_) < 1e-15
        )
    return False


def _leaf_sort_key(expr: CostExpression) -> float:
    """Return a sort key for ordering leaves (by μ descending)."""
    if isinstance(expr, Leaf):
        return -expr.element.mu
    return float("-inf")


def _collect_sequential_chain(expr: CostExpression) -> List[CostExpression]:
    """Flatten a left- or right-leaning sequential chain into a list."""
    if isinstance(expr, Sequential):
        left_chain = _collect_sequential_chain(expr.left)
        right_chain = _collect_sequential_chain(expr.right)
        return left_chain + right_chain
    return [expr]


def _collect_parallel_group(expr: CostExpression) -> List[CostExpression]:
    """Flatten a left- or right-leaning parallel tree into a list."""
    if isinstance(expr, Parallel):
        left_group = _collect_parallel_group(expr.left)
        right_group = _collect_parallel_group(expr.right)
        return left_group + right_group
    return [expr]


def _build_sequential_chain(
    elements: List[CostExpression], coupling: float = 0.0
) -> CostExpression:
    """Build a right-associative sequential chain from a list."""
    if not elements:
        return Leaf(CostElement.zero())
    if len(elements) == 1:
        return elements[0]
    result = elements[-1]
    for i in range(len(elements) - 2, -1, -1):
        result = Sequential(left=elements[i], right=result, coupling=coupling)
    return result


def _build_parallel_group(
    elements: List[CostExpression], interference: float = 0.0
) -> CostExpression:
    """Build a balanced parallel tree from a list."""
    if not elements:
        return Leaf(CostElement.zero())
    if len(elements) == 1:
        return elements[0]
    # balanced binary tree
    mid = len(elements) // 2
    left = _build_parallel_group(elements[:mid], interference)
    right = _build_parallel_group(elements[mid:], interference)
    return Parallel(left=left, right=right, interference=interference)


# ---------------------------------------------------------------------------
# AlgebraicOptimizer
# ---------------------------------------------------------------------------


class AlgebraicOptimizer:
    """Simplify :class:`CostExpression` trees via algebraic rewriting.

    Usage::

        opt = AlgebraicOptimizer()
        simplified = opt.optimize(expr)
        cost = opt.estimate_cost(expr)
    """

    def __init__(self, *, max_passes: int = 10) -> None:
        """
        Parameters
        ----------
        max_passes : int
            Maximum number of rewriting passes.  The optimizer stops early
            if a pass produces no changes.
        """
        self._max_passes = max_passes

    # -- main entry point ----------------------------------------------------

    def optimize(self, expr: CostExpression) -> CostExpression:
        """Apply all simplification rules until convergence.

        Parameters
        ----------
        expr : CostExpression
            The expression tree to optimise.

        Returns
        -------
        CostExpression
            The simplified expression tree.
        """
        prev_cost = self.estimate_cost(expr)
        for _ in range(self._max_passes):
            expr = self._flatten_sequential(expr)
            expr = self._flatten_parallel(expr)
            expr = self._eliminate_zero(expr)
            expr = self._factor_common(expr)
            expr = self._reorder_commutative(expr)

            new_cost = self.estimate_cost(expr)
            if new_cost >= prev_cost:
                break
            prev_cost = new_cost

        return expr

    # -- flatten sequential: (a ⊕ b) ⊕ c → [a, b, c] → chain ---------------

    def _flatten_sequential(self, expr: CostExpression) -> CostExpression:
        """Flatten nested sequential compositions.

        Sequential composition is associative (when coupling is uniform):
        ``(a ⊕ b) ⊕ c ≈ a ⊕ (b ⊕ c)``

        We flatten the tree into a list, then rebuild a balanced tree.
        """
        if isinstance(expr, Sequential):
            # Recursively flatten children first
            left = self._flatten_sequential(expr.left)
            right = self._flatten_sequential(expr.right)

            chain = _collect_sequential_chain(
                Sequential(left=left, right=right, coupling=expr.coupling)
            )

            if len(chain) > 2:
                return _build_sequential_chain(chain, coupling=expr.coupling)
            return Sequential(left=left, right=right, coupling=expr.coupling)

        elif isinstance(expr, Parallel):
            return Parallel(
                left=self._flatten_sequential(expr.left),
                right=self._flatten_sequential(expr.right),
                interference=expr.interference,
            )
        elif isinstance(expr, ContextMod):
            return ContextMod(
                expr=self._flatten_sequential(expr.expr),
                context=expr.context,
            )
        return expr

    # -- flatten parallel: (a ⊗ b) ⊗ c → balanced tree ----------------------

    def _flatten_parallel(self, expr: CostExpression) -> CostExpression:
        """Flatten nested parallel compositions into a balanced tree."""
        if isinstance(expr, Parallel):
            left = self._flatten_parallel(expr.left)
            right = self._flatten_parallel(expr.right)

            group = _collect_parallel_group(
                Parallel(left=left, right=right, interference=expr.interference)
            )

            if len(group) > 2:
                return _build_parallel_group(group, interference=expr.interference)
            return Parallel(left=left, right=right, interference=expr.interference)

        elif isinstance(expr, Sequential):
            return Sequential(
                left=self._flatten_parallel(expr.left),
                right=self._flatten_parallel(expr.right),
                coupling=expr.coupling,
            )
        elif isinstance(expr, ContextMod):
            return ContextMod(
                expr=self._flatten_parallel(expr.expr),
                context=expr.context,
            )
        return expr

    # -- eliminate zero: a ⊕ 0 → a ------------------------------------------

    def _eliminate_zero(self, expr: CostExpression) -> CostExpression:
        """Remove zero-cost elements from compositions.

        * ``a ⊕ 0 → a`` and ``0 ⊕ a → a``
        * ``a ⊗ 0 → a`` and ``0 ⊗ a → a``
        """
        if isinstance(expr, Sequential):
            left = self._eliminate_zero(expr.left)
            right = self._eliminate_zero(expr.right)
            if _is_zero(left):
                return right
            if _is_zero(right):
                return left
            return Sequential(left=left, right=right, coupling=expr.coupling)

        elif isinstance(expr, Parallel):
            left = self._eliminate_zero(expr.left)
            right = self._eliminate_zero(expr.right)
            if _is_zero(left):
                return right
            if _is_zero(right):
                return left
            return Parallel(left=left, right=right, interference=expr.interference)

        elif isinstance(expr, ContextMod):
            inner = self._eliminate_zero(expr.expr)
            if _is_zero(inner):
                return inner
            return ContextMod(expr=inner, context=expr.context)

        return expr

    # -- factor common sub-expressions ---------------------------------------

    def _factor_common(self, expr: CostExpression) -> CostExpression:
        """Factor common sub-expressions in parallel groups.

        If the same leaf appears multiple times in a parallel group,
        replace with a single instance (since doing the same thing
        concurrently with itself is redundant — idempotency).
        """
        if isinstance(expr, Parallel):
            left = self._factor_common(expr.left)
            right = self._factor_common(expr.right)

            group = _collect_parallel_group(
                Parallel(left=left, right=right, interference=expr.interference)
            )

            # Deduplicate leaves with identical cost elements
            seen: Dict[int, CostExpression] = {}
            unique: List[CostExpression] = []
            for item in group:
                if isinstance(item, Leaf):
                    key = hash(item.element)
                    if key not in seen:
                        seen[key] = item
                        unique.append(item)
                else:
                    unique.append(item)

            if len(unique) < len(group):
                return _build_parallel_group(unique, interference=expr.interference)
            return Parallel(left=left, right=right, interference=expr.interference)

        elif isinstance(expr, Sequential):
            return Sequential(
                left=self._factor_common(expr.left),
                right=self._factor_common(expr.right),
                coupling=expr.coupling,
            )
        elif isinstance(expr, ContextMod):
            return ContextMod(
                expr=self._factor_common(expr.expr),
                context=expr.context,
            )
        return expr

    # -- reorder commutative operators ---------------------------------------

    def _reorder_commutative(self, expr: CostExpression) -> CostExpression:
        """Reorder parallel groups into a canonical form (by descending μ).

        Since ⊗ is commutative, ``a ⊗ b = b ⊗ a``.  Canonical ordering
        makes expressions easier to compare and cache.
        """
        if isinstance(expr, Parallel):
            left = self._reorder_commutative(expr.left)
            right = self._reorder_commutative(expr.right)

            group = _collect_parallel_group(
                Parallel(left=left, right=right, interference=expr.interference)
            )

            # Sort by descending mean cost
            group.sort(key=_leaf_sort_key)

            if len(group) >= 2:
                return _build_parallel_group(group, interference=expr.interference)
            return Parallel(left=left, right=right, interference=expr.interference)

        elif isinstance(expr, Sequential):
            return Sequential(
                left=self._reorder_commutative(expr.left),
                right=self._reorder_commutative(expr.right),
                coupling=expr.coupling,
            )
        elif isinstance(expr, ContextMod):
            return ContextMod(
                expr=self._reorder_commutative(expr.expr),
                context=expr.context,
            )
        return expr

    # -- cost estimation -----------------------------------------------------

    def estimate_cost(self, expr: CostExpression) -> int:
        """Estimate the computational cost of evaluating an expression.

        Each node has a unit cost; the total is the number of nodes.
        This can be used to compare optimization strategies.

        Parameters
        ----------
        expr : CostExpression

        Returns
        -------
        int
            Estimated evaluation cost (node count).
        """
        return expr.node_count()

    # -- expression comparison -----------------------------------------------

    def structurally_equal(
        self, a: CostExpression, b: CostExpression, tolerance: float = 1e-10
    ) -> bool:
        """Check if two expressions are structurally equivalent.

        Two expressions are structurally equal if they have the same tree
        shape and all leaf elements are equal within *tolerance*.
        """
        if type(a) != type(b):
            return False

        if isinstance(a, Leaf) and isinstance(b, Leaf):
            return (
                math.isclose(a.element.mu, b.element.mu, abs_tol=tolerance)
                and math.isclose(a.element.sigma_sq, b.element.sigma_sq, abs_tol=tolerance)
                and math.isclose(a.element.kappa, b.element.kappa, abs_tol=tolerance)
                and math.isclose(a.element.lambda_, b.element.lambda_, abs_tol=tolerance)
            )

        if isinstance(a, Sequential) and isinstance(b, Sequential):
            return (
                math.isclose(a.coupling, b.coupling, abs_tol=tolerance)
                and self.structurally_equal(a.left, b.left, tolerance)
                and self.structurally_equal(a.right, b.right, tolerance)
            )

        if isinstance(a, Parallel) and isinstance(b, Parallel):
            return (
                math.isclose(a.interference, b.interference, abs_tol=tolerance)
                and self.structurally_equal(a.left, b.left, tolerance)
                and self.structurally_equal(a.right, b.right, tolerance)
            )

        if isinstance(a, ContextMod) and isinstance(b, ContextMod):
            return (
                a.context == b.context
                and self.structurally_equal(a.expr, b.expr, tolerance)
            )

        return False

    # -- pretty print --------------------------------------------------------

    @staticmethod
    def pretty_print(expr: CostExpression, indent: int = 0) -> str:
        """Return a human-readable string representation of the tree."""
        prefix = "  " * indent
        if isinstance(expr, Leaf):
            e = expr.element
            return f"{prefix}Leaf(μ={e.mu:.3f}, σ²={e.sigma_sq:.3f}, κ={e.kappa:.3f}, λ={e.lambda_:.3f})"

        if isinstance(expr, Sequential):
            lines = [f"{prefix}Sequential(ρ={expr.coupling:.2f})"]
            lines.append(AlgebraicOptimizer.pretty_print(expr.left, indent + 1))
            lines.append(AlgebraicOptimizer.pretty_print(expr.right, indent + 1))
            return "\n".join(lines)

        if isinstance(expr, Parallel):
            lines = [f"{prefix}Parallel(η={expr.interference:.2f})"]
            lines.append(AlgebraicOptimizer.pretty_print(expr.left, indent + 1))
            lines.append(AlgebraicOptimizer.pretty_print(expr.right, indent + 1))
            return "\n".join(lines)

        if isinstance(expr, ContextMod):
            lines = [f"{prefix}ContextMod({expr.context})"]
            lines.append(AlgebraicOptimizer.pretty_print(expr.expr, indent + 1))
            return "\n".join(lines)

        return f"{prefix}<unknown expression>"
