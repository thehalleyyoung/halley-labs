"""
Abstract domain implementations for CEGAR-based DP verification.

Provides interval, polyhedral, and zonotope abstract domains with
Galois connections, abstract transformers (join, meet, widen, narrow),
and sound over-approximation of privacy loss computations.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import optimize

from dp_forge.types import (
    AbstractDomainType,
    AbstractValue,
    AdjacencyRelation,
    PrivacyBudget,
)


# ---------------------------------------------------------------------------
# Base abstract domain
# ---------------------------------------------------------------------------


class AbstractDomain(ABC):
    """Base class for abstract domains used in CEGAR verification."""

    @abstractmethod
    def bottom(self, ndim: int) -> AbstractValue:
        """Return the bottom element (empty set)."""
        ...

    @abstractmethod
    def top(self, ndim: int) -> AbstractValue:
        """Return the top element (universal set)."""
        ...

    @abstractmethod
    def join(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Compute the least upper bound (union over-approximation)."""
        ...

    @abstractmethod
    def meet(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Compute the greatest lower bound (intersection)."""
        ...

    @abstractmethod
    def widen(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Widening operator to ensure convergence of fixpoint iteration."""
        ...

    @abstractmethod
    def narrow(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Narrowing operator to improve precision after widening."""
        ...

    @abstractmethod
    def is_bottom(self, a: AbstractValue) -> bool:
        """Check if the abstract value represents the empty set."""
        ...

    @abstractmethod
    def leq(self, a: AbstractValue, b: AbstractValue) -> bool:
        """Check if a is a subset of b in the abstract domain."""
        ...

    @abstractmethod
    def alpha(self, concrete: npt.NDArray[np.float64]) -> AbstractValue:
        """Abstraction function: map concrete values to abstract domain."""
        ...

    @abstractmethod
    def gamma_contains(self, abstract: AbstractValue, point: npt.NDArray[np.float64]) -> bool:
        """Check if a concrete point is in the concretization of the abstract value."""
        ...


# ---------------------------------------------------------------------------
# Interval abstract domain
# ---------------------------------------------------------------------------


class IntervalAbstraction(AbstractDomain):
    """Interval abstract domain for privacy parameters.

    Represents sets of real vectors as axis-aligned boxes [l_i, u_i].
    Efficient but imprecise for relational properties.
    """

    def __init__(self, widen_threshold: float = 1e6) -> None:
        """Initialize interval abstraction.

        Args:
            widen_threshold: Threshold for widening to ±infinity.
        """
        self._widen_threshold = widen_threshold

    @property
    def domain_type(self) -> AbstractDomainType:
        """Return the domain type identifier."""
        return AbstractDomainType.INTERVAL

    def bottom(self, ndim: int) -> AbstractValue:
        """Return bottom element (empty interval)."""
        lower = np.full(ndim, np.inf, dtype=np.float64)
        upper = np.full(ndim, -np.inf, dtype=np.float64)
        return AbstractValue(
            domain_type=AbstractDomainType.INTERVAL,
            lower=lower,
            upper=upper,
        )

    def top(self, ndim: int) -> AbstractValue:
        """Return top element (entire space)."""
        lower = np.full(ndim, -np.inf, dtype=np.float64)
        upper = np.full(ndim, np.inf, dtype=np.float64)
        return AbstractValue(
            domain_type=AbstractDomainType.INTERVAL,
            lower=lower,
            upper=upper,
        )

    def join(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Compute join (componentwise min/max of bounds).

        Args:
            a: First abstract value.
            b: Second abstract value.

        Returns:
            Least upper bound in the interval domain.
        """
        if self.is_bottom(a):
            return copy.deepcopy(b)
        if self.is_bottom(b):
            return copy.deepcopy(a)
        lower = np.minimum(a.lower, b.lower)
        upper = np.maximum(a.upper, b.upper)
        return AbstractValue(
            domain_type=AbstractDomainType.INTERVAL,
            lower=lower,
            upper=upper,
        )

    def meet(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Compute meet (componentwise max/min of bounds).

        Args:
            a: First abstract value.
            b: Second abstract value.

        Returns:
            Greatest lower bound in the interval domain.
        """
        lower = np.maximum(a.lower, b.lower)
        upper = np.minimum(a.upper, b.upper)
        return AbstractValue(
            domain_type=AbstractDomainType.INTERVAL,
            lower=lower,
            upper=upper,
        )

    def widen(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Standard interval widening.

        For each dimension i:
        - If b.lower[i] < a.lower[i], set lower[i] = -inf
        - If b.upper[i] > a.upper[i], set upper[i] = +inf
        Otherwise keep a's bounds.

        Args:
            a: Previous abstract value.
            b: New abstract value.

        Returns:
            Widened abstract value guaranteeing convergence.
        """
        if self.is_bottom(a):
            return copy.deepcopy(b)
        lower = np.where(b.lower < a.lower - 1e-12, -np.inf, a.lower)
        upper = np.where(b.upper > a.upper + 1e-12, np.inf, a.upper)
        return AbstractValue(
            domain_type=AbstractDomainType.INTERVAL,
            lower=lower,
            upper=upper,
        )

    def narrow(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Standard interval narrowing.

        Replaces infinite bounds in a with finite bounds from b.

        Args:
            a: Widened abstract value.
            b: More precise abstract value.

        Returns:
            Narrowed abstract value.
        """
        lower = np.where(np.isinf(a.lower) & (a.lower < 0), b.lower, a.lower)
        upper = np.where(np.isinf(a.upper) & (a.upper > 0), b.upper, a.upper)
        return AbstractValue(
            domain_type=AbstractDomainType.INTERVAL,
            lower=lower,
            upper=upper,
        )

    def is_bottom(self, a: AbstractValue) -> bool:
        """Check if any lower > upper (empty interval)."""
        return bool(np.any(a.lower > a.upper + 1e-12))

    def leq(self, a: AbstractValue, b: AbstractValue) -> bool:
        """Check containment: a ⊆ b iff a.lower >= b.lower and a.upper <= b.upper."""
        if self.is_bottom(a):
            return True
        if self.is_bottom(b):
            return False
        return bool(
            np.all(a.lower >= b.lower - 1e-12)
            and np.all(a.upper <= b.upper + 1e-12)
        )

    def alpha(self, concrete: npt.NDArray[np.float64]) -> AbstractValue:
        """Abstract a set of concrete points to their bounding box.

        Args:
            concrete: Array of shape (m, ndim) with m concrete points.

        Returns:
            Interval abstract value bounding all points.
        """
        concrete = np.atleast_2d(concrete)
        lower = np.min(concrete, axis=0)
        upper = np.max(concrete, axis=0)
        return AbstractValue(
            domain_type=AbstractDomainType.INTERVAL,
            lower=lower,
            upper=upper,
        )

    def gamma_contains(self, abstract: AbstractValue, point: npt.NDArray[np.float64]) -> bool:
        """Check if a concrete point lies within the interval bounds.

        Args:
            abstract: An interval abstract value.
            point: Concrete point to check.

        Returns:
            True if point is within the interval bounds.
        """
        point = np.asarray(point, dtype=np.float64)
        return bool(
            np.all(point >= abstract.lower - 1e-12)
            and np.all(point <= abstract.upper + 1e-12)
        )

    def abstract_privacy_ratio(
        self,
        mechanism_row_a: AbstractValue,
        mechanism_row_b: AbstractValue,
    ) -> Tuple[float, float]:
        """Compute sound over-approximation of privacy loss ratio bounds.

        For adjacent rows a and b of a mechanism, computes interval bounds
        on max_j P(a,j)/P(b,j).

        Args:
            mechanism_row_a: Abstract value for row a probabilities.
            mechanism_row_b: Abstract value for row b probabilities.

        Returns:
            (lower_bound, upper_bound) on the privacy loss ratio.
        """
        eps = 1e-300
        max_ratio_upper = np.max(mechanism_row_a.upper / np.maximum(mechanism_row_b.lower, eps))
        min_ratio_lower = np.min(mechanism_row_a.lower / np.maximum(mechanism_row_b.upper, eps))
        return float(min_ratio_lower), float(max_ratio_upper)

    def widen_with_thresholds(
        self,
        a: AbstractValue,
        b: AbstractValue,
        thresholds: npt.NDArray[np.float64],
    ) -> AbstractValue:
        """Widening with thresholds for convergence acceleration.

        Instead of jumping to ±infinity, uses the next threshold value.

        Args:
            a: Previous abstract value.
            b: New abstract value.
            thresholds: Sorted array of threshold values per dimension.

        Returns:
            Widened abstract value using thresholds.
        """
        if self.is_bottom(a):
            return copy.deepcopy(b)

        ndim = a.lower.shape[0]
        lower = np.copy(a.lower)
        upper = np.copy(a.upper)

        for i in range(ndim):
            if b.lower[i] < a.lower[i] - 1e-12:
                # Find the largest threshold below b.lower[i]
                candidates = thresholds[thresholds <= b.lower[i] + 1e-12]
                lower[i] = candidates[-1] if len(candidates) > 0 else -np.inf
            if b.upper[i] > a.upper[i] + 1e-12:
                # Find the smallest threshold above b.upper[i]
                candidates = thresholds[thresholds >= b.upper[i] - 1e-12]
                upper[i] = candidates[0] if len(candidates) > 0 else np.inf

        return AbstractValue(
            domain_type=AbstractDomainType.INTERVAL,
            lower=lower,
            upper=upper,
        )


# ---------------------------------------------------------------------------
# Polyhedral abstract domain
# ---------------------------------------------------------------------------


@dataclass
class PolyhedralConstraint:
    """A linear constraint of the form coeffs @ x <= rhs.

    Attributes:
        coeffs: Coefficient vector.
        rhs: Right-hand side scalar.
    """
    coeffs: npt.NDArray[np.float64]
    rhs: float


class PolyhedralAbstraction(AbstractDomain):
    """Polyhedral abstract domain using constraint matrices.

    Represents convex polyhedra {x : Ax <= b} for precise relational
    analysis of privacy parameters.
    """

    def __init__(self, max_constraints: int = 200) -> None:
        """Initialize polyhedral abstraction.

        Args:
            max_constraints: Maximum number of constraints to maintain.
        """
        self._max_constraints = max_constraints

    @property
    def domain_type(self) -> AbstractDomainType:
        """Return the domain type identifier."""
        return AbstractDomainType.POLYHEDRA

    def _make_value(
        self,
        ndim: int,
        constraints: List[PolyhedralConstraint],
    ) -> AbstractValue:
        """Create an AbstractValue with polyhedral constraints encoded."""
        lower, upper = self._bounds_from_constraints(ndim, constraints)
        return AbstractValue(
            domain_type=AbstractDomainType.POLYHEDRA,
            lower=lower,
            upper=upper,
            constraints=constraints,
        )

    def _bounds_from_constraints(
        self,
        ndim: int,
        constraints: List[PolyhedralConstraint],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute bounding box from polyhedral constraints via LP.

        Args:
            ndim: Number of dimensions.
            constraints: List of polyhedral constraints.

        Returns:
            (lower, upper) bounding box arrays.
        """
        if not constraints:
            return (
                np.full(ndim, -np.inf, dtype=np.float64),
                np.full(ndim, np.inf, dtype=np.float64),
            )

        A = np.array([c.coeffs for c in constraints], dtype=np.float64)
        b_ub = np.array([c.rhs for c in constraints], dtype=np.float64)

        lower = np.full(ndim, -np.inf, dtype=np.float64)
        upper = np.full(ndim, np.inf, dtype=np.float64)

        for i in range(ndim):
            # Minimize x_i
            c_obj = np.zeros(ndim, dtype=np.float64)
            c_obj[i] = 1.0
            res = optimize.linprog(c_obj, A_ub=A, b_ub=b_ub, method="highs")
            if res.success:
                lower[i] = res.fun
            # Maximize x_i
            c_obj[i] = -1.0
            res = optimize.linprog(c_obj, A_ub=A, b_ub=b_ub, method="highs")
            if res.success:
                upper[i] = -res.fun
            c_obj[i] = 0.0

        return lower, upper

    def bottom(self, ndim: int) -> AbstractValue:
        """Return bottom element (infeasible polyhedron)."""
        # Contradictory constraint: x_0 <= -1 and x_0 >= 1
        c1 = PolyhedralConstraint(
            coeffs=np.eye(1, ndim, 0, dtype=np.float64).flatten(),
            rhs=-1.0,
        )
        c2 = PolyhedralConstraint(
            coeffs=-np.eye(1, ndim, 0, dtype=np.float64).flatten(),
            rhs=-1.0,
        )
        return AbstractValue(
            domain_type=AbstractDomainType.POLYHEDRA,
            lower=np.full(ndim, np.inf, dtype=np.float64),
            upper=np.full(ndim, -np.inf, dtype=np.float64),
            constraints=[c1, c2],
        )

    def top(self, ndim: int) -> AbstractValue:
        """Return top element (unconstrained polyhedron)."""
        return self._make_value(ndim, [])

    def join(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Convex hull over-approximation via constraint intersection.

        Takes constraints valid in both a and b (those satisfied by both).

        Args:
            a: First polyhedral abstract value.
            b: Second polyhedral abstract value.

        Returns:
            Over-approximation of the convex hull.
        """
        if self.is_bottom(a):
            return copy.deepcopy(b)
        if self.is_bottom(b):
            return copy.deepcopy(a)

        ndim = a.lower.shape[0]
        constraints_a = a.constraints or []
        constraints_b = b.constraints or []

        # Keep constraints from a that are satisfied by b's bounding box,
        # and vice versa
        joint: List[PolyhedralConstraint] = []
        for c in constraints_a:
            # Check if constraint holds for b's bounding box
            max_val = np.sum(np.maximum(c.coeffs, 0) * b.upper + np.minimum(c.coeffs, 0) * b.lower)
            if max_val <= c.rhs + 1e-9:
                joint.append(c)
        for c in constraints_b:
            max_val = np.sum(np.maximum(c.coeffs, 0) * a.upper + np.minimum(c.coeffs, 0) * a.lower)
            if max_val <= c.rhs + 1e-9:
                # Only add if not already present
                is_dup = any(
                    np.allclose(c.coeffs, existing.coeffs) and abs(c.rhs - existing.rhs) < 1e-9
                    for existing in joint
                )
                if not is_dup:
                    joint.append(c)

        return self._make_value(ndim, joint[:self._max_constraints])

    def meet(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Intersection: combine all constraints from both polyhedra.

        Args:
            a: First polyhedral abstract value.
            b: Second polyhedral abstract value.

        Returns:
            Intersection of the two polyhedra.
        """
        ndim = a.lower.shape[0]
        constraints = list(a.constraints or []) + list(b.constraints or [])
        return self._make_value(ndim, constraints[:self._max_constraints])

    def widen(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Standard polyhedral widening.

        Keeps constraints from a that are satisfied by b.

        Args:
            a: Previous abstract value.
            b: New abstract value (after one iteration).

        Returns:
            Widened polyhedral abstract value.
        """
        if self.is_bottom(a):
            return copy.deepcopy(b)

        ndim = a.lower.shape[0]
        constraints_a = a.constraints or []
        kept: List[PolyhedralConstraint] = []
        for c in constraints_a:
            # Check if b satisfies this constraint
            max_val = np.sum(np.maximum(c.coeffs, 0) * b.upper + np.minimum(c.coeffs, 0) * b.lower)
            if max_val <= c.rhs + 1e-9:
                kept.append(c)
        return self._make_value(ndim, kept)

    def narrow(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Polyhedral narrowing: add constraints from b to a.

        Args:
            a: Widened abstract value.
            b: More precise abstract value.

        Returns:
            Narrowed abstract value.
        """
        ndim = a.lower.shape[0]
        constraints = list(a.constraints or []) + list(b.constraints or [])
        return self._make_value(ndim, constraints[:self._max_constraints])

    def is_bottom(self, a: AbstractValue) -> bool:
        """Check if the polyhedron is empty."""
        return bool(np.any(a.lower > a.upper + 1e-12))

    def leq(self, a: AbstractValue, b: AbstractValue) -> bool:
        """Check containment a ⊆ b using LP feasibility.

        For each constraint c in b, verify that max c·x over a <= c.rhs.
        """
        if self.is_bottom(a):
            return True
        if self.is_bottom(b):
            return False

        constraints_b = b.constraints or []
        if not constraints_b:
            return True

        constraints_a = a.constraints or []
        if not constraints_a:
            # a is top — only contained in b if b is also top
            return not constraints_b

        ndim = a.lower.shape[0]
        A_a = np.array([c.coeffs for c in constraints_a], dtype=np.float64)
        b_a = np.array([c.rhs for c in constraints_a], dtype=np.float64)

        for c in constraints_b:
            res = optimize.linprog(
                -c.coeffs, A_ub=A_a, b_ub=b_a, method="highs",
            )
            if res.success and -res.fun > c.rhs + 1e-9:
                return False
        return True

    def alpha(self, concrete: npt.NDArray[np.float64]) -> AbstractValue:
        """Abstract concrete points to their convex hull (bounding box + octagonal).

        Args:
            concrete: Array of shape (m, ndim).

        Returns:
            Polyhedral abstract value containing all points.
        """
        concrete = np.atleast_2d(concrete)
        ndim = concrete.shape[1]

        constraints: List[PolyhedralConstraint] = []
        # Add bounding box constraints
        for i in range(ndim):
            lo = np.min(concrete[:, i])
            hi = np.max(concrete[:, i])
            # x_i <= hi
            c = np.zeros(ndim, dtype=np.float64)
            c[i] = 1.0
            constraints.append(PolyhedralConstraint(coeffs=c.copy(), rhs=hi))
            # -x_i <= -lo
            c[i] = -1.0
            constraints.append(PolyhedralConstraint(coeffs=c.copy(), rhs=-lo))

        # Add pairwise difference constraints (octagonal)
        for i in range(min(ndim, 10)):
            for j in range(i + 1, min(ndim, 10)):
                diffs = concrete[:, i] - concrete[:, j]
                sums = concrete[:, i] + concrete[:, j]
                c = np.zeros(ndim, dtype=np.float64)
                # x_i - x_j <= max(diffs)
                c[i] = 1.0
                c[j] = -1.0
                constraints.append(PolyhedralConstraint(coeffs=c.copy(), rhs=float(np.max(diffs))))
                # x_j - x_i <= max(-diffs)
                c[i] = -1.0
                c[j] = 1.0
                constraints.append(PolyhedralConstraint(coeffs=c.copy(), rhs=float(np.max(-diffs))))
                # x_i + x_j <= max(sums)
                c[i] = 1.0
                c[j] = 1.0
                constraints.append(PolyhedralConstraint(coeffs=c.copy(), rhs=float(np.max(sums))))
                # -x_i - x_j <= -min(sums)
                c[i] = -1.0
                c[j] = -1.0
                constraints.append(PolyhedralConstraint(coeffs=c.copy(), rhs=float(-np.min(sums))))

        return self._make_value(ndim, constraints[:self._max_constraints])

    def gamma_contains(self, abstract: AbstractValue, point: npt.NDArray[np.float64]) -> bool:
        """Check if a concrete point satisfies all polyhedral constraints.

        Args:
            abstract: Polyhedral abstract value.
            point: Concrete point to check.

        Returns:
            True if point satisfies all constraints.
        """
        point = np.asarray(point, dtype=np.float64)
        constraints = abstract.constraints or []
        for c in constraints:
            if np.dot(c.coeffs, point) > c.rhs + 1e-9:
                return False
        return True

    def add_constraint(
        self,
        abstract: AbstractValue,
        coeffs: npt.NDArray[np.float64],
        rhs: float,
    ) -> AbstractValue:
        """Add a linear constraint to the polyhedron.

        Args:
            abstract: Existing polyhedral abstract value.
            coeffs: Constraint coefficients.
            rhs: Right-hand side bound.

        Returns:
            New abstract value with the added constraint.
        """
        ndim = abstract.lower.shape[0]
        constraints = list(abstract.constraints or [])
        constraints.append(PolyhedralConstraint(
            coeffs=np.asarray(coeffs, dtype=np.float64),
            rhs=float(rhs),
        ))
        return self._make_value(ndim, constraints[:self._max_constraints])


# ---------------------------------------------------------------------------
# Zonotope abstract domain
# ---------------------------------------------------------------------------


class ZonotopeAbstraction(AbstractDomain):
    """Zonotope abstract domain for efficient over-approximation.

    A zonotope Z(c, G) = {c + G·ξ : ξ ∈ [-1,1]^p} where c is the center
    and G is the generator matrix. Zonotopes are closed under affine
    transformations and provide a good precision/cost trade-off.
    """

    def __init__(self, max_generators: int = 100) -> None:
        """Initialize zonotope abstraction.

        Args:
            max_generators: Maximum number of generators to maintain.
        """
        self._max_generators = max_generators

    @property
    def domain_type(self) -> AbstractDomainType:
        """Return the domain type identifier."""
        return AbstractDomainType.ZONE

    def _make_value(
        self,
        center: npt.NDArray[np.float64],
        generators: npt.NDArray[np.float64],
    ) -> AbstractValue:
        """Create an AbstractValue from zonotope center and generators.

        The center is stored as the midpoint of lower/upper, and generators
        are stored in the constraints field.

        Args:
            center: Center vector of shape (ndim,).
            generators: Generator matrix of shape (ndim, p).

        Returns:
            AbstractValue representing the zonotope.
        """
        # Compute bounding box: lower = center - sum|generators|, upper = center + sum|generators|
        extent = np.sum(np.abs(generators), axis=1) if generators.size > 0 else np.zeros_like(center)
        lower = center - extent
        upper = center + extent
        return AbstractValue(
            domain_type=AbstractDomainType.ZONE,
            lower=lower,
            upper=upper,
            constraints=[{"center": center.copy(), "generators": generators.copy()}],
        )

    def _get_zonotope_data(
        self, a: AbstractValue
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Extract center and generators from an AbstractValue.

        Args:
            a: Zonotope abstract value.

        Returns:
            (center, generators) tuple.
        """
        if a.constraints and isinstance(a.constraints[0], dict):
            return a.constraints[0]["center"], a.constraints[0]["generators"]
        # Fallback: reconstruct from bounds
        center = (a.lower + a.upper) / 2.0
        half_widths = (a.upper - a.lower) / 2.0
        generators = np.diag(half_widths)
        return center, generators

    def bottom(self, ndim: int) -> AbstractValue:
        """Return bottom element (degenerate zonotope)."""
        return AbstractValue(
            domain_type=AbstractDomainType.ZONE,
            lower=np.full(ndim, np.inf, dtype=np.float64),
            upper=np.full(ndim, -np.inf, dtype=np.float64),
            constraints=[{"center": np.zeros(ndim), "generators": np.zeros((ndim, 0))}],
        )

    def top(self, ndim: int) -> AbstractValue:
        """Return top element (entire space)."""
        lower = np.full(ndim, -np.inf, dtype=np.float64)
        upper = np.full(ndim, np.inf, dtype=np.float64)
        center = np.zeros(ndim, dtype=np.float64)
        generators = np.diag(np.full(ndim, 1e15, dtype=np.float64))
        return AbstractValue(
            domain_type=AbstractDomainType.ZONE,
            lower=lower,
            upper=upper,
            constraints=[{"center": center, "generators": generators}],
        )

    def join(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Zonotope join via generator concatenation and reduction.

        Computes a zonotope containing both a and b by combining their
        generators with an additional generator for the center difference.

        Args:
            a: First zonotope abstract value.
            b: Second zonotope abstract value.

        Returns:
            Zonotope containing both inputs.
        """
        if self.is_bottom(a):
            return copy.deepcopy(b)
        if self.is_bottom(b):
            return copy.deepcopy(a)

        c_a, G_a = self._get_zonotope_data(a)
        c_b, G_b = self._get_zonotope_data(b)

        # New center is midpoint
        center = (c_a + c_b) / 2.0
        # Additional generator for center difference
        diff_gen = ((c_a - c_b) / 2.0).reshape(-1, 1)
        # Combine generators: [G_a, G_b, diff_gen]
        generators = np.hstack([G_a, G_b, diff_gen]) if G_a.size > 0 and G_b.size > 0 else diff_gen

        # Reduce generators if too many
        generators = self._reduce_generators(generators)

        return self._make_value(center, generators)

    def meet(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Zonotope meet via interval intersection on bounding box.

        Exact meet is not efficiently computable for zonotopes, so we
        use the bounding box intersection.

        Args:
            a: First zonotope abstract value.
            b: Second zonotope abstract value.

        Returns:
            Over-approximation of the intersection.
        """
        lower = np.maximum(a.lower, b.lower)
        upper = np.minimum(a.upper, b.upper)
        if np.any(lower > upper + 1e-12):
            return self.bottom(a.lower.shape[0])
        center = (lower + upper) / 2.0
        half_widths = np.maximum((upper - lower) / 2.0, 0.0)
        generators = np.diag(half_widths)
        return self._make_value(center, generators)

    def widen(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Zonotope widening: scale generators outward.

        Args:
            a: Previous abstract value.
            b: New abstract value.

        Returns:
            Widened zonotope.
        """
        if self.is_bottom(a):
            return copy.deepcopy(b)

        joined = self.join(a, b)
        c_j, G_j = self._get_zonotope_data(joined)

        # Scale generators by factor 2 where b exceeds a
        scale = np.ones(G_j.shape[1], dtype=np.float64)
        for i in range(G_j.shape[1]):
            col_extent = np.sum(np.abs(G_j[:, i]))
            if col_extent > 1e-12:
                scale[i] = 2.0

        G_widened = G_j * scale[np.newaxis, :]
        return self._make_value(c_j, G_widened)

    def narrow(self, a: AbstractValue, b: AbstractValue) -> AbstractValue:
        """Zonotope narrowing: intersect bounding boxes.

        Args:
            a: Widened abstract value.
            b: More precise value.

        Returns:
            Narrowed zonotope.
        """
        return self.meet(a, b)

    def is_bottom(self, a: AbstractValue) -> bool:
        """Check if the zonotope is empty."""
        return bool(np.any(a.lower > a.upper + 1e-12))

    def leq(self, a: AbstractValue, b: AbstractValue) -> bool:
        """Check containment via bounding box (over-approximate).

        Args:
            a: First abstract value.
            b: Second abstract value.

        Returns:
            True if a's bounding box is contained in b's bounding box.
        """
        if self.is_bottom(a):
            return True
        if self.is_bottom(b):
            return False
        return bool(
            np.all(a.lower >= b.lower - 1e-12)
            and np.all(a.upper <= b.upper + 1e-12)
        )

    def alpha(self, concrete: npt.NDArray[np.float64]) -> AbstractValue:
        """Abstract concrete points to a zonotope.

        Uses PCA-like approach: center at mean, generators from
        principal directions scaled to contain all points.

        Args:
            concrete: Array of shape (m, ndim).

        Returns:
            Zonotope abstract value containing all points.
        """
        concrete = np.atleast_2d(concrete)
        m, ndim = concrete.shape

        center = np.mean(concrete, axis=0)
        centered = concrete - center

        if m <= 1:
            generators = np.diag(np.maximum(np.abs(center) * 0.01, 1e-10))
            return self._make_value(center, generators)

        # Use SVD to find principal directions
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        k = min(ndim, m, self._max_generators)
        generators = (Vt[:k].T * S[:k][np.newaxis, :])

        # Scale to ensure containment
        for i in range(m):
            pt = concrete[i]
            if not self.gamma_contains(self._make_value(center, generators), pt):
                # Add an axis-aligned generator to cover this point
                diff = pt - center
                generators = np.hstack([generators, diff.reshape(-1, 1)])

        generators = self._reduce_generators(generators)
        return self._make_value(center, generators)

    def gamma_contains(self, abstract: AbstractValue, point: npt.NDArray[np.float64]) -> bool:
        """Check if point is in the zonotope's bounding box (sound over-approximation)."""
        point = np.asarray(point, dtype=np.float64)
        return bool(
            np.all(point >= abstract.lower - 1e-12)
            and np.all(point <= abstract.upper + 1e-12)
        )

    def affine_transform(
        self,
        abstract: AbstractValue,
        matrix: npt.NDArray[np.float64],
        offset: npt.NDArray[np.float64],
    ) -> AbstractValue:
        """Apply affine transformation M·x + t to the zonotope.

        Zonotopes are closed under affine maps: Z(c, G) -> Z(Mc+t, MG).

        Args:
            abstract: Zonotope abstract value.
            matrix: Transformation matrix M.
            offset: Translation vector t.

        Returns:
            Transformed zonotope.
        """
        center, generators = self._get_zonotope_data(abstract)
        new_center = matrix @ center + offset
        new_generators = matrix @ generators if generators.size > 0 else np.zeros((matrix.shape[0], 0))
        return self._make_value(new_center, new_generators)

    def _reduce_generators(
        self, generators: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Reduce the number of generators using the Girard method.

        Keeps the largest generators and merges the rest into
        an axis-aligned bounding box.

        Args:
            generators: Generator matrix of shape (ndim, p).

        Returns:
            Reduced generator matrix.
        """
        ndim, p = generators.shape
        if p <= self._max_generators:
            return generators

        # Sort generators by column norm (descending)
        norms = np.linalg.norm(generators, axis=0)
        order = np.argsort(-norms)

        keep = self._max_generators - ndim  # Reserve room for box generators
        keep = max(keep, 0)

        kept = generators[:, order[:keep]]
        merged = generators[:, order[keep:]]

        # Merge remaining into axis-aligned generators
        box_extent = np.sum(np.abs(merged), axis=1)
        box_gens = np.diag(box_extent)

        return np.hstack([kept, box_gens])


# ---------------------------------------------------------------------------
# Galois connection utilities
# ---------------------------------------------------------------------------


class GaloisConnection:
    """Galois connection between concrete and abstract domains.

    Provides the alpha (abstraction) and gamma (concretization) maps
    forming a Galois connection (α, γ) between P(R^n) and the abstract
    domain lattice.
    """

    def __init__(self, domain: AbstractDomain) -> None:
        """Initialize with a concrete abstract domain implementation.

        Args:
            domain: The abstract domain to use.
        """
        self._domain = domain

    @property
    def domain(self) -> AbstractDomain:
        """The underlying abstract domain."""
        return self._domain

    def abstract(self, concrete: npt.NDArray[np.float64]) -> AbstractValue:
        """Apply the abstraction function α.

        Maps a set of concrete points to their best abstract approximation
        in the abstract domain.

        Args:
            concrete: Array of concrete points, shape (m, ndim).

        Returns:
            Abstract value in the domain.
        """
        return self._domain.alpha(concrete)

    def concretization_contains(
        self,
        abstract: AbstractValue,
        point: npt.NDArray[np.float64],
    ) -> bool:
        """Check membership in γ(abstract).

        Args:
            abstract: Abstract value.
            point: Concrete point to test.

        Returns:
            True if point ∈ γ(abstract).
        """
        return self._domain.gamma_contains(abstract, point)

    def is_sound_approximation(
        self,
        concrete_set: npt.NDArray[np.float64],
        abstract: AbstractValue,
    ) -> bool:
        """Verify that γ(abstract) ⊇ concrete_set (soundness check).

        Args:
            concrete_set: Array of concrete points.
            abstract: Abstract value to check.

        Returns:
            True if all concrete points are contained in γ(abstract).
        """
        concrete_set = np.atleast_2d(concrete_set)
        for point in concrete_set:
            if not self._domain.gamma_contains(abstract, point):
                return False
        return True

    def abstract_transfer(
        self,
        abstract_input: AbstractValue,
        transformer: AbstractDomain,
        *,
        join_with: Optional[AbstractValue] = None,
    ) -> AbstractValue:
        """Apply abstract transfer function with optional join.

        Computes transformer(abstract_input) ⊔ join_with.

        Args:
            abstract_input: Input abstract value.
            transformer: Domain providing the transfer.
            join_with: Optional value to join with result.

        Returns:
            Result of abstract transfer.
        """
        result = abstract_input  # Identity transfer in base case
        if join_with is not None:
            result = transformer.join(result, join_with)
        return result


# ---------------------------------------------------------------------------
# Privacy loss over-approximation
# ---------------------------------------------------------------------------


class PrivacyLossAbstraction:
    """Sound over-approximation of privacy loss using abstract domains.

    Computes abstract bounds on the privacy loss function
    L(o) = ln(P[M(x)=o] / P[M(x')=o]) for adjacent x, x'.
    """

    def __init__(self, domain: AbstractDomain) -> None:
        """Initialize with abstract domain.

        Args:
            domain: Abstract domain for bounding computations.
        """
        self._domain = domain

    def abstract_log_ratio(
        self,
        numerator: AbstractValue,
        denominator: AbstractValue,
    ) -> Tuple[float, float]:
        """Compute abstract bounds on ln(numerator / denominator).

        Uses interval arithmetic on the bounding boxes for soundness.

        Args:
            numerator: Abstract value for numerator probabilities.
            denominator: Abstract value for denominator probabilities.

        Returns:
            (lower_bound, upper_bound) on the log-ratio.
        """
        eps = 1e-300  # Avoid log(0)

        # Sound bounds: element-wise log ratios, then take max/min
        upper = float(np.max(np.log(np.maximum(numerator.upper, eps)
                                    / np.maximum(denominator.lower, eps))))
        lower = float(np.min(np.log(np.maximum(numerator.lower, eps)
                                    / np.maximum(denominator.upper, eps))))
        return lower, upper

    def check_epsilon_bound(
        self,
        mechanism: npt.NDArray[np.float64],
        adjacency: AdjacencyRelation,
        epsilon: float,
    ) -> Tuple[bool, Optional[Tuple[int, int, float]]]:
        """Check if mechanism satisfies ε-DP using abstract interpretation.

        For each adjacent pair (i, i'), verifies that the abstract privacy
        loss is bounded by ε.

        Args:
            mechanism: Probability matrix of shape (n, k).
            adjacency: Adjacency relation.
            epsilon: Privacy parameter.

        Returns:
            (is_verified, violation_info) where violation_info is
            (i, i', max_loss) if a potential violation is found.
        """
        n, k = mechanism.shape
        worst_violation: Optional[Tuple[int, int, float]] = None
        max_loss = 0.0

        all_edges = list(adjacency.edges)
        if adjacency.symmetric:
            all_edges = all_edges + [(j, i) for (i, j) in adjacency.edges]

        for (i, ip) in all_edges:
            row_i = self._domain.alpha(mechanism[i:i+1, :])
            row_ip = self._domain.alpha(mechanism[ip:ip+1, :])
            _, loss_upper = self.abstract_log_ratio(row_i, row_ip)

            if loss_upper > max_loss:
                max_loss = loss_upper
                if loss_upper > epsilon + 1e-9:
                    worst_violation = (i, ip, loss_upper)

        if worst_violation is not None:
            return False, worst_violation
        return True, None

    def abstract_composition(
        self,
        losses: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Abstract composition of privacy losses (sequential composition).

        Computes abstract bounds on the total privacy loss under sequential
        composition using interval arithmetic.

        Args:
            losses: List of (lower, upper) bounds on individual privacy losses.

        Returns:
            (total_lower, total_upper) bounds on composed privacy loss.
        """
        total_lower = sum(lo for lo, _ in losses)
        total_upper = sum(hi for _, hi in losses)
        return total_lower, total_upper
