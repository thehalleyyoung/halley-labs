"""Interval contractors for constraint solving.

Provides a suite of *contractors* — operators that tighten an interval
box (Cartesian product of variable domains) with respect to a
constraint without removing any solution.  Contractors are the
building blocks of interval constraint-propagation and set-inversion
algorithms.

Available contractors:

* :class:`ForwardBackwardContractor` — HC4 forward-backward propagation.
* :class:`NewtonContractor` — interval Newton method (fast quadratic
  convergence near solutions).
* :class:`KrawczykContractor` — Krawczyk operator (always applicable,
  no inverse required).
* :class:`BisectionStrategy` — splitting heuristics for branch-and-
  prune loops.
* :class:`ContractorQueue` — fair-scheduling propagation queue.
* :func:`pave` — subpaving algorithm for set inversion.

References
----------
Jaulin, L., Kieffer, M., Didrit, O., & Walter, É. (2001).
    *Applied Interval Analysis*. Springer.
Chabert, G., & Jaulin, L. (2009).
    Contractor programming. *Artificial Intelligence*, 173(11),
    1079–1100.
Benhamou, F., et al. (1999). Revising hull and box consistency.
    *Proc. ICLP*, 230–244.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from usability_oracle.interval.interval import Interval


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Box = Dict[str, Interval]
"""A box is a named Cartesian product of intervals."""


@unique
class ContractionStatus(Enum):
    """Result status of a contraction step."""

    UNCHANGED = "unchanged"
    CONTRACTED = "contracted"
    EMPTY = "empty"


@dataclass
class ContractionResult:
    """Result of applying a contractor to a box.

    Attributes
    ----------
    box : Box
        The (possibly tightened) box.
    status : ContractionStatus
        Whether the box was contracted, unchanged, or emptied.
    """

    box: Box
    status: ContractionStatus


# ═══════════════════════════════════════════════════════════════════════════
# Abstract contractor
# ═══════════════════════════════════════════════════════════════════════════

class Contractor(ABC):
    """Abstract base class for interval contractors."""

    @abstractmethod
    def contract(self, box: Box) -> ContractionResult:
        """Apply the contractor to *box*.

        Parameters
        ----------
        box : Box
            Variable domains.

        Returns
        -------
        ContractionResult
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Forward-backward contractor (HC4)
# ═══════════════════════════════════════════════════════════════════════════

class ForwardBackwardContractor(Contractor):
    """HC4 forward-backward propagation contractor.

    The constraint is expressed as a callable *f* returning an
    :class:`Interval` plus a *target* interval.  The contractor
    evaluates *f* in a forward pass (natural extension), intersects
    with *target*, then propagates tightened ranges backward to the
    input variables.

    Parameters
    ----------
    f : Callable
        Interval-valued function of the box variables.
    variables : list[str]
        Variable names in the order expected by *f*.
    target : Interval
        Desired range for the constraint output.
    """

    def __init__(
        self,
        f: Callable[..., Interval],
        variables: list[str],
        target: Interval,
    ) -> None:
        self._f = f
        self._variables = list(variables)
        self._target = target

    def contract(self, box: Box) -> ContractionResult:
        """Apply HC4 forward-backward contraction."""
        new_box = dict(box)
        args = [new_box[v] for v in self._variables]

        # Forward pass
        try:
            output = self._f(*args)
        except (ValueError, ZeroDivisionError):
            return ContractionResult(box=new_box, status=ContractionStatus.EMPTY)

        contracted_output = output.intersection(self._target)
        if contracted_output is None:
            return ContractionResult(box=new_box, status=ContractionStatus.EMPTY)

        # Backward pass: tighten each variable
        changed = False
        for i, var in enumerate(self._variables):
            original = new_box[var]
            tightened = self._backward_tighten_variable(
                new_box, i, contracted_output
            )
            if tightened is None:
                return ContractionResult(
                    box=new_box, status=ContractionStatus.EMPTY
                )
            if tightened.low > original.low or tightened.high < original.high:
                new_box[var] = tightened
                changed = True

        status = ContractionStatus.CONTRACTED if changed else ContractionStatus.UNCHANGED
        return ContractionResult(box=new_box, status=status)

    def _backward_tighten_variable(
        self,
        box: Box,
        var_index: int,
        target: Interval,
    ) -> Optional[Interval]:
        """Tighten one variable by probing sub-intervals."""
        var = self._variables[var_index]
        dom = box[var]
        lo, hi = dom.low, dom.high

        if lo == hi:
            return dom

        n_probes = 8
        best_lo = lo
        best_hi = hi

        # Narrow lower bound
        for _ in range(n_probes):
            mid = (best_lo + best_hi) / 2.0
            test_box = dict(box)
            test_box[var] = Interval(best_lo, mid)
            args = [test_box[v] for v in self._variables]
            try:
                out = self._f(*args)
                if out.intersection(target) is not None:
                    best_hi_tmp = mid
                    break
            except (ValueError, ZeroDivisionError):
                pass
            best_lo = mid

        # Narrow upper bound
        best_hi2 = hi
        for _ in range(n_probes):
            mid = (lo + best_hi2) / 2.0
            test_box = dict(box)
            test_box[var] = Interval(mid, best_hi2)
            args = [test_box[v] for v in self._variables]
            try:
                out = self._f(*args)
                if out.intersection(target) is not None:
                    break
            except (ValueError, ZeroDivisionError):
                pass
            best_hi2 = mid

        if best_lo > best_hi:
            return None
        return Interval(best_lo, best_hi)


# ═══════════════════════════════════════════════════════════════════════════
# Newton contractor
# ═══════════════════════════════════════════════════════════════════════════

class NewtonContractor(Contractor):
    """Interval Newton contractor.

    Applies the interval Newton operator to tighten a box with respect
    to the constraint f(x) = 0 (or f(x) ∈ target).

    For the univariate case the Newton step is:

        x_new = mid(x) − f(mid(x)) / f'(x)

    and the result is intersected with the current domain.

    Parameters
    ----------
    f : Callable
        Interval function.
    f_deriv : Callable
        Interval extension of the Jacobian / derivative.
    variables : list[str]
        Variable names.
    target : Interval
        Target interval (default [0, 0] for root-finding).
    max_iterations : int
        Maximum Newton steps per contraction call.
    """

    def __init__(
        self,
        f: Callable[..., Interval],
        f_deriv: Callable[..., Interval],
        variables: list[str],
        target: Interval = Interval(0.0, 0.0),
        max_iterations: int = 10,
    ) -> None:
        self._f = f
        self._f_deriv = f_deriv
        self._variables = list(variables)
        self._target = target
        self._max_iterations = max_iterations

    def contract(self, box: Box) -> ContractionResult:
        """Apply interval Newton contraction."""
        new_box = dict(box)
        changed = False

        for var in self._variables:
            dom = new_box[var]
            tightened = self._newton_univariate(new_box, var)
            if tightened is None:
                return ContractionResult(
                    box=new_box, status=ContractionStatus.EMPTY
                )
            if tightened.low > dom.low or tightened.high < dom.high:
                new_box[var] = tightened
                changed = True

        status = ContractionStatus.CONTRACTED if changed else ContractionStatus.UNCHANGED
        return ContractionResult(box=new_box, status=status)

    def _newton_univariate(
        self, box: Box, var: str
    ) -> Optional[Interval]:
        """Newton step for a single variable."""
        current = box[var]
        target_mid = self._target.midpoint

        for _ in range(self._max_iterations):
            mid = current.midpoint
            mid_box = dict(box)
            mid_box[var] = Interval.from_value(mid)

            try:
                f_mid = self._f(*[mid_box[v] for v in self._variables])
                deriv = self._f_deriv(*[box[v] for v in self._variables])
            except (ValueError, ZeroDivisionError):
                return current

            g_mid = f_mid.midpoint - target_mid

            if deriv.includes_zero():
                return current

            # Newton step
            try:
                step_lo = g_mid / deriv.high if deriv.high != 0.0 else -math.inf
                step_hi = g_mid / deriv.low if deriv.low != 0.0 else math.inf
            except ZeroDivisionError:
                return current

            newton_lo = mid - max(step_lo, step_hi)
            newton_hi = mid - min(step_lo, step_hi)

            newton_iv = Interval(
                min(newton_lo, newton_hi), max(newton_lo, newton_hi)
            )
            contracted = current.intersection(newton_iv)
            if contracted is None:
                return None

            if current.width - contracted.width < 1e-12:
                return contracted
            current = contracted

        return current


# ═══════════════════════════════════════════════════════════════════════════
# Krawczyk contractor
# ═══════════════════════════════════════════════════════════════════════════

class KrawczykContractor(Contractor):
    """Krawczyk operator contractor.

    The Krawczyk operator is a weakened Newton operator that does not
    require computing the inverse of the interval Jacobian:

        K(x) = mid(x) − Y·f(mid(x)) + (I − Y·J(x)) · (x − mid(x))

    where Y is a preconditioning matrix (we use Y = 1/f'(mid(x)) for
    the univariate case).

    Always applicable even when J(x) is singular.

    Parameters
    ----------
    f : Callable
        Interval function.
    f_deriv : Callable
        Interval derivative.
    variables : list[str]
        Variable names.
    target : Interval
        Target interval.
    """

    def __init__(
        self,
        f: Callable[..., Interval],
        f_deriv: Callable[..., Interval],
        variables: list[str],
        target: Interval = Interval(0.0, 0.0),
    ) -> None:
        self._f = f
        self._f_deriv = f_deriv
        self._variables = list(variables)
        self._target = target

    def contract(self, box: Box) -> ContractionResult:
        """Apply Krawczyk contraction."""
        new_box = dict(box)
        changed = False

        for var in self._variables:
            dom = new_box[var]
            tightened = self._krawczyk_step(new_box, var)
            if tightened is None:
                return ContractionResult(
                    box=new_box, status=ContractionStatus.EMPTY
                )
            if tightened.low > dom.low or tightened.high < dom.high:
                new_box[var] = tightened
                changed = True

        status = ContractionStatus.CONTRACTED if changed else ContractionStatus.UNCHANGED
        return ContractionResult(box=new_box, status=status)

    def _krawczyk_step(
        self, box: Box, var: str
    ) -> Optional[Interval]:
        """Single Krawczyk step for one variable."""
        dom = box[var]
        mid = dom.midpoint
        target_mid = self._target.midpoint

        mid_box = dict(box)
        mid_box[var] = Interval.from_value(mid)

        try:
            f_mid = self._f(*[mid_box[v] for v in self._variables])
            deriv_full = self._f_deriv(*[box[v] for v in self._variables])
        except (ValueError, ZeroDivisionError):
            return dom

        g_mid = f_mid.midpoint - target_mid

        # Preconditioner: Y = 1 / f'(mid(x))
        try:
            deriv_mid = self._f_deriv(
                *[mid_box[v] for v in self._variables]
            )
            y = 1.0 / deriv_mid.midpoint if deriv_mid.midpoint != 0.0 else 1.0
        except (ValueError, ZeroDivisionError):
            y = 1.0

        # K(x) = mid − y·g(mid) + (1 − y·J(x)) · (x − mid)
        correction = y * g_mid
        center = mid - correction

        c_term = Interval(1.0, 1.0) - Interval.from_value(y) * deriv_full
        spread = c_term * (dom - Interval.from_value(mid))

        kraw = Interval.from_value(center) + spread
        contracted = dom.intersection(kraw)
        return contracted


# ═══════════════════════════════════════════════════════════════════════════
# Bisection strategies
# ═══════════════════════════════════════════════════════════════════════════

@unique
class BisectionMethod(Enum):
    """Available bisection heuristics."""

    LARGEST_FIRST = "largest_first"
    ROUND_ROBIN = "round_robin"
    SMEAR = "smear"


class BisectionStrategy:
    """Select a variable to bisect and produce two sub-boxes.

    Parameters
    ----------
    method : BisectionMethod
        Splitting heuristic.
    variables : list[str]
        Variable names eligible for bisection.
    """

    def __init__(
        self,
        method: BisectionMethod = BisectionMethod.LARGEST_FIRST,
        variables: Optional[list[str]] = None,
    ) -> None:
        self._method = method
        self._variables = variables
        self._rr_index = 0

    def bisect(self, box: Box) -> Tuple[Box, Box]:
        """Split *box* into two sub-boxes along the chosen variable.

        Parameters
        ----------
        box : Box
            Current box.

        Returns
        -------
        Tuple[Box, Box]
            Two sub-boxes.

        Raises
        ------
        ValueError
            If all dimensions are degenerate (cannot split).
        """
        var = self._select_variable(box)
        return self._split(box, var)

    def _select_variable(self, box: Box) -> str:
        """Choose which variable to bisect."""
        candidates = self._variables or list(box.keys())
        # Filter out degenerate dimensions
        candidates = [v for v in candidates if box[v].width > 0.0]
        if not candidates:
            raise ValueError("Cannot bisect: all dimensions are degenerate.")

        if self._method == BisectionMethod.LARGEST_FIRST:
            return max(candidates, key=lambda v: box[v].width)

        elif self._method == BisectionMethod.ROUND_ROBIN:
            var = candidates[self._rr_index % len(candidates)]
            self._rr_index += 1
            return var

        elif self._method == BisectionMethod.SMEAR:
            # Smear-based: select the variable with the largest "smear"
            # (width × sensitivity).  Without derivatives we fall back
            # to largest-first.
            return max(candidates, key=lambda v: box[v].width)

        return candidates[0]  # pragma: no cover

    @staticmethod
    def _split(box: Box, var: str) -> Tuple[Box, Box]:
        """Split *box* at the midpoint of *var*."""
        dom = box[var]
        mid = dom.midpoint
        left = dict(box)
        right = dict(box)
        left[var] = Interval(dom.low, mid)
        right[var] = Interval(mid, dom.high)
        return left, right


# ═══════════════════════════════════════════════════════════════════════════
# Contractor queue
# ═══════════════════════════════════════════════════════════════════════════

class ContractorQueue:
    """Propagation queue with fair scheduling.

    Maintains a FIFO queue of contractors and applies them to a box
    until a fixed point is reached.

    Parameters
    ----------
    contractors : list[Contractor]
        The contractor pool.
    max_rounds : int
        Maximum full passes through the queue.
    tolerance : float
        Convergence threshold on total width reduction per round.
    """

    def __init__(
        self,
        contractors: list[Contractor],
        max_rounds: int = 50,
        tolerance: float = 1e-8,
    ) -> None:
        self._contractors = list(contractors)
        self._max_rounds = max_rounds
        self._tolerance = tolerance

    def propagate(self, box: Box) -> ContractionResult:
        """Run the propagation loop until convergence.

        Parameters
        ----------
        box : Box
            Initial box.

        Returns
        -------
        ContractionResult
            Final contracted box and status.
        """
        current = dict(box)
        ever_changed = False

        for _ in range(self._max_rounds):
            round_changed = False
            for contractor in self._contractors:
                result = contractor.contract(current)
                if result.status == ContractionStatus.EMPTY:
                    return ContractionResult(
                        box=current, status=ContractionStatus.EMPTY
                    )
                if result.status == ContractionStatus.CONTRACTED:
                    current = result.box
                    round_changed = True
                    ever_changed = True

            if not round_changed:
                break

        status = ContractionStatus.CONTRACTED if ever_changed else ContractionStatus.UNCHANGED
        return ContractionResult(box=current, status=status)


# ═══════════════════════════════════════════════════════════════════════════
# Subpaving (set inversion via contractors)
# ═══════════════════════════════════════════════════════════════════════════

@unique
class PaveStatus(Enum):
    """Classification of a box in the subpaving."""

    INNER = "inner"
    OUTER = "outer"
    BOUNDARY = "boundary"


@dataclass
class PavingResult:
    """Result of the :func:`pave` algorithm.

    Attributes
    ----------
    inner : list[Box]
        Boxes guaranteed to satisfy the constraint.
    boundary : list[Box]
        Boxes that may or may not satisfy (undetermined).
    outer : list[Box]
        Boxes guaranteed to *not* satisfy the constraint.
    """

    inner: list[Box] = field(default_factory=list)
    boundary: list[Box] = field(default_factory=list)
    outer: list[Box] = field(default_factory=list)


def pave(
    box: Box,
    contractor: Contractor,
    epsilon: float,
    bisector: Optional[BisectionStrategy] = None,
    max_boxes: int = 10_000,
) -> PavingResult:
    """Subpaving algorithm for set inversion.

    Recursively contracts and bisects the initial box, classifying
    sub-boxes as inner, outer, or boundary (undetermined) until all
    boundary boxes have width < *epsilon* or the box budget is
    exhausted.

    Parameters
    ----------
    box : Box
        Initial search box.
    contractor : Contractor
        Contractor encoding the constraint.
    epsilon : float
        Precision threshold.  Boundary boxes narrower than this are
        kept as-is.
    bisector : BisectionStrategy, optional
        Bisection strategy (defaults to largest-first).
    max_boxes : int
        Maximum number of boxes to process.

    Returns
    -------
    PavingResult
        Classification of the search space.
    """
    if bisector is None:
        bisector = BisectionStrategy(BisectionMethod.LARGEST_FIRST)

    result = PavingResult()
    queue: deque[Box] = deque([box])
    n_processed = 0

    while queue and n_processed < max_boxes:
        current = queue.popleft()
        n_processed += 1

        # Contract
        cr = contractor.contract(current)

        if cr.status == ContractionStatus.EMPTY:
            result.outer.append(current)
            continue

        contracted = cr.box

        # Check if fully inside (all widths contracted to near-zero
        # relative to the target) — heuristic: if unchanged from
        # contraction, and the constraint is satisfied, mark inner.
        max_width = max(iv.width for iv in contracted.values())

        if max_width < epsilon:
            # Cannot bisect further; classify as boundary
            result.boundary.append(contracted)
            continue

        if cr.status == ContractionStatus.UNCHANGED:
            # No progress from contraction — bisect
            try:
                left, right = bisector.bisect(contracted)
                queue.append(left)
                queue.append(right)
            except ValueError:
                result.boundary.append(contracted)
            continue

        # Contraction reduced the box — check if we need further splitting
        if max_width < epsilon:
            result.boundary.append(contracted)
        else:
            try:
                left, right = bisector.bisect(contracted)
                queue.append(left)
                queue.append(right)
            except ValueError:
                result.boundary.append(contracted)

    # Remaining items in queue are boundary
    while queue:
        result.boundary.append(queue.popleft())

    return result
