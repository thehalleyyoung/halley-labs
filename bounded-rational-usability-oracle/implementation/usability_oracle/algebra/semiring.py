"""
usability_oracle.algebra.semiring — Cost semiring algebra.

Provides a family of semirings for computing various aggregates over
cognitive cost graphs.  Each semiring ``(S, ⊕, ⊗, 0̄, 1̄)`` satisfies:

* ``(S, ⊕, 0̄)`` is a commutative monoid (additive identity)
* ``(S, ⊗, 1̄)`` is a monoid (multiplicative identity)
* ``⊗`` distributes over ``⊕``
* ``0̄`` annihilates under ``⊗``

Semiring Instances
------------------
* **Tropical** ``(ℝ ∪ {∞}, min, +, ∞, 0)`` — shortest-path (minimum cost)
* **MaxPlus** ``(ℝ ∪ {-∞}, max, +, -∞, 0)`` — longest-path (critical path)
* **Log** ``(ℝ ∪ {∞}, ⊕_log, +, ∞, 0)`` — log-domain probability sums
* **Viterbi** ``([0,1], max, ×, 0, 1)`` — most-likely path decoding
* **Boolean** ``({0,1}, ∨, ∧, 0, 1)`` — reachability
* **ExpectedCost** ``((μ, σ²), …)`` — expected cost with variance propagation
* **Interval** ``([a,b], …)`` — interval arithmetic for uncertainty

Application
~~~~~~~~~~~
All-pairs cognitive cost computation via semiring matrix multiplication
and Kleene closure (transitive closure / all-paths summary).

References
----------
* Mohri, *Semiring Frameworks and Algorithms for Shortest-Distance
  Problems*, J. Automata, Languages and Combinatorics, 2002.
* Goodman, *Semiring Parsing*, Computational Linguistics, 1999.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Abstract semiring
# ---------------------------------------------------------------------------


class Semiring(ABC, Generic[T]):
    """Abstract base class for a semiring ``(S, ⊕, ⊗, 0̄, 1̄)``."""

    @abstractmethod
    def zero(self) -> T:
        """Additive identity ``0̄``."""
        ...

    @abstractmethod
    def one(self) -> T:
        """Multiplicative identity ``1̄``."""
        ...

    @abstractmethod
    def add(self, a: T, b: T) -> T:
        """Semiring addition ``a ⊕ b``."""
        ...

    @abstractmethod
    def mul(self, a: T, b: T) -> T:
        """Semiring multiplication ``a ⊗ b``."""
        ...

    def sum(self, values: Sequence[T]) -> T:
        """Fold addition over a sequence."""
        result = self.zero()
        for v in values:
            result = self.add(result, v)
        return result

    def product(self, values: Sequence[T]) -> T:
        """Fold multiplication over a sequence."""
        result = self.one()
        for v in values:
            result = self.mul(result, v)
        return result


# ---------------------------------------------------------------------------
# Concrete semirings
# ---------------------------------------------------------------------------

_INF = float("inf")
_NINF = float("-inf")


class TropicalSemiring(Semiring[float]):
    r"""Tropical semiring ``(ℝ ∪ {∞}, min, +, ∞, 0)``.

    Used for **shortest-path** (minimum total cost) computation.
    Floyd-Warshall becomes tropical matrix multiplication.
    """

    def zero(self) -> float:
        return _INF

    def one(self) -> float:
        return 0.0

    def add(self, a: float, b: float) -> float:
        return min(a, b)

    def mul(self, a: float, b: float) -> float:
        if a == _INF or b == _INF:
            return _INF
        return a + b


class MaxPlusSemiring(Semiring[float]):
    r"""Max-plus semiring ``(ℝ ∪ {-∞}, max, +, -∞, 0)``.

    Used for **longest-path** (critical path) computation.
    """

    def zero(self) -> float:
        return _NINF

    def one(self) -> float:
        return 0.0

    def add(self, a: float, b: float) -> float:
        return max(a, b)

    def mul(self, a: float, b: float) -> float:
        if a == _NINF or b == _NINF:
            return _NINF
        return a + b


class LogSemiring(Semiring[float]):
    r"""Log semiring for probability computations.

    Represents log-probabilities; addition is log-sum-exp:

    .. math::

        a \oplus b = -\log(e^{-a} + e^{-b})

    Multiplication is ordinary addition of log-values.

    ``0̄ = ∞`` (probability 0), ``1̄ = 0`` (probability 1).
    """

    def zero(self) -> float:
        return _INF

    def one(self) -> float:
        return 0.0

    def add(self, a: float, b: float) -> float:
        if a == _INF:
            return b
        if b == _INF:
            return a
        # Numerically stable log-sum-exp
        lo, hi = min(a, b), max(a, b)
        return -np.logaddexp(-lo, -hi)

    def mul(self, a: float, b: float) -> float:
        if a == _INF or b == _INF:
            return _INF
        return a + b


class ViterbiSemiring(Semiring[float]):
    r"""Viterbi semiring ``([0,1], max, ×, 0, 1)``.

    Used for **most-likely path** decoding.
    Probabilities are in ``[0, 1]``.
    """

    def zero(self) -> float:
        return 0.0

    def one(self) -> float:
        return 1.0

    def add(self, a: float, b: float) -> float:
        return max(a, b)

    def mul(self, a: float, b: float) -> float:
        return a * b


class BooleanSemiring(Semiring[bool]):
    r"""Boolean semiring ``({False, True}, ∨, ∧, False, True)``.

    Used for **reachability** analysis: can state *j* be reached from *i*?
    """

    def zero(self) -> bool:
        return False

    def one(self) -> bool:
        return True

    def add(self, a: bool, b: bool) -> bool:
        return a or b

    def mul(self, a: bool, b: bool) -> bool:
        return a and b


@dataclass
class ExpectedCostValue:
    """Value in the expected cost semiring: ``(mean, variance)``."""
    mu: float = 0.0
    var: float = 0.0

    def __repr__(self) -> str:
        return f"E[{self.mu:.4f}, σ²={self.var:.4f}]"


class ExpectedCostSemiring(Semiring[ExpectedCostValue]):
    r"""Expected cost semiring with variance propagation.

    * **Addition** ``(μ₁, σ₁²) ⊕ (μ₂, σ₂²)``: selects the minimum expected
      cost (for shortest-path semantics) while propagating variance.
    * **Multiplication** ``(μ₁, σ₁²) ⊗ (μ₂, σ₂²)``: sequential composition
      ``(μ₁ + μ₂, σ₁² + σ₂²)`` (independent).

    ``0̄ = (∞, 0)``; ``1̄ = (0, 0)``.
    """

    def zero(self) -> ExpectedCostValue:
        return ExpectedCostValue(mu=_INF, var=0.0)

    def one(self) -> ExpectedCostValue:
        return ExpectedCostValue(mu=0.0, var=0.0)

    def add(self, a: ExpectedCostValue, b: ExpectedCostValue) -> ExpectedCostValue:
        if a.mu <= b.mu:
            return ExpectedCostValue(mu=a.mu, var=a.var)
        return ExpectedCostValue(mu=b.mu, var=b.var)

    def mul(self, a: ExpectedCostValue, b: ExpectedCostValue) -> ExpectedCostValue:
        if a.mu == _INF or b.mu == _INF:
            return ExpectedCostValue(mu=_INF, var=0.0)
        return ExpectedCostValue(mu=a.mu + b.mu, var=a.var + b.var)


@dataclass
class IntervalValue:
    """Value in the interval semiring: ``[lo, hi]``."""
    lo: float = 0.0
    hi: float = 0.0

    def width(self) -> float:
        return self.hi - self.lo

    def midpoint(self) -> float:
        return (self.lo + self.hi) / 2.0

    def __repr__(self) -> str:
        return f"[{self.lo:.4f}, {self.hi:.4f}]"


class IntervalSemiring(Semiring[IntervalValue]):
    r"""Interval semiring for uncertainty propagation.

    * **Addition**: interval union (hull)
      ``[a, b] ⊕ [c, d] = [min(a, c), max(b, d)]``
    * **Multiplication**: interval addition
      ``[a, b] ⊗ [c, d] = [a + c, b + d]``

    ``0̄ = [∞, -∞]`` (empty); ``1̄ = [0, 0]``.
    """

    def zero(self) -> IntervalValue:
        return IntervalValue(lo=_INF, hi=_NINF)

    def one(self) -> IntervalValue:
        return IntervalValue(lo=0.0, hi=0.0)

    def add(self, a: IntervalValue, b: IntervalValue) -> IntervalValue:
        return IntervalValue(lo=min(a.lo, b.lo), hi=max(a.hi, b.hi))

    def mul(self, a: IntervalValue, b: IntervalValue) -> IntervalValue:
        if a.lo == _INF or b.lo == _INF:
            return self.zero()
        return IntervalValue(lo=a.lo + b.lo, hi=a.hi + b.hi)


# ---------------------------------------------------------------------------
# Semiring matrix operations
# ---------------------------------------------------------------------------


class SemiringMatrix(Generic[T]):
    r"""An ``n × n`` matrix over a semiring, supporting multiplication and
    Kleene closure.

    Parameters
    ----------
    semiring : Semiring[T]
        The underlying semiring.
    data : list[list[T]]
        The matrix entries (row-major).
    """

    def __init__(self, semiring: Semiring[T], data: List[List[T]]) -> None:
        self.sr = semiring
        self.n = len(data)
        self.data = [row[:] for row in data]

    @classmethod
    def zeros(cls, semiring: Semiring[T], n: int) -> "SemiringMatrix[T]":
        """Create an ``n × n`` matrix of additive zeros."""
        z = semiring.zero()
        return cls(semiring, [[z for _ in range(n)] for _ in range(n)])

    @classmethod
    def identity(cls, semiring: Semiring[T], n: int) -> "SemiringMatrix[T]":
        """Create an ``n × n`` identity matrix (ones on diagonal, zeros elsewhere)."""
        z = semiring.zero()
        o = semiring.one()
        data = [[o if i == j else z for j in range(n)] for i in range(n)]
        return cls(semiring, data)

    def multiply(self, other: "SemiringMatrix[T]") -> "SemiringMatrix[T]":
        r"""Matrix multiplication: ``C[i,j] = ⊕_k (A[i,k] ⊗ B[k,j])``.

        This is the standard matrix multiplication with semiring operations
        replacing conventional ``+`` and ``×``.

        Complexity: ``O(n³)`` semiring operations.
        """
        n = self.n
        result = SemiringMatrix.zeros(self.sr, n)
        for i in range(n):
            for j in range(n):
                s = self.sr.zero()
                for k in range(n):
                    s = self.sr.add(s, self.sr.mul(self.data[i][k], other.data[k][j]))
                result.data[i][j] = s
        return result

    def add(self, other: "SemiringMatrix[T]") -> "SemiringMatrix[T]":
        """Element-wise semiring addition."""
        n = self.n
        result = SemiringMatrix.zeros(self.sr, n)
        for i in range(n):
            for j in range(n):
                result.data[i][j] = self.sr.add(self.data[i][j], other.data[i][j])
        return result

    def closure(self, max_iter: int = 0) -> "SemiringMatrix[T]":
        r"""Kleene star / transitive closure: ``A* = I ⊕ A ⊕ A² ⊕ A³ ⊕ …``.

        Uses the Floyd-Warshall-style algorithm which runs in ``O(n³)``
        and computes the all-pairs shortest/longest/etc. paths depending
        on the semiring.

        Parameters
        ----------
        max_iter : int
            If > 0, truncate to ``max_iter`` iterations (power series).
            If 0 (default), use Floyd-Warshall.

        Returns
        -------
        SemiringMatrix[T]
            The closure matrix.
        """
        n = self.n
        if max_iter > 0:
            return self._power_closure(max_iter)
        return self._floyd_warshall_closure()

    def _floyd_warshall_closure(self) -> "SemiringMatrix[T]":
        """Floyd-Warshall algorithm generalised to an arbitrary semiring."""
        n = self.n
        # Start from I ⊕ A
        result = SemiringMatrix.identity(self.sr, n).add(self)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    # d[i][j] = d[i][j] ⊕ (d[i][k] ⊗ d[k][j])
                    via_k = self.sr.mul(result.data[i][k], result.data[k][j])
                    result.data[i][j] = self.sr.add(result.data[i][j], via_k)
        return result

    def _power_closure(self, max_iter: int) -> "SemiringMatrix[T]":
        """Compute closure by iterated squaring up to ``max_iter`` terms."""
        n = self.n
        total = SemiringMatrix.identity(self.sr, n)
        power = SemiringMatrix(self.sr, [row[:] for row in self.data])
        for _ in range(max_iter):
            total = total.add(power)
            power = power.multiply(self)
        return total

    def __getitem__(self, idx: Tuple[int, int]) -> T:
        i, j = idx
        return self.data[i][j]

    def __repr__(self) -> str:
        rows = []
        for row in self.data:
            rows.append("[" + ", ".join(repr(x) for x in row) + "]")
        return "SemiringMatrix([\n  " + "\n  ".join(rows) + "\n])"


# ---------------------------------------------------------------------------
# Application: all-pairs cognitive cost computation
# ---------------------------------------------------------------------------


def all_pairs_cost(
    adjacency: np.ndarray,
    semiring_name: str = "tropical",
) -> np.ndarray:
    r"""Compute all-pairs cognitive cost using semiring matrix closure.

    Parameters
    ----------
    adjacency : np.ndarray
        ``n × n`` matrix of edge costs.  Use ``np.inf`` (or ``-np.inf``)
        for non-edges, depending on the semiring.
    semiring_name : str
        One of ``"tropical"`` (min cost), ``"maxplus"`` (critical path),
        ``"boolean"`` (reachability).

    Returns
    -------
    np.ndarray
        ``n × n`` matrix of all-pairs costs under the chosen semiring.

    Examples
    --------
    >>> import numpy as np
    >>> adj = np.array([[0, 3, np.inf], [np.inf, 0, 1], [np.inf, np.inf, 0]])
    >>> all_pairs_cost(adj, "tropical")
    array([[0., 3., 4.],
           [inf, 0., 1.],
           [inf, inf, 0.]])
    """
    semirings = {
        "tropical": TropicalSemiring(),
        "maxplus": MaxPlusSemiring(),
        "boolean": BooleanSemiring(),
    }
    sr_key = semiring_name.lower()
    if sr_key not in semirings:
        raise ValueError(f"Unknown semiring {semiring_name!r}. Choose from {list(semirings)}.")

    n = adjacency.shape[0]

    if sr_key == "boolean":
        sr = semirings[sr_key]  # type: ignore[assignment]
        data: List[List] = [
            [bool(adjacency[i, j] != 0 and not np.isinf(adjacency[i, j]))
             if i != j else True
             for j in range(n)]
            for i in range(n)
        ]
        mat: SemiringMatrix = SemiringMatrix(sr, data)
        result_mat = mat.closure()
        out = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                out[i, j] = 1.0 if result_mat[i, j] else 0.0
        return out

    sr = semirings[sr_key]  # type: ignore[assignment]
    data = [[float(adjacency[i, j]) for j in range(n)] for i in range(n)]
    mat = SemiringMatrix(sr, data)
    result_mat = mat.closure()
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = result_mat[i, j]  # type: ignore[assignment]
    return out


def cost_element_all_pairs(
    adjacency_mu: np.ndarray,
    adjacency_var: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""All-pairs cost computation using the expected cost semiring.

    Computes both shortest expected cost and associated variance.

    Parameters
    ----------
    adjacency_mu : np.ndarray
        ``n × n`` matrix of expected costs (use ``np.inf`` for non-edges).
    adjacency_var : np.ndarray
        ``n × n`` matrix of cost variances (use ``0`` for non-edges).

    Returns
    -------
    (result_mu, result_var) : tuple[np.ndarray, np.ndarray]
        All-pairs expected cost and variance matrices.
    """
    n = adjacency_mu.shape[0]
    sr = ExpectedCostSemiring()
    data = [
        [ExpectedCostValue(mu=float(adjacency_mu[i, j]), var=float(adjacency_var[i, j]))
         for j in range(n)]
        for i in range(n)
    ]
    mat = SemiringMatrix(sr, data)
    result_mat = mat.closure()

    out_mu = np.zeros((n, n))
    out_var = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v = result_mat[i, j]
            out_mu[i, j] = v.mu
            out_var[i, j] = v.var
    return out_mu, out_var
