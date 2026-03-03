"""
Forward-mode automatic differentiation via dual numbers.

Provides a full-featured ``DualNumber`` class supporting arithmetic,
transcendental functions, and higher-order jets for second derivatives.
Also includes vectorised operations via numpy integration and sparse
Jacobian computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
from scipy import sparse


# ---------------------------------------------------------------------------
# Extended DualNumber
# ---------------------------------------------------------------------------

Scalar = Union[int, float]


@dataclass
class DualNumber:
    """Forward-mode AD primitive carrying value + derivative.

    Supports full arithmetic, transcendental functions, and comparison.

    Attributes:
        value: The primal (real) value.
        derivative: The tangent (derivative) value.
    """

    value: float
    derivative: float = 0.0

    # -- Arithmetic ----------------------------------------------------------

    def __add__(self, other: Union[DualNumber, Scalar]) -> DualNumber:
        if isinstance(other, DualNumber):
            return DualNumber(self.value + other.value,
                              self.derivative + other.derivative)
        return DualNumber(self.value + float(other), self.derivative)

    def __radd__(self, other: Scalar) -> DualNumber:
        return DualNumber(float(other) + self.value, self.derivative)

    def __sub__(self, other: Union[DualNumber, Scalar]) -> DualNumber:
        if isinstance(other, DualNumber):
            return DualNumber(self.value - other.value,
                              self.derivative - other.derivative)
        return DualNumber(self.value - float(other), self.derivative)

    def __rsub__(self, other: Scalar) -> DualNumber:
        return DualNumber(float(other) - self.value, -self.derivative)

    def __mul__(self, other: Union[DualNumber, Scalar]) -> DualNumber:
        if isinstance(other, DualNumber):
            return DualNumber(
                self.value * other.value,
                self.value * other.derivative + self.derivative * other.value,
            )
        c = float(other)
        return DualNumber(self.value * c, self.derivative * c)

    def __rmul__(self, other: Scalar) -> DualNumber:
        c = float(other)
        return DualNumber(c * self.value, c * self.derivative)

    def __truediv__(self, other: Union[DualNumber, Scalar]) -> DualNumber:
        if isinstance(other, DualNumber):
            if other.value == 0.0:
                raise ZeroDivisionError("DualNumber division by zero")
            inv = 1.0 / other.value
            return DualNumber(
                self.value * inv,
                (self.derivative * other.value - self.value * other.derivative)
                * inv * inv,
            )
        c = float(other)
        if c == 0.0:
            raise ZeroDivisionError("DualNumber division by zero")
        return DualNumber(self.value / c, self.derivative / c)

    def __rtruediv__(self, other: Scalar) -> DualNumber:
        if self.value == 0.0:
            raise ZeroDivisionError("DualNumber division by zero")
        c = float(other)
        inv = 1.0 / self.value
        return DualNumber(c * inv, -c * self.derivative * inv * inv)

    def __pow__(self, other: Union[DualNumber, Scalar]) -> DualNumber:
        if isinstance(other, DualNumber):
            # f^g  =>  exp(g * ln(f))
            if self.value <= 0:
                raise ValueError("DualNumber power requires positive base")
            ln_f = math.log(self.value)
            val = self.value ** other.value
            deriv = val * (
                other.derivative * ln_f
                + other.value * self.derivative / self.value
            )
            return DualNumber(val, deriv)
        n = float(other)
        val = self.value ** n
        deriv = n * self.value ** (n - 1) * self.derivative
        return DualNumber(val, deriv)

    def __rpow__(self, other: Scalar) -> DualNumber:
        # c^self = exp(self * ln(c))
        c = float(other)
        if c <= 0:
            raise ValueError("DualNumber power requires positive base")
        ln_c = math.log(c)
        val = c ** self.value
        return DualNumber(val, val * ln_c * self.derivative)

    def __neg__(self) -> DualNumber:
        return DualNumber(-self.value, -self.derivative)

    def __abs__(self) -> DualNumber:
        if self.value > 0:
            return DualNumber(self.value, self.derivative)
        elif self.value < 0:
            return DualNumber(-self.value, -self.derivative)
        return DualNumber(0.0, 0.0)

    # -- Comparison (on primal value only) -----------------------------------

    def __lt__(self, other: Union[DualNumber, Scalar]) -> bool:
        ov = other.value if isinstance(other, DualNumber) else float(other)
        return self.value < ov

    def __le__(self, other: Union[DualNumber, Scalar]) -> bool:
        ov = other.value if isinstance(other, DualNumber) else float(other)
        return self.value <= ov

    def __gt__(self, other: Union[DualNumber, Scalar]) -> bool:
        ov = other.value if isinstance(other, DualNumber) else float(other)
        return self.value > ov

    def __ge__(self, other: Union[DualNumber, Scalar]) -> bool:
        ov = other.value if isinstance(other, DualNumber) else float(other)
        return self.value >= ov

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DualNumber):
            return self.value == other.value
        if isinstance(other, (int, float)):
            return self.value == float(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.value, self.derivative))

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"DualNumber({self.value}, d={self.derivative})"


# ---------------------------------------------------------------------------
# Transcendental functions for DualNumbers
# ---------------------------------------------------------------------------

def dual_exp(x: Union[DualNumber, float]) -> DualNumber:
    """Compute exp(x) preserving dual number derivatives."""
    if isinstance(x, DualNumber):
        val = math.exp(x.value)
        return DualNumber(val, val * x.derivative)
    return DualNumber(math.exp(float(x)), 0.0)


def dual_log(x: Union[DualNumber, float]) -> DualNumber:
    """Compute log(x) preserving dual number derivatives."""
    if isinstance(x, DualNumber):
        if x.value <= 0:
            raise ValueError("log of non-positive DualNumber")
        return DualNumber(math.log(x.value), x.derivative / x.value)
    v = float(x)
    if v <= 0:
        raise ValueError("log of non-positive value")
    return DualNumber(math.log(v), 0.0)


def dual_abs(x: Union[DualNumber, float]) -> DualNumber:
    """Compute abs(x) preserving dual number derivatives."""
    if isinstance(x, DualNumber):
        return abs(x)
    return DualNumber(abs(float(x)), 0.0)


def dual_max(a: Union[DualNumber, float],
             b: Union[DualNumber, float]) -> DualNumber:
    """Compute max(a, b), propagating the derivative of the larger operand."""
    da = a if isinstance(a, DualNumber) else DualNumber(float(a))
    db = b if isinstance(b, DualNumber) else DualNumber(float(b))
    if da.value >= db.value:
        return DualNumber(da.value, da.derivative)
    return DualNumber(db.value, db.derivative)


def dual_min(a: Union[DualNumber, float],
             b: Union[DualNumber, float]) -> DualNumber:
    """Compute min(a, b), propagating the derivative of the smaller operand."""
    da = a if isinstance(a, DualNumber) else DualNumber(float(a))
    db = b if isinstance(b, DualNumber) else DualNumber(float(b))
    if da.value <= db.value:
        return DualNumber(da.value, da.derivative)
    return DualNumber(db.value, db.derivative)


def dual_sqrt(x: Union[DualNumber, float]) -> DualNumber:
    """Compute sqrt(x) preserving dual number derivatives."""
    if isinstance(x, DualNumber):
        if x.value < 0:
            raise ValueError("sqrt of negative DualNumber")
        val = math.sqrt(x.value)
        deriv = x.derivative / (2.0 * val) if val != 0 else 0.0
        return DualNumber(val, deriv)
    v = float(x)
    return DualNumber(math.sqrt(v), 0.0)


def dual_sin(x: Union[DualNumber, float]) -> DualNumber:
    """Compute sin(x) preserving dual number derivatives."""
    if isinstance(x, DualNumber):
        return DualNumber(math.sin(x.value), math.cos(x.value) * x.derivative)
    return DualNumber(math.sin(float(x)), 0.0)


def dual_cos(x: Union[DualNumber, float]) -> DualNumber:
    """Compute cos(x) preserving dual number derivatives."""
    if isinstance(x, DualNumber):
        return DualNumber(math.cos(x.value), -math.sin(x.value) * x.derivative)
    return DualNumber(math.cos(float(x)), 0.0)


# ---------------------------------------------------------------------------
# Higher-order dual numbers (jets)
# ---------------------------------------------------------------------------

@dataclass
class JetNumber:
    """Higher-order dual number for second (and higher) derivatives.

    Stores a truncated Taylor coefficient vector [f, f', f''/2!, ...].
    Jet of order *k* carries derivatives up to order *k*.

    Attributes:
        coeffs: Taylor coefficients of length k+1.
    """

    coeffs: npt.NDArray[np.float64]

    def __init__(self, coeffs: Union[Sequence[float], npt.NDArray[np.float64]]) -> None:
        self.coeffs = np.asarray(coeffs, dtype=np.float64)

    @property
    def value(self) -> float:
        return float(self.coeffs[0])

    @property
    def first_deriv(self) -> float:
        return float(self.coeffs[1]) if len(self.coeffs) > 1 else 0.0

    @property
    def second_deriv(self) -> float:
        return float(2.0 * self.coeffs[2]) if len(self.coeffs) > 2 else 0.0

    @property
    def order(self) -> int:
        return len(self.coeffs) - 1

    @staticmethod
    def variable(value: float, order: int = 2) -> JetNumber:
        """Create a jet for an independent variable (derivative = 1)."""
        c = np.zeros(order + 1, dtype=np.float64)
        c[0] = value
        if order >= 1:
            c[1] = 1.0
        return JetNumber(c)

    @staticmethod
    def constant(value: float, order: int = 2) -> JetNumber:
        """Create a jet for a constant (all derivatives zero)."""
        c = np.zeros(order + 1, dtype=np.float64)
        c[0] = value
        return JetNumber(c)

    def _match_order(self, other: JetNumber) -> Tuple[npt.NDArray, npt.NDArray]:
        n = max(len(self.coeffs), len(other.coeffs))
        a = np.zeros(n, dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)
        a[:len(self.coeffs)] = self.coeffs
        b[:len(other.coeffs)] = other.coeffs
        return a, b

    def __add__(self, other: Union[JetNumber, Scalar]) -> JetNumber:
        if isinstance(other, JetNumber):
            a, b = self._match_order(other)
            return JetNumber(a + b)
        c = np.copy(self.coeffs)
        c[0] += float(other)
        return JetNumber(c)

    def __radd__(self, other: Scalar) -> JetNumber:
        return self.__add__(other)

    def __sub__(self, other: Union[JetNumber, Scalar]) -> JetNumber:
        if isinstance(other, JetNumber):
            a, b = self._match_order(other)
            return JetNumber(a - b)
        c = np.copy(self.coeffs)
        c[0] -= float(other)
        return JetNumber(c)

    def __rsub__(self, other: Scalar) -> JetNumber:
        c = -self.coeffs.copy()
        c[0] += float(other)
        return JetNumber(c)

    def __mul__(self, other: Union[JetNumber, Scalar]) -> JetNumber:
        if isinstance(other, JetNumber):
            a, b = self._match_order(other)
            n = len(a)
            result = np.zeros(n, dtype=np.float64)
            for k in range(n):
                for j in range(k + 1):
                    result[k] += a[j] * b[k - j]
            return JetNumber(result)
        return JetNumber(self.coeffs * float(other))

    def __rmul__(self, other: Scalar) -> JetNumber:
        return JetNumber(self.coeffs * float(other))

    def __truediv__(self, other: Union[JetNumber, Scalar]) -> JetNumber:
        if isinstance(other, JetNumber):
            a, b = self._match_order(other)
            n = len(a)
            if b[0] == 0.0:
                raise ZeroDivisionError("JetNumber division by zero")
            result = np.zeros(n, dtype=np.float64)
            for k in range(n):
                s = a[k]
                for j in range(1, k + 1):
                    s -= result[k - j] * b[j]
                result[k] = s / b[0]
            return JetNumber(result)
        if float(other) == 0.0:
            raise ZeroDivisionError("JetNumber division by zero")
        return JetNumber(self.coeffs / float(other))

    def __neg__(self) -> JetNumber:
        return JetNumber(-self.coeffs)

    def __repr__(self) -> str:
        return f"JetNumber(order={self.order}, value={self.value:.6f})"


def jet_exp(x: JetNumber) -> JetNumber:
    """Compute exp(x) for a JetNumber using the recurrence relation."""
    n = len(x.coeffs)
    result = np.zeros(n, dtype=np.float64)
    result[0] = math.exp(x.coeffs[0])
    for k in range(1, n):
        s = 0.0
        for j in range(1, k + 1):
            s += j * x.coeffs[j] * result[k - j]
        result[k] = s / k
    return JetNumber(result)


def jet_log(x: JetNumber) -> JetNumber:
    """Compute log(x) for a JetNumber using the recurrence relation."""
    n = len(x.coeffs)
    if x.coeffs[0] <= 0:
        raise ValueError("log of non-positive JetNumber")
    result = np.zeros(n, dtype=np.float64)
    result[0] = math.log(x.coeffs[0])
    for k in range(1, n):
        s = x.coeffs[k]
        for j in range(1, k):
            s -= j * result[j] * x.coeffs[k - j] / k
        result[k] = s / x.coeffs[0]
    return JetNumber(result)


# ---------------------------------------------------------------------------
# Vectorised forward-mode operations
# ---------------------------------------------------------------------------

@dataclass
class DualVector:
    """Vectorised dual numbers for batch forward-mode AD.

    Stores arrays of values and Jacobian columns for efficient
    computation over vectors.

    Attributes:
        values: Primal values array of shape (n,).
        jacobian: Jacobian matrix of shape (n, m) where m is the number
                  of independent variables being differentiated w.r.t.
    """

    values: npt.NDArray[np.float64]
    jacobian: npt.NDArray[np.float64]

    def __init__(
        self,
        values: npt.NDArray[np.float64],
        jacobian: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        self.values = np.asarray(values, dtype=np.float64)
        if jacobian is None:
            self.jacobian = np.eye(len(self.values), dtype=np.float64)
        else:
            self.jacobian = np.asarray(jacobian, dtype=np.float64)

    @staticmethod
    def identity(values: npt.NDArray[np.float64]) -> DualVector:
        """Create a DualVector seeded with the identity Jacobian."""
        v = np.asarray(values, dtype=np.float64)
        return DualVector(v, np.eye(len(v), dtype=np.float64))

    @staticmethod
    def constant(values: npt.NDArray[np.float64], n_vars: int) -> DualVector:
        """Create a constant DualVector (zero Jacobian)."""
        v = np.asarray(values, dtype=np.float64)
        return DualVector(v, np.zeros((len(v), n_vars), dtype=np.float64))

    def __add__(self, other: Union[DualVector, npt.NDArray]) -> DualVector:
        if isinstance(other, DualVector):
            return DualVector(self.values + other.values,
                              self.jacobian + other.jacobian)
        return DualVector(self.values + np.asarray(other), self.jacobian.copy())

    def __radd__(self, other: npt.NDArray) -> DualVector:
        return DualVector(np.asarray(other) + self.values, self.jacobian.copy())

    def __sub__(self, other: Union[DualVector, npt.NDArray]) -> DualVector:
        if isinstance(other, DualVector):
            return DualVector(self.values - other.values,
                              self.jacobian - other.jacobian)
        return DualVector(self.values - np.asarray(other), self.jacobian.copy())

    def __mul__(self, other: Union[DualVector, npt.NDArray, float]) -> DualVector:
        if isinstance(other, DualVector):
            new_vals = self.values * other.values
            new_jac = (self.jacobian * other.values[:, None]
                       + other.jacobian * self.values[:, None])
            return DualVector(new_vals, new_jac)
        c = np.asarray(other, dtype=np.float64)
        if c.ndim == 0:
            return DualVector(self.values * float(c), self.jacobian * float(c))
        return DualVector(self.values * c, self.jacobian * c[:, None])

    def __rmul__(self, other: Union[npt.NDArray, float]) -> DualVector:
        return self.__mul__(other)

    def __neg__(self) -> DualVector:
        return DualVector(-self.values, -self.jacobian)

    def sum(self) -> Tuple[float, npt.NDArray[np.float64]]:
        """Sum all elements, returning (scalar_value, gradient_vector)."""
        return float(np.sum(self.values)), np.sum(self.jacobian, axis=0)

    def __repr__(self) -> str:
        return f"DualVector(n={len(self.values)}, vars={self.jacobian.shape[1]})"


def dualvec_exp(dv: DualVector) -> DualVector:
    """Element-wise exp for DualVector."""
    ev = np.exp(dv.values)
    return DualVector(ev, dv.jacobian * ev[:, None])


def dualvec_log(dv: DualVector) -> DualVector:
    """Element-wise log for DualVector."""
    return DualVector(np.log(dv.values), dv.jacobian / dv.values[:, None])


def dualvec_abs(dv: DualVector) -> DualVector:
    """Element-wise abs for DualVector."""
    signs = np.sign(dv.values)
    return DualVector(np.abs(dv.values), dv.jacobian * signs[:, None])


# ---------------------------------------------------------------------------
# Sparse Jacobian computation
# ---------------------------------------------------------------------------


def compute_jacobian(
    fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute full Jacobian of fn at x via forward-mode AD.

    Uses one forward pass per input dimension, seeding the tangent
    vector with each standard basis vector.

    Args:
        fn: Function mapping R^n -> R^m.
        x: Point at which to compute the Jacobian.

    Returns:
        Jacobian matrix of shape (m, n).
    """
    n = len(x)
    columns: List[npt.NDArray[np.float64]] = []
    for j in range(n):
        duals = [DualNumber(x[i], 1.0 if i == j else 0.0) for i in range(n)]
        out = fn(np.array(duals, dtype=object))
        col = np.array([o.derivative if isinstance(o, DualNumber) else 0.0
                        for o in np.atleast_1d(out)], dtype=np.float64)
        columns.append(col)
    return np.column_stack(columns)


def compute_sparse_jacobian(
    fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x: npt.NDArray[np.float64],
    sparsity_pattern: Optional[npt.NDArray[np.bool_]] = None,
    threshold: float = 1e-12,
) -> sparse.csr_matrix:
    """Compute sparse Jacobian exploiting a known sparsity pattern.

    If *sparsity_pattern* is ``None``, computes the full Jacobian first
    and then sparsifies. When provided, uses graph colouring to reduce
    the number of forward passes.

    Args:
        fn: Function mapping R^n -> R^m.
        x: Point at which to compute the Jacobian.
        sparsity_pattern: Boolean matrix (m, n) indicating non-zero entries.
        threshold: Entries below this are treated as zero.

    Returns:
        Sparse CSR Jacobian.
    """
    if sparsity_pattern is None:
        J_dense = compute_jacobian(fn, x)
        J_dense[np.abs(J_dense) < threshold] = 0.0
        return sparse.csr_matrix(J_dense)

    n = x.shape[0]
    m = sparsity_pattern.shape[0]
    # Greedy column colouring
    colors = _greedy_column_coloring(sparsity_pattern)
    n_colors = int(np.max(colors)) + 1 if len(colors) > 0 else 0

    J = np.zeros((m, n), dtype=np.float64)
    for c in range(n_colors):
        cols_in_color = np.where(colors == c)[0]
        seed = np.zeros(n, dtype=np.float64)
        seed[cols_in_color] = 1.0
        duals = [DualNumber(x[i], seed[i]) for i in range(n)]
        out = fn(np.array(duals, dtype=object))
        compressed = np.array(
            [o.derivative if isinstance(o, DualNumber) else 0.0
             for o in np.atleast_1d(out)],
            dtype=np.float64,
        )
        for j in cols_in_color:
            rows = np.where(sparsity_pattern[:, j])[0]
            J[rows, j] = compressed[rows]

    J[np.abs(J) < threshold] = 0.0
    return sparse.csr_matrix(J)


def _greedy_column_coloring(pattern: npt.NDArray[np.bool_]) -> npt.NDArray[np.int32]:
    """Greedy graph colouring of columns for sparse Jacobian computation.

    Two columns conflict if they share a non-zero row.
    """
    m, n = pattern.shape
    colors = -np.ones(n, dtype=np.int32)
    for j in range(n):
        rows_j = set(np.where(pattern[:, j])[0])
        forbidden: set = set()
        for k in range(j):
            if colors[k] >= 0:
                rows_k = set(np.where(pattern[:, k])[0])
                if rows_j & rows_k:
                    forbidden.add(int(colors[k]))
        c = 0
        while c in forbidden:
            c += 1
        colors[j] = c
    return colors


# ---------------------------------------------------------------------------
# Utility: forward-mode derivative of scalar function
# ---------------------------------------------------------------------------


def forward_derivative(
    fn: Callable[..., Union[DualNumber, float]],
    x: float,
    *args: Any,
) -> Tuple[float, float]:
    """Compute value and derivative of a scalar function at *x*.

    Args:
        fn: Function accepting a DualNumber (or float) as first arg.
        x: Point at which to differentiate.
        *args: Extra arguments passed to fn.

    Returns:
        (value, derivative) tuple.
    """
    d = DualNumber(x, 1.0)
    result = fn(d, *args)
    if isinstance(result, DualNumber):
        return result.value, result.derivative
    return float(result), 0.0


def second_derivative(
    fn: Callable[..., Any],
    x: float,
    *args: Any,
    order: int = 2,
) -> Tuple[float, float, float]:
    """Compute value, first and second derivative via JetNumber.

    Args:
        fn: Function accepting a JetNumber as first arg.
        x: Point at which to differentiate.
        *args: Extra arguments.
        order: Jet order (default 2 for second derivative).

    Returns:
        (value, first_derivative, second_derivative).
    """
    j = JetNumber.variable(x, order=order)
    result = fn(j, *args)
    if isinstance(result, JetNumber):
        return result.value, result.first_deriv, result.second_deriv
    return float(result), 0.0, 0.0


__all__ = [
    "DualNumber",
    "JetNumber",
    "DualVector",
    "dual_exp",
    "dual_log",
    "dual_abs",
    "dual_max",
    "dual_min",
    "dual_sqrt",
    "dual_sin",
    "dual_cos",
    "jet_exp",
    "jet_log",
    "dualvec_exp",
    "dualvec_log",
    "dualvec_abs",
    "compute_jacobian",
    "compute_sparse_jacobian",
    "forward_derivative",
    "second_derivative",
]
