"""
Interval matrix and vector operations.

Provides rigorous matrix arithmetic, interval linear system solving,
and spectral enclosure methods for validated numerics.
"""

import numpy as np
from typing import List, Tuple, Optional
from .interval import Interval


class IntervalVector:
    """Vector of intervals with element-wise arithmetic."""
    
    def __init__(self, components):
        if isinstance(components, IntervalVector):
            self.components = list(components.components)
        elif isinstance(components, np.ndarray):
            self.components = [Interval(float(x)) for x in components]
        elif isinstance(components, list):
            self.components = [
                x if isinstance(x, Interval) else Interval(float(x))
                for x in components
            ]
        else:
            raise TypeError(f"Cannot create IntervalVector from {type(components)}")
    
    @classmethod
    def zeros(cls, n: int) -> 'IntervalVector':
        """Create zero vector."""
        return cls([Interval(0.0) for _ in range(n)])
    
    @classmethod
    def from_bounds(cls, lo: np.ndarray, hi: np.ndarray) -> 'IntervalVector':
        """Create from lower and upper bound arrays."""
        return cls([Interval(l, h) for l, h in zip(lo, hi)])
    
    @classmethod
    def from_midpoint_radius(cls, mid: np.ndarray, rad: np.ndarray) -> 'IntervalVector':
        """Create from midpoint and radius arrays."""
        return cls([Interval(m - r, m + r) for m, r in zip(mid, rad)])
    
    @property
    def n(self) -> int:
        """Dimension of vector."""
        return len(self.components)
    
    def __len__(self):
        return len(self.components)
    
    def __getitem__(self, i) -> Interval:
        return self.components[i]
    
    def __setitem__(self, i, val):
        if isinstance(val, (int, float)):
            val = Interval(val)
        self.components[i] = val
    
    def __add__(self, other):
        if isinstance(other, IntervalVector):
            if self.n != other.n:
                raise ValueError("Dimension mismatch")
            return IntervalVector([a + b for a, b in zip(self.components, other.components)])
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, IntervalVector):
            if self.n != other.n:
                raise ValueError("Dimension mismatch")
            return IntervalVector([a - b for a, b in zip(self.components, other.components)])
        return NotImplemented
    
    def __neg__(self):
        return IntervalVector([-x for x in self.components])
    
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float, Interval)):
            if isinstance(scalar, (int, float)):
                scalar = Interval(scalar)
            return IntervalVector([x * scalar for x in self.components])
        return NotImplemented
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def dot(self, other: 'IntervalVector') -> Interval:
        """Interval dot product."""
        if self.n != other.n:
            raise ValueError("Dimension mismatch")
        result = Interval(0.0)
        for a, b in zip(self.components, other.components):
            result = result + a * b
        return result
    
    def norm_inf(self) -> Interval:
        """Infinity norm enclosure."""
        mags = [abs(x) for x in self.components]
        lo = max(m.lo for m in mags)
        hi = max(m.hi for m in mags)
        return Interval(lo, hi)
    
    def norm_2_upper(self) -> float:
        """Upper bound on 2-norm."""
        s = sum(x.mag ** 2 for x in self.components)
        return np.sqrt(s)
    
    def midpoint(self) -> np.ndarray:
        """Return midpoint vector."""
        return np.array([x.mid for x in self.components])
    
    def radius(self) -> np.ndarray:
        """Return radius vector."""
        return np.array([x.rad for x in self.components])
    
    def width(self) -> np.ndarray:
        """Return width vector."""
        return np.array([x.width for x in self.components])
    
    def max_width(self) -> float:
        """Maximum component width."""
        return max(x.width for x in self.components)
    
    def contains(self, other) -> bool:
        """Check if this interval vector contains another."""
        if isinstance(other, IntervalVector):
            return all(a.contains(b) for a, b in zip(self.components, other.components))
        if isinstance(other, np.ndarray):
            return all(self.components[i].contains(float(other[i])) for i in range(self.n))
        return False
    
    def hull(self, other: 'IntervalVector') -> 'IntervalVector':
        """Component-wise hull."""
        return IntervalVector([a.hull(b) for a, b in zip(self.components, other.components)])
    
    def intersection(self, other: 'IntervalVector') -> Optional['IntervalVector']:
        """Component-wise intersection."""
        result = []
        for a, b in zip(self.components, other.components):
            inter = a.intersection(b)
            if inter.is_empty():
                return None
            result.append(inter)
        return IntervalVector(result)
    
    def inflate(self, eps: float) -> 'IntervalVector':
        """Inflate each component by eps."""
        return IntervalVector([x.inflate(eps) for x in self.components])
    
    def split(self, dim: Optional[int] = None) -> Tuple['IntervalVector', 'IntervalVector']:
        """Split along widest dimension or specified dimension."""
        if dim is None:
            dim = int(np.argmax(self.width()))
        left = list(self.components)
        right = list(self.components)
        l, r = self.components[dim].split()
        left[dim] = l
        right[dim] = r
        return IntervalVector(left), IntervalVector(right)
    
    def to_list(self) -> List[Interval]:
        """Convert to list of intervals."""
        return list(self.components)
    
    def __repr__(self):
        return f"IntervalVector({self.components})"
    
    def __str__(self):
        entries = ", ".join(str(x) for x in self.components)
        return f"[{entries}]"


class IntervalMatrix:
    """Matrix of intervals with rigorous arithmetic."""
    
    def __init__(self, data):
        if isinstance(data, IntervalMatrix):
            self.rows = data.rows
            self.cols = data.cols
            self.entries = [list(row) for row in data.entries]
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                self.rows, self.cols = data.shape
                self.entries = [
                    [Interval(float(data[i, j])) for j in range(self.cols)]
                    for i in range(self.rows)
                ]
            else:
                raise ValueError("Expected 2D array")
        elif isinstance(data, list):
            self.rows = len(data)
            self.cols = len(data[0]) if data else 0
            self.entries = []
            for row in data:
                self.entries.append([
                    x if isinstance(x, Interval) else Interval(float(x))
                    for x in row
                ])
        else:
            raise TypeError(f"Cannot create IntervalMatrix from {type(data)}")
    
    @classmethod
    def identity(cls, n: int) -> 'IntervalMatrix':
        """Create n x n identity matrix."""
        entries = [
            [Interval(1.0) if i == j else Interval(0.0) for j in range(n)]
            for i in range(n)
        ]
        return cls(entries)
    
    @classmethod
    def zeros(cls, rows: int, cols: int) -> 'IntervalMatrix':
        """Create zero matrix."""
        entries = [
            [Interval(0.0) for _ in range(cols)]
            for _ in range(rows)
        ]
        return cls(entries)
    
    @classmethod
    def diagonal(cls, diag: List[Interval]) -> 'IntervalMatrix':
        """Create diagonal matrix."""
        n = len(diag)
        entries = [
            [diag[i] if i == j else Interval(0.0) for j in range(n)]
            for i in range(n)
        ]
        return cls(entries)
    
    @classmethod
    def from_numpy(cls, A: np.ndarray) -> 'IntervalMatrix':
        """Create from numpy array."""
        return cls(A)
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            i, j = idx
            return self.entries[i][j]
        return self.entries[idx]
    
    def __setitem__(self, idx, val):
        if isinstance(idx, tuple):
            i, j = idx
            if isinstance(val, (int, float)):
                val = Interval(val)
            self.entries[i][j] = val
        else:
            self.entries[idx] = val
    
    def __add__(self, other):
        if isinstance(other, IntervalMatrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Shape mismatch")
            entries = [
                [self.entries[i][j] + other.entries[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
            return IntervalMatrix(entries)
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, IntervalMatrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Shape mismatch")
            entries = [
                [self.entries[i][j] - other.entries[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
            return IntervalMatrix(entries)
        return NotImplemented
    
    def __neg__(self):
        entries = [
            [-self.entries[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return IntervalMatrix(entries)
    
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float, Interval)):
            if isinstance(scalar, (int, float)):
                scalar = Interval(scalar)
            entries = [
                [self.entries[i][j] * scalar for j in range(self.cols)]
                for i in range(self.rows)
            ]
            return IntervalMatrix(entries)
        return NotImplemented
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def matmul(self, other):
        """Matrix multiplication."""
        if isinstance(other, IntervalMatrix):
            if self.cols != other.rows:
                raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")
            entries = []
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    s = Interval(0.0)
                    for k in range(self.cols):
                        s = s + self.entries[i][k] * other.entries[k][j]
                    row.append(s)
                entries.append(row)
            return IntervalMatrix(entries)
        if isinstance(other, IntervalVector):
            if self.cols != other.n:
                raise ValueError("Dimension mismatch")
            result = []
            for i in range(self.rows):
                s = Interval(0.0)
                for j in range(self.cols):
                    s = s + self.entries[i][j] * other[j]
                result.append(s)
            return IntervalVector(result)
        return NotImplemented
    
    def __matmul__(self, other):
        return self.matmul(other)
    
    def transpose(self) -> 'IntervalMatrix':
        """Transpose."""
        entries = [
            [self.entries[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ]
        return IntervalMatrix(entries)
    
    @property
    def T(self) -> 'IntervalMatrix':
        return self.transpose()
    
    def midpoint_matrix(self) -> np.ndarray:
        """Return midpoint matrix as numpy array."""
        return np.array([
            [self.entries[i][j].mid for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def radius_matrix(self) -> np.ndarray:
        """Return radius matrix as numpy array."""
        return np.array([
            [self.entries[i][j].rad for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def width_matrix(self) -> np.ndarray:
        """Return width matrix."""
        return np.array([
            [self.entries[i][j].width for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def norm_inf(self) -> float:
        """Infinity norm upper bound."""
        row_sums = []
        for i in range(self.rows):
            s = sum(self.entries[i][j].mag for j in range(self.cols))
            row_sums.append(s)
        return max(row_sums)
    
    def norm_1(self) -> float:
        """1-norm upper bound."""
        col_sums = []
        for j in range(self.cols):
            s = sum(self.entries[i][j].mag for i in range(self.rows))
            col_sums.append(s)
        return max(col_sums)
    
    def spectral_radius_bound(self) -> float:
        """Upper bound on spectral radius using Gershgorin circles."""
        if self.rows != self.cols:
            raise ValueError("Spectral radius requires square matrix")
        max_r = 0.0
        for i in range(self.rows):
            center_mag = self.entries[i][i].mag
            row_sum = sum(
                self.entries[i][j].mag for j in range(self.cols) if j != i
            )
            max_r = max(max_r, center_mag + row_sum)
        return max_r
    
    def gershgorin_disks(self) -> List[Tuple[Interval, float]]:
        """Compute Gershgorin disk centers and radii."""
        if self.rows != self.cols:
            raise ValueError("Requires square matrix")
        disks = []
        for i in range(self.rows):
            center = self.entries[i][i]
            radius = sum(
                self.entries[i][j].mag for j in range(self.cols) if j != i
            )
            disks.append((center, radius))
        return disks
    
    def is_m_matrix(self) -> bool:
        """Check if matrix is an M-matrix (positive diagonal, non-positive off-diagonal)."""
        if self.rows != self.cols:
            return False
        for i in range(self.rows):
            if self.entries[i][i].lo <= 0:
                return False
            for j in range(self.cols):
                if i != j and self.entries[i][j].hi > 0:
                    return False
        return True
    
    def is_strictly_diagonally_dominant(self) -> bool:
        """Check strict diagonal dominance."""
        if self.rows != self.cols:
            return False
        for i in range(self.rows):
            diag = abs(self.entries[i][i]).lo
            off_diag_sum = sum(
                abs(self.entries[i][j]).hi for j in range(self.cols) if j != i
            )
            if diag <= off_diag_sum:
                return False
        return True
    
    def preconditioned(self, R: np.ndarray) -> 'IntervalMatrix':
        """Precondition: R @ self, where R is a point matrix."""
        R_iv = IntervalMatrix.from_numpy(R)
        return R_iv.matmul(self)
    
    def column(self, j: int) -> IntervalVector:
        """Extract column j as interval vector."""
        return IntervalVector([self.entries[i][j] for i in range(self.rows)])
    
    def row(self, i: int) -> IntervalVector:
        """Extract row i as interval vector."""
        return IntervalVector(self.entries[i])
    
    def set_column(self, j: int, v: IntervalVector):
        """Set column j from interval vector."""
        for i in range(self.rows):
            self.entries[i][j] = v[i]
    
    def set_row(self, i: int, v: IntervalVector):
        """Set row i from interval vector."""
        for j in range(self.cols):
            self.entries[i][j] = v[j]
    
    def submatrix(self, row_start: int, row_end: int,
                  col_start: int, col_end: int) -> 'IntervalMatrix':
        """Extract submatrix."""
        entries = [
            [self.entries[i][j] for j in range(col_start, col_end)]
            for i in range(row_start, row_end)
        ]
        return IntervalMatrix(entries)
    
    def determinant(self) -> Interval:
        """Compute interval determinant for small matrices."""
        if self.rows != self.cols:
            raise ValueError("Determinant requires square matrix")
        n = self.rows
        if n == 1:
            return self.entries[0][0]
        if n == 2:
            return (self.entries[0][0] * self.entries[1][1] -
                    self.entries[0][1] * self.entries[1][0])
        if n == 3:
            a, b, c = self.entries[0]
            d, e, f = self.entries[1]
            g, h, k = self.entries[2]
            return (a * (e * k - f * h) -
                    b * (d * k - f * g) +
                    c * (d * h - e * g))
        det = Interval(0.0)
        for j in range(n):
            minor = self._minor(0, j)
            sign = Interval(1.0) if j % 2 == 0 else Interval(-1.0)
            det = det + sign * self.entries[0][j] * minor.determinant()
        return det
    
    def _minor(self, row: int, col: int) -> 'IntervalMatrix':
        """Compute minor matrix by removing row and column."""
        entries = []
        for i in range(self.rows):
            if i == row:
                continue
            r = []
            for j in range(self.cols):
                if j == col:
                    continue
                r.append(self.entries[i][j])
            entries.append(r)
        return IntervalMatrix(entries)
    
    def trace(self) -> Interval:
        """Compute trace."""
        if self.rows != self.cols:
            raise ValueError("Trace requires square matrix")
        result = Interval(0.0)
        for i in range(self.rows):
            result = result + self.entries[i][i]
        return result
    
    def __repr__(self):
        return f"IntervalMatrix({self.rows}x{self.cols})"
    
    def __str__(self):
        lines = []
        for i in range(self.rows):
            entries = ", ".join(str(self.entries[i][j]) for j in range(self.cols))
            lines.append(f"  [{entries}]")
        return "[\n" + "\n".join(lines) + "\n]"


def interval_gauss_seidel(A: IntervalMatrix, b: IntervalVector,
                          x0: IntervalVector,
                          max_iter: int = 100,
                          tol: float = 1e-12) -> IntervalVector:
    """
    Interval Gauss-Seidel iteration for solving Ax = b.
    Returns an enclosure of all solutions within x0.
    """
    n = A.rows
    x = IntervalVector(list(x0.components))
    for iteration in range(max_iter):
        x_old = IntervalVector(list(x.components))
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s = s - A[i, j] * x[j]
            if A[i, i].contains(0.0):
                continue
            new_xi = s / A[i, i]
            x[i] = x[i].intersection(new_xi)
            if x[i].is_empty():
                return x
        max_change = max(
            abs(x[i].mid - x_old[i].mid) for i in range(n)
        )
        if max_change < tol:
            break
    return x


def interval_hull_solve(A: IntervalMatrix, b: IntervalVector) -> IntervalVector:
    """
    Compute an enclosure for solutions of Ax = b using preconditioned
    interval Gauss-Seidel with midpoint inverse preconditioning.
    """
    n = A.rows
    A_mid = A.midpoint_matrix()
    try:
        R = np.linalg.inv(A_mid)
    except np.linalg.LinAlgError:
        mag = A.norm_inf()
        return IntervalVector([Interval(-mag, mag) for _ in range(n)])
    RA = A.preconditioned(R)
    R_iv = IntervalMatrix.from_numpy(R)
    Rb = R_iv.matmul(b)
    x0_mid = np.linalg.solve(A_mid, b.midpoint())
    x0_rad = np.ones(n) * max(1.0, np.max(np.abs(x0_mid)))
    x0 = IntervalVector.from_midpoint_radius(x0_mid, x0_rad)
    return interval_gauss_seidel(RA, Rb, x0)


def verified_linear_solve(A: IntervalMatrix, b: IntervalVector,
                          eps: float = 1e-8) -> Tuple[IntervalVector, bool]:
    """
    Verified linear system solve with existence/uniqueness check.
    Returns (solution_enclosure, is_verified).
    """
    n = A.rows
    A_mid = A.midpoint_matrix()
    try:
        R = np.linalg.inv(A_mid)
    except np.linalg.LinAlgError:
        return IntervalVector.zeros(n), False
    I = IntervalMatrix.identity(n)
    R_iv = IntervalMatrix.from_numpy(R)
    C = I - R_iv.matmul(A)
    c_norm = C.norm_inf()
    if c_norm >= 1.0:
        return IntervalVector.zeros(n), False
    x_approx = np.linalg.solve(A_mid, b.midpoint())
    Rb = R_iv.matmul(b)
    residual = Rb - IntervalVector(x_approx)
    correction_bound = 1.0 / (1.0 - c_norm)
    r = IntervalVector([
        Interval(-correction_bound * abs(residual[i]).hi - eps,
                 correction_bound * abs(residual[i]).hi + eps)
        for i in range(n)
    ])
    x_enc = IntervalVector([
        Interval(x_approx[i] + r[i].lo, x_approx[i] + r[i].hi)
        for i in range(n)
    ])
    return x_enc, True


def eigenvalue_enclosure(A: IntervalMatrix) -> List[Interval]:
    """
    Compute enclosures for eigenvalues using Gershgorin circles.
    Returns list of interval enclosures, one per Gershgorin disk.
    """
    n = A.rows
    disks = A.gershgorin_disks()
    enclosures = []
    for center, radius in disks:
        enclosures.append(Interval(center.lo - radius, center.hi + radius))
    merged = _merge_overlapping(enclosures)
    return merged


def _merge_overlapping(intervals: List[Interval]) -> List[Interval]:
    """Merge overlapping intervals."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x.lo)
    merged = [sorted_ivs[0]]
    for iv in sorted_ivs[1:]:
        if iv.lo <= merged[-1].hi:
            merged[-1] = merged[-1].hull(iv)
        else:
            merged.append(iv)
    return merged
