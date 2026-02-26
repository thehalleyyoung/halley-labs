"""
Smith normal form computation for integer matrices.

Used in homology computation to determine the rank and
torsion of chain complex boundary operators.
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy import sparse


class SmithNormalForm:
    """
    Computes the Smith normal form of an integer matrix.
    
    For integer matrix A, find invertible integer matrices U, V
    such that U @ A @ V = D where D is diagonal with
    d_1 | d_2 | ... | d_r (divisibility condition).
    """
    
    def __init__(self):
        self._U = None
        self._V = None
        self._D = None
    
    def compute(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Smith normal form.
        Returns (D, U, V) where U @ A @ V = D.
        """
        A = np.array(A, dtype=np.int64)
        m, n = A.shape
        D = A.copy()
        U = np.eye(m, dtype=np.int64)
        V = np.eye(n, dtype=np.int64)
        min_dim = min(m, n)
        for k in range(min_dim):
            pivot_row, pivot_col = self._find_pivot(D, k)
            if pivot_row is None:
                break
            if pivot_row != k:
                D[[k, pivot_row]] = D[[pivot_row, k]]
                U[[k, pivot_row]] = U[[pivot_row, k]]
            if pivot_col != k:
                D[:, [k, pivot_col]] = D[:, [pivot_col, k]]
                V[:, [k, pivot_col]] = V[:, [pivot_col, k]]
            if D[k, k] < 0:
                D[k, :] *= -1
                U[k, :] *= -1
            changed = True
            max_iter = 100
            iteration = 0
            while changed and iteration < max_iter:
                changed = False
                iteration += 1
                for i in range(k + 1, m):
                    if D[i, k] != 0:
                        q = D[i, k] // D[k, k]
                        D[i, :] -= q * D[k, :]
                        U[i, :] -= q * U[k, :]
                        if D[i, k] != 0:
                            D[[k, i]] = D[[i, k]]
                            U[[k, i]] = U[[i, k]]
                            changed = True
                            break
                for j in range(k + 1, n):
                    if D[k, j] != 0:
                        q = D[k, j] // D[k, k]
                        D[:, j] -= q * D[:, k]
                        V[:, j] -= q * V[:, k]
                        if D[k, j] != 0:
                            D[:, [k, j]] = D[:, [j, k]]
                            V[:, [k, j]] = V[:, [j, k]]
                            changed = True
                            break
        self._D = D
        self._U = U
        self._V = V
        return D, U, V
    
    def _find_pivot(self, D: np.ndarray, k: int) -> Tuple[Optional[int], Optional[int]]:
        """Find smallest non-zero entry in submatrix D[k:, k:]."""
        m, n = D.shape
        min_val = float('inf')
        best_row, best_col = None, None
        for i in range(k, m):
            for j in range(k, n):
                if D[i, j] != 0 and abs(D[i, j]) < min_val:
                    min_val = abs(D[i, j])
                    best_row, best_col = i, j
        return best_row, best_col
    
    @property
    def diagonal(self) -> List[int]:
        """Get the diagonal entries (invariant factors)."""
        if self._D is None:
            return []
        min_dim = min(self._D.shape)
        return [int(self._D[i, i]) for i in range(min_dim) if self._D[i, i] != 0]
    
    @property
    def rank(self) -> int:
        """Rank of the matrix."""
        return len(self.diagonal)
    
    @property
    def torsion_coefficients(self) -> List[int]:
        """Get torsion coefficients (diagonal entries > 1)."""
        return [d for d in self.diagonal if d > 1]
