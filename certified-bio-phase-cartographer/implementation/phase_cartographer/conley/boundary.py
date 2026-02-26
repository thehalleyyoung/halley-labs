"""
Boundary operator computation for cubical complexes.

Computes the boundary operator ∂_d: C_d -> C_{d-1}
for chains of elementary cubes.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy import sparse

from .cubical import Cube, ElementaryInterval, CubicalComplex


class BoundaryOperator:
    """
    Computes and stores boundary operators for a cubical complex.
    
    The boundary of an elementary cube Q = I_1 x ... x I_n is:
    ∂Q = sum_i (-1)^{sigma(i)} (Q with I_i replaced by its boundary)
    """
    
    def __init__(self, complex: CubicalComplex):
        self.complex = complex
        self._matrices: Dict[int, sparse.csr_matrix] = {}
        self._compute_all()
    
    def _compute_all(self):
        """Compute all boundary matrices."""
        for d in range(1, self.complex.dimension + 1):
            self._matrices[d] = self._compute_boundary_matrix(d)
    
    def _compute_boundary_matrix(self, d: int) -> sparse.csr_matrix:
        """Compute the boundary matrix ∂_d."""
        n_d = self.complex.n_cells(d)
        n_dm1 = self.complex.n_cells(d - 1)
        if n_d == 0 or n_dm1 == 0:
            return sparse.csr_matrix((n_dm1, n_d), dtype=int)
        rows, cols, data = [], [], []
        d_cubes = self.complex.indexed_cubes(d)
        for j, cube in enumerate(d_cubes):
            boundary_faces = self._boundary_of_cube(cube)
            for face, sign in boundary_faces:
                i = self.complex.cube_index(face)
                if i >= 0:
                    rows.append(i)
                    cols.append(j)
                    data.append(sign)
        return sparse.csr_matrix((data, (rows, cols)),
                                shape=(n_dm1, n_d), dtype=int)
    
    def _boundary_of_cube(self, cube: Cube) -> List[Tuple[Cube, int]]:
        """Compute boundary faces of a cube with orientations."""
        faces = []
        sigma = 0
        for i in range(len(cube.intervals)):
            iv = cube.intervals[i]
            if not iv.is_degenerate:
                lower_face = list(cube.intervals)
                lower_face[i] = ElementaryInterval(iv.lower, iv.lower)
                faces.append((Cube(tuple(lower_face)), (-1) ** sigma))
                upper_face = list(cube.intervals)
                upper_face[i] = ElementaryInterval(iv.upper, iv.upper)
                faces.append((Cube(tuple(upper_face)), (-1) ** (sigma + 1)))
                sigma += 1
        return faces
    
    def matrix(self, d: int) -> sparse.csr_matrix:
        """Get boundary matrix ∂_d."""
        return self._matrices.get(d, sparse.csr_matrix((0, 0), dtype=int))
    
    def verify_dd_zero(self) -> bool:
        """Verify ∂_{d-1} ∘ ∂_d = 0 for all d."""
        for d in range(2, self.complex.dimension + 1):
            if d in self._matrices and d - 1 in self._matrices:
                product = self._matrices[d - 1] @ self._matrices[d]
                if product.nnz > 0:
                    if np.max(np.abs(product.toarray())) > 0:
                        return False
        return True
    
    def kernel_dimension(self, d: int) -> int:
        """Dimension of kernel of ∂_d."""
        if d not in self._matrices:
            return self.complex.n_cells(d)
        M = self._matrices[d].toarray()
        return M.shape[1] - np.linalg.matrix_rank(M)
    
    def image_dimension(self, d: int) -> int:
        """Dimension of image of ∂_d."""
        if d not in self._matrices:
            return 0
        M = self._matrices[d].toarray()
        return np.linalg.matrix_rank(M)
