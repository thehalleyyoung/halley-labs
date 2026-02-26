"""
Homology computation for cubical complexes.

Computes the homology groups H_d of cubical complexes
using Smith normal form of boundary matrices.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from .cubical import CubicalComplex, CubicalSet
from .boundary import BoundaryOperator
from .smith import SmithNormalForm


@dataclass
class HomologyResult:
    """Result of homology computation."""
    betti_numbers: List[int] = field(default_factory=list)
    torsion: List[List[int]] = field(default_factory=list)
    euler_characteristic: int = 0
    max_dimension: int = 0
    
    def __repr__(self):
        parts = []
        for d in range(len(self.betti_numbers)):
            b = self.betti_numbers[d]
            t = self.torsion[d] if d < len(self.torsion) else []
            if b > 0 or t:
                parts.append(f"H_{d}: Z^{b}" + (f" + {t}" if t else ""))
        return ", ".join(parts) if parts else "Trivial"
    
    @property
    def is_trivial(self) -> bool:
        return all(b == 0 for b in self.betti_numbers)
    
    def matches(self, other: 'HomologyResult') -> bool:
        """Check if two homology results are equal."""
        n = max(len(self.betti_numbers), len(other.betti_numbers))
        for d in range(n):
            b1 = self.betti_numbers[d] if d < len(self.betti_numbers) else 0
            b2 = other.betti_numbers[d] if d < len(other.betti_numbers) else 0
            if b1 != b2:
                return False
        return True


class HomologyComputer:
    """
    Computes homology groups of cubical complexes.
    
    H_d = ker(∂_d) / im(∂_{d+1})
    β_d = rank(ker(∂_d)) - rank(im(∂_{d+1}))
    """
    
    def __init__(self, use_smith: bool = True):
        self.use_smith = use_smith
    
    def compute(self, complex: CubicalComplex) -> HomologyResult:
        """Compute homology of a cubical complex."""
        boundary = BoundaryOperator(complex)
        assert boundary.verify_dd_zero(), "Boundary operator failed ∂∘∂=0 check"
        max_d = complex.dimension
        result = HomologyResult()
        result.max_dimension = max_d
        result.euler_characteristic = complex.euler_characteristic
        for d in range(max_d + 1):
            if self.use_smith:
                betti, torsion = self._compute_homology_smith(boundary, d, max_d)
            else:
                betti, torsion = self._compute_homology_rank(boundary, d, max_d)
            result.betti_numbers.append(betti)
            result.torsion.append(torsion)
        return result
    
    def _compute_homology_rank(self, boundary: BoundaryOperator,
                              d: int, max_d: int) -> Tuple[int, List[int]]:
        """Compute Betti number using rank computation."""
        n_d = boundary.complex.n_cells(d)
        if n_d == 0:
            return 0, []
        kernel_dim = boundary.kernel_dimension(d)
        image_dim = boundary.image_dimension(d + 1) if d < max_d else 0
        betti = max(0, kernel_dim - image_dim)
        return betti, []
    
    def _compute_homology_smith(self, boundary: BoundaryOperator,
                               d: int, max_d: int) -> Tuple[int, List[int]]:
        """Compute homology using Smith normal form."""
        n_d = boundary.complex.n_cells(d)
        if n_d == 0:
            return 0, []
        M_d = boundary.matrix(d).toarray() if d > 0 else np.zeros((0, n_d), dtype=int)
        M_dp1 = boundary.matrix(d + 1).toarray() if d < max_d else np.zeros((n_d, 0), dtype=int)
        if M_d.shape[0] > 0 and M_d.shape[1] > 0:
            snf_d = SmithNormalForm()
            snf_d.compute(M_d)
            rank_d = snf_d.rank
        else:
            rank_d = 0
        kernel_dim = n_d - rank_d
        if M_dp1.shape[0] > 0 and M_dp1.shape[1] > 0:
            snf_dp1 = SmithNormalForm()
            snf_dp1.compute(M_dp1)
            image_dim = snf_dp1.rank
            torsion = snf_dp1.torsion_coefficients
        else:
            image_dim = 0
            torsion = []
        betti = max(0, kernel_dim - image_dim)
        return betti, torsion
    
    def compute_from_grid(self, grid: np.ndarray) -> HomologyResult:
        """Compute homology from a binary grid."""
        cubical_set = CubicalSet.from_grid(grid)
        complex = CubicalComplex(cubical_set)
        return self.compute(complex)
    
    def compute_relative(self, complex: CubicalComplex,
                        subcomplex: CubicalComplex) -> HomologyResult:
        """Compute relative homology H_*(X, A)."""
        boundary_X = BoundaryOperator(complex)
        boundary_A = BoundaryOperator(subcomplex)
        max_d = complex.dimension
        result = HomologyResult()
        result.max_dimension = max_d
        for d in range(max_d + 1):
            n_X = complex.n_cells(d)
            n_A = subcomplex.n_cells(d)
            n_rel = max(0, n_X - n_A)
            if n_rel == 0:
                result.betti_numbers.append(0)
                result.torsion.append([])
                continue
            kernel_X = boundary_X.kernel_dimension(d)
            kernel_A = boundary_A.kernel_dimension(d) if n_A > 0 else 0
            image_X = boundary_X.image_dimension(d + 1) if d < max_d else 0
            image_A = boundary_A.image_dimension(d + 1) if d < max_d and n_A > 0 else 0
            betti = max(0, (kernel_X - kernel_A) - (image_X - image_A))
            result.betti_numbers.append(betti)
            result.torsion.append([])
        return result
