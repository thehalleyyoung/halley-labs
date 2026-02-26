"""
Isolating neighborhood verification and Conley index computation.

Verifies that a cubical set N is an isolating neighborhood for
a flow, and computes the Conley index as the relative homology
of the index pair (N, L).
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector
from ..ode.rhs import ODERightHandSide
from .cubical import CubicalComplex, CubicalSet, Cube, ElementaryInterval
from .homology import HomologyComputer, HomologyResult


@dataclass
class ConleyIndex:
    """Conley index of an isolated invariant set."""
    homology: HomologyResult
    index_pair_N: Optional[CubicalSet] = None
    index_pair_L: Optional[CubicalSet] = None
    verified: bool = False
    classification: str = ""
    
    @property
    def is_trivial(self) -> bool:
        return self.homology.is_trivial
    
    def classify(self) -> str:
        """Classify based on Conley index."""
        if self.is_trivial:
            return "empty"
        b = self.homology.betti_numbers
        if len(b) >= 1 and b[0] == 1 and all(bi == 0 for bi in b[1:]):
            return "stable_equilibrium"
        if len(b) >= 2 and b[0] == 0 and b[1] == 1:
            if all(bi == 0 for bi in b[2:]):
                return "saddle_1"
        if len(b) >= 3 and b[0] == 0 and b[1] == 0 and b[2] == 1:
            return "unstable_equilibrium_2d"
        if len(b) >= 2 and b[0] == 1 and b[1] == 1:
            return "periodic_orbit"
        return "unknown"


class IsolatingNeighborhood:
    """
    Verifies isolating neighborhoods and computes Conley indices.
    
    A compact set N is an isolating neighborhood if the maximal
    invariant set S in N is contained in the interior of N:
    Inv(N) ⊂ int(N).
    """
    
    def __init__(self, rhs: ODERightHandSide,
                 grid_resolution: np.ndarray = None,
                 domain: IntervalVector = None):
        self.rhs = rhs
        self.grid_resolution = grid_resolution
        self.domain = domain
        self.homology_computer = HomologyComputer()
    
    def build_cubical_set(self, domain: IntervalVector,
                         resolution: np.ndarray) -> CubicalSet:
        """Build cubical representation of a domain."""
        n = domain.n
        grid_shape = tuple(int(r) for r in resolution)
        grid = np.ones(grid_shape, dtype=int)
        return CubicalSet.from_grid(grid)
    
    def verify_isolating(self, N: CubicalSet,
                        mu: np.ndarray,
                        grid_size: np.ndarray = None) -> bool:
        """
        Verify that N is an isolating neighborhood.
        
        Checks that the flow on the boundary of N points outward
        or is tangent (no trajectories can re-enter through the boundary).
        """
        if grid_size is None:
            grid_size = np.ones(self.rhs.n_states)
        boundary_cubes = self._compute_boundary(N)
        for cube in boundary_cubes:
            center = np.array(cube.center, dtype=float)
            x = center * grid_size
            if self.domain is not None:
                for i in range(len(x)):
                    x[i] = self.domain[i].lo + x[i] * self.domain[i].width / grid_size[i]
            try:
                f = self.rhs.evaluate(x, mu)
            except (ZeroDivisionError, ValueError):
                return False
            outward_normal = self._compute_outward_normal(cube, N)
            if outward_normal is not None:
                dot_product = np.dot(f[:len(outward_normal)], outward_normal)
                if dot_product < -1e-10:
                    return False
        return True
    
    def compute_index(self, domain: IntervalVector,
                     mu: np.ndarray,
                     resolution: Optional[np.ndarray] = None) -> ConleyIndex:
        """
        Compute Conley index for the invariant set in a domain.
        """
        n = self.rhs.n_states
        if resolution is None:
            resolution = np.full(n, 10)
        N = self.build_cubical_set(domain, resolution)
        L = self._compute_exit_set(N, mu, domain, resolution)
        N_complex = CubicalComplex(N)
        if L.n_cubes() > 0:
            L_complex = CubicalComplex(L)
            homology = self.homology_computer.compute_relative(N_complex, L_complex)
        else:
            homology = self.homology_computer.compute(N_complex)
        verified = self.verify_isolating(N, mu, resolution.astype(float))
        index = ConleyIndex(
            homology=homology,
            index_pair_N=N,
            index_pair_L=L,
            verified=verified
        )
        index.classification = index.classify()
        return index
    
    def _compute_exit_set(self, N: CubicalSet,
                         mu: np.ndarray,
                         domain: IntervalVector,
                         resolution: np.ndarray) -> CubicalSet:
        """Compute the exit set L ⊂ N."""
        n = domain.n
        L = CubicalSet(n)
        boundary = self._compute_boundary(N)
        grid_size = resolution.astype(float)
        for cube in boundary:
            center = np.array(cube.center, dtype=float)
            x = np.zeros(n)
            for i in range(n):
                x[i] = domain[i].lo + center[i] * domain[i].width / grid_size[i]
            try:
                f = self.rhs.evaluate(x, mu)
            except (ZeroDivisionError, ValueError):
                L.add_cube(cube)
                continue
            normal = self._compute_outward_normal(cube, N)
            if normal is not None:
                if np.dot(f[:len(normal)], normal) > 0:
                    L.add_cube(cube)
        return L
    
    def _compute_boundary(self, N: CubicalSet) -> List[Cube]:
        """Compute boundary cubes of a cubical set."""
        boundary = []
        max_d = N.max_dimension
        for cube in N.cubes_of_dimension(max_d):
            for i in range(len(cube.intervals)):
                if not cube.intervals[i].is_degenerate:
                    for val in [cube.intervals[i].lower, cube.intervals[i].upper]:
                        new_intervals = list(cube.intervals)
                        new_intervals[i] = ElementaryInterval(val, val)
                        face = Cube(tuple(new_intervals))
                        neighbor_intervals = list(cube.intervals)
                        if val == cube.intervals[i].lower:
                            test_val = val - 1
                        else:
                            test_val = val
                        neighbor_intervals[i] = ElementaryInterval(test_val, test_val + 1)
                        neighbor = Cube(tuple(neighbor_intervals))
                        if not N.contains(neighbor) or neighbor == cube:
                            boundary.append(face)
        return boundary
    
    def _compute_outward_normal(self, boundary_cube: Cube,
                               N: CubicalSet) -> Optional[np.ndarray]:
        """Compute outward-pointing normal at a boundary face."""
        n = boundary_cube.embedding_dimension
        for i in range(n):
            if boundary_cube.intervals[i].is_degenerate:
                val = boundary_cube.intervals[i].lower
                test_plus = list(boundary_cube.intervals)
                test_plus[i] = ElementaryInterval(val, val + 1)
                test_minus = list(boundary_cube.intervals)
                test_minus[i] = ElementaryInterval(val - 1, val)
                has_plus = N.contains(Cube(tuple(test_plus)))
                has_minus = N.contains(Cube(tuple(test_minus)))
                if has_plus and not has_minus:
                    normal = np.zeros(n)
                    normal[i] = -1.0
                    return normal
                elif has_minus and not has_plus:
                    normal = np.zeros(n)
                    normal[i] = 1.0
                    return normal
        return None
