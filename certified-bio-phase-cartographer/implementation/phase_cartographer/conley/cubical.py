"""
Cubical complex representation for Conley index computation.

Implements elementary cubes, cubical sets, and cubical complexes
for computing the topology of isolating neighborhoods.
"""

import numpy as np
from typing import List, Tuple, Optional, Set, FrozenSet, Dict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ElementaryInterval:
    """An elementary interval: either degenerate [k, k] or non-degenerate [k, k+1]."""
    lower: int
    upper: int
    
    @property
    def is_degenerate(self) -> bool:
        return self.lower == self.upper
    
    @property
    def dimension(self) -> int:
        return 0 if self.is_degenerate else 1
    
    def __repr__(self):
        if self.is_degenerate:
            return f"[{self.lower}]"
        return f"[{self.lower},{self.upper}]"


@dataclass(frozen=True)
class Cube:
    """
    Elementary cube: product of elementary intervals.
    Q = I_1 x I_2 x ... x I_n
    """
    intervals: Tuple[ElementaryInterval, ...]
    
    @property
    def embedding_dimension(self) -> int:
        return len(self.intervals)
    
    @property
    def dimension(self) -> int:
        return sum(1 for iv in self.intervals if not iv.is_degenerate)
    
    @property
    def vertices(self) -> List[Tuple[int, ...]]:
        """Get all vertices of the cube."""
        verts = [[]]
        for iv in self.intervals:
            if iv.is_degenerate:
                verts = [v + [iv.lower] for v in verts]
            else:
                verts = [v + [iv.lower] for v in verts] + [v + [iv.upper] for v in verts]
        return [tuple(v) for v in verts]
    
    @property
    def center(self) -> Tuple[float, ...]:
        """Center of the cube."""
        return tuple((iv.lower + iv.upper) / 2.0 for iv in self.intervals)
    
    def contains_point(self, point: Tuple[int, ...]) -> bool:
        """Check if point is in the cube."""
        if len(point) != self.embedding_dimension:
            return False
        for p, iv in zip(point, self.intervals):
            if p < iv.lower or p > iv.upper:
                return False
        return True
    
    @classmethod
    def from_coordinates(cls, lower: Tuple[int, ...],
                        upper: Tuple[int, ...]) -> 'Cube':
        """Create cube from lower and upper coordinates."""
        intervals = tuple(
            ElementaryInterval(l, u) for l, u in zip(lower, upper)
        )
        return cls(intervals)
    
    @classmethod
    def degenerate(cls, point: Tuple[int, ...]) -> 'Cube':
        """Create degenerate (vertex) cube."""
        intervals = tuple(ElementaryInterval(p, p) for p in point)
        return cls(intervals)
    
    def __repr__(self):
        return " x ".join(repr(iv) for iv in self.intervals)


class CubicalSet:
    """
    Set of elementary cubes forming a cubical set.
    """
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.cubes: Dict[int, Set[Cube]] = {}
    
    def add_cube(self, cube: Cube):
        """Add a cube to the set."""
        d = cube.dimension
        if d not in self.cubes:
            self.cubes[d] = set()
        self.cubes[d].add(cube)
    
    def remove_cube(self, cube: Cube):
        """Remove a cube from the set."""
        d = cube.dimension
        if d in self.cubes:
            self.cubes[d].discard(cube)
    
    def cubes_of_dimension(self, d: int) -> Set[Cube]:
        """Get all cubes of dimension d."""
        return self.cubes.get(d, set())
    
    def n_cubes(self, d: Optional[int] = None) -> int:
        """Number of cubes, optionally of dimension d."""
        if d is not None:
            return len(self.cubes.get(d, set()))
        return sum(len(cubes) for cubes in self.cubes.values())
    
    @property
    def max_dimension(self) -> int:
        """Maximum dimension of any cube."""
        return max(self.cubes.keys()) if self.cubes else -1
    
    @property
    def all_cubes(self) -> Set[Cube]:
        """All cubes in the set."""
        result = set()
        for cubes in self.cubes.values():
            result.update(cubes)
        return result
    
    def contains(self, cube: Cube) -> bool:
        """Check if cube is in the set."""
        d = cube.dimension
        return cube in self.cubes.get(d, set())
    
    def union(self, other: 'CubicalSet') -> 'CubicalSet':
        """Set union."""
        result = CubicalSet(self.embedding_dim)
        for d, cubes in self.cubes.items():
            for c in cubes:
                result.add_cube(c)
        for d, cubes in other.cubes.items():
            for c in cubes:
                result.add_cube(c)
        return result
    
    def intersection(self, other: 'CubicalSet') -> 'CubicalSet':
        """Set intersection."""
        result = CubicalSet(self.embedding_dim)
        for d in self.cubes:
            if d in other.cubes:
                for c in self.cubes[d]:
                    if c in other.cubes[d]:
                        result.add_cube(c)
        return result
    
    def difference(self, other: 'CubicalSet') -> 'CubicalSet':
        """Set difference."""
        result = CubicalSet(self.embedding_dim)
        for d, cubes in self.cubes.items():
            for c in cubes:
                if not other.contains(c):
                    result.add_cube(c)
        return result
    
    def closure(self) -> 'CubicalSet':
        """Compute closure: add all faces of all cubes."""
        result = CubicalSet(self.embedding_dim)
        for cubes in self.cubes.values():
            for cube in cubes:
                for face in _all_faces(cube):
                    result.add_cube(face)
        return result
    
    @classmethod
    def from_grid(cls, grid: np.ndarray) -> 'CubicalSet':
        """Create cubical set from binary grid."""
        dim = grid.ndim
        result = cls(dim)
        for idx in np.argwhere(grid > 0):
            lower = tuple(idx)
            upper = tuple(i + 1 for i in idx)
            cube = Cube.from_coordinates(lower, upper)
            result.add_cube(cube)
        return result


def _all_faces(cube: Cube) -> List[Cube]:
    """Generate all faces of a cube."""
    faces = [cube]
    for i in range(len(cube.intervals)):
        if not cube.intervals[i].is_degenerate:
            for val in [cube.intervals[i].lower, cube.intervals[i].upper]:
                new_intervals = list(cube.intervals)
                new_intervals[i] = ElementaryInterval(val, val)
                face = Cube(tuple(new_intervals))
                faces.append(face)
                faces.extend(_all_faces(face))
    return list(set(faces))


class CubicalComplex:
    """
    Full cubical complex with boundary structure.
    """
    
    def __init__(self, cubical_set: CubicalSet):
        self.cubical_set = cubical_set.closure()
        self._cube_index: Dict[int, Dict[Cube, int]] = {}
        self._build_index()
    
    def _build_index(self):
        """Build indexing for cubes."""
        for d in range(self.cubical_set.max_dimension + 1):
            cubes = sorted(self.cubical_set.cubes_of_dimension(d),
                         key=lambda c: c.intervals)
            self._cube_index[d] = {c: i for i, c in enumerate(cubes)}
    
    def cube_index(self, cube: Cube) -> int:
        """Get index of a cube."""
        d = cube.dimension
        if d in self._cube_index and cube in self._cube_index[d]:
            return self._cube_index[d][cube]
        return -1
    
    def indexed_cubes(self, d: int) -> List[Cube]:
        """Get cubes of dimension d in index order."""
        if d not in self._cube_index:
            return []
        inv = {v: k for k, v in self._cube_index[d].items()}
        return [inv[i] for i in range(len(inv))]
    
    def n_cells(self, d: int) -> int:
        """Number of d-cells."""
        return len(self._cube_index.get(d, {}))
    
    @property
    def dimension(self) -> int:
        return self.cubical_set.max_dimension
    
    @property
    def euler_characteristic(self) -> int:
        """Compute Euler characteristic."""
        chi = 0
        for d in range(self.dimension + 1):
            chi += (-1) ** d * self.n_cells(d)
        return chi
