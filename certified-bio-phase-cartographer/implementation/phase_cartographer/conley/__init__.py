"""
Conley index computation module.

Provides cubical complex representation, boundary operator computation,
Smith normal form for homology computation, and isolating neighborhood
verification for Conley index computation.
"""

from .cubical import CubicalComplex, Cube, CubicalSet
from .boundary import BoundaryOperator
from .smith import SmithNormalForm
from .homology import HomologyComputer, HomologyResult
from .isolating import IsolatingNeighborhood, ConleyIndex

__all__ = [
    "CubicalComplex", "Cube", "CubicalSet",
    "BoundaryOperator",
    "SmithNormalForm",
    "HomologyComputer", "HomologyResult",
    "IsolatingNeighborhood", "ConleyIndex",
]
