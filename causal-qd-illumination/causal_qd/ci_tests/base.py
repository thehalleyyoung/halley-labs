"""Base class for conditional independence tests."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from causal_qd.types import DataMatrix, PValue


class CITest(ABC):
    """Abstract conditional independence test."""

    @abstractmethod
    def test(self, data: DataMatrix, x: int, y: int, z: List[int]) -> PValue:
        """Test X ⊥ Y | Z and return a p-value."""

    def __call__(self, data: DataMatrix, x: int, y: int, z: List[int]) -> PValue:
        return self.test(data, x, y, z)
