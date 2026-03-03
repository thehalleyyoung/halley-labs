"""Base class for behavioral descriptor computation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import AdjacencyMatrix, BehavioralDescriptor, DataMatrix


class DescriptorComputer(ABC):
    """Abstract base class for behavioral descriptor computation.

    Subclasses implement :meth:`compute` to map a DAG (and optionally
    observed data) to a fixed-length descriptor vector used by the
    MAP-Elites archive for niche assignment.
    """

    @abstractmethod
    def compute(
        self, dag: AdjacencyMatrix, data: Optional[DataMatrix] = None
    ) -> BehavioralDescriptor:
        """Compute the behavioral descriptor for *dag*.

        Parameters
        ----------
        dag:
            Adjacency matrix of the directed acyclic graph.
        data:
            Optional N × p data matrix.  Some descriptors require observed
            data (e.g., information-theoretic features).

        Returns
        -------
        BehavioralDescriptor
            A 1-D float64 array of length :pyattr:`descriptor_dim`.
        """

    @property
    @abstractmethod
    def descriptor_dim(self) -> int:
        """Dimensionality of the descriptor vector."""

    @property
    @abstractmethod
    def descriptor_bounds(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Per-dimension lower and upper bounds as ``(low, high)`` arrays."""
