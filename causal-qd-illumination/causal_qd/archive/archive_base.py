"""Abstract archive interface and shared entry type for MAP-Elites."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.types import (
    AdjacencyMatrix,
    BehavioralDescriptor,
    CellIndex,
    QualityScore,
)


@dataclasses.dataclass
class ArchiveEntry:
    """A single elite stored in the archive.

    Attributes
    ----------
    solution : AdjacencyMatrix
        Adjacency matrix (or any representation) of the causal graph.
    descriptor : BehavioralDescriptor
        Behavioral descriptor vector that determines the cell.
    quality : QualityScore
        Scalar quality / fitness (higher is better).
    metadata : dict
        Arbitrary metadata attached to this entry.
    timestamp : int
        Insertion counter (monotonically increasing).
    """

    solution: AdjacencyMatrix
    descriptor: BehavioralDescriptor
    quality: QualityScore
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    timestamp: int = 0


class Archive(ABC):
    """Abstract base class for MAP-Elites archives.

    Sub-classes must implement all ``@abstractmethod`` members.

    Parameters
    ----------
    lower_bounds : npt.NDArray[np.float64]
        Per-dimension lower bounds of the descriptor space.
    upper_bounds : npt.NDArray[np.float64]
        Per-dimension upper bounds of the descriptor space.
    """

    def __init__(
        self,
        lower_bounds: npt.NDArray[np.float64],
        upper_bounds: npt.NDArray[np.float64],
    ) -> None:
        self._lower = np.asarray(lower_bounds, dtype=np.float64)
        self._upper = np.asarray(upper_bounds, dtype=np.float64)
        self._insertion_count: int = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def add(self, entry: ArchiveEntry) -> bool:
        """Insert *entry* into the archive.

        Returns ``True`` if the entry was added or replaced an existing
        inferior elite; ``False`` otherwise.
        """

    @abstractmethod
    def get(self, index: CellIndex) -> Optional[ArchiveEntry]:
        """Return the elite at *index*, or ``None``."""

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> List[ArchiveEntry]:
        """Return *n* elites sampled uniformly at random (with replacement)."""

    @abstractmethod
    def best(self) -> ArchiveEntry:
        """Return the highest-quality elite.

        Raises
        ------
        ValueError
            If the archive is empty.
        """

    @abstractmethod
    def elites(self) -> List[ArchiveEntry]:
        """Return all elites currently stored in the archive."""

    @abstractmethod
    def coverage(self) -> float:
        """Return the fraction of cells that contain an elite."""

    @abstractmethod
    def qd_score(self) -> float:
        """Return the QD-score (sum of all elite qualities)."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of occupied cells."""

    @abstractmethod
    def __iter__(self) -> Iterator[ArchiveEntry]:
        """Iterate over all elites."""

    @abstractmethod
    def __contains__(self, index: CellIndex) -> bool:
        """Return ``True`` if *index* has an elite."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all elites from the archive."""

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    @property
    def descriptor_bounds(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return ``(lower_bounds, upper_bounds)``."""
        return self._lower.copy(), self._upper.copy()
