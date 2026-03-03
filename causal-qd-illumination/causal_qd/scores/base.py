"""Base class for score functions used in score-based causal discovery."""
from __future__ import annotations

import warnings

warnings.warn(
    "causal_qd.scores.base is deprecated. Use causal_qd.scores.score_base instead.",
    DeprecationWarning,
    stacklevel=2,
)

from abc import ABC, abstractmethod
from typing import List

from causal_qd.types import DataMatrix


class ScoreFunction(ABC):
    """Abstract scoring function for evaluating DAG structures."""

    @abstractmethod
    def local_score(self, data: DataMatrix, node: int, parents: List[int]) -> float:
        """Compute the local score of *node* given *parents*."""

    def score_dag(self, data: DataMatrix, adjacency: "AdjacencyMatrix") -> float:  # noqa: F821
        """Score an entire DAG as the sum of local scores."""
        import numpy as np

        n = adjacency.shape[0]
        total = 0.0
        for j in range(n):
            pa = sorted(int(i) for i in np.nonzero(adjacency[:, j])[0])
            total += self.local_score(data, j, pa)
        return total
