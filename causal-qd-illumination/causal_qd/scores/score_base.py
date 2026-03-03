"""Base classes for scoring functions."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from causal_qd.types import AdjacencyMatrix, DataMatrix, QualityScore


class ScoreFunction(ABC):
    """Abstract base class for DAG scoring functions."""

    @abstractmethod
    def score(self, dag: AdjacencyMatrix, data: DataMatrix) -> QualityScore:
        """Compute the quality score of *dag* given *data*.

        Parameters
        ----------
        dag:
            Adjacency matrix of the candidate DAG.
        data:
            N × p observed data matrix.

        Returns
        -------
        QualityScore
            Scalar score (higher is better by convention).
        """


class DecomposableScore(ScoreFunction, ABC):
    """A score that decomposes as a sum of per-node local scores.

    Subclasses implement :meth:`local_score` which evaluates the fitness of
    a single node given its parent set.  The full DAG score is the sum of
    local scores across all nodes.
    """

    @abstractmethod
    def local_score(
        self,
        node: int,
        parents: list[int],
        data: DataMatrix,
    ) -> float:
        """Compute the local score for *node* given *parents*.

        Parameters
        ----------
        node:
            Index of the child node.
        parents:
            Indices of the parent nodes.
        data:
            N × p data matrix.

        Returns
        -------
        float
            Local score contribution.
        """

    def score(self, dag: AdjacencyMatrix, data: DataMatrix) -> QualityScore:
        """Sum of local scores over all nodes."""
        n = dag.shape[0]
        total = 0.0
        for j in range(n):
            parents = list(np.where(dag[:, j])[0])
            total += self.local_score(j, parents, data)
        return total
