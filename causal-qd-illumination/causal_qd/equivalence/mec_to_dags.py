"""Enumerate or sample DAGs from an MEC given its CPDAG."""
from __future__ import annotations

from typing import List

import numpy as np

from causal_qd.core.dag import DAG
from causal_qd.mec.enumerator import MECEnumerator
from causal_qd.types import AdjacencyMatrix


class MECtoDAGs:
    """Recover DAGs from a CPDAG (MEC representative).

    Wraps :class:`~causal_qd.mec.enumerator.MECEnumerator` with a
    simpler list-returning interface.
    """

    def __init__(self) -> None:
        self._enumerator = MECEnumerator()

    def enumerate_all(self, cpdag: AdjacencyMatrix) -> List[DAG]:
        """Enumerate every DAG in the MEC represented by *cpdag*.

        .. warning::

            The number of DAGs can grow super-exponentially with the
            number of reversible edges.  Use only for small CPDAGs.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            CPDAG adjacency matrix.

        Returns
        -------
        List[DAG]
        """
        return list(self._enumerator.enumerate(cpdag))

    def sample_uniform(
        self,
        cpdag: AdjacencyMatrix,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> List[DAG]:
        """Sample *n* DAGs approximately uniformly from the MEC.

        Parameters
        ----------
        cpdag : AdjacencyMatrix
        n : int
            Number of DAGs to sample.
        rng : numpy.random.Generator or None
            Random number generator.

        Returns
        -------
        List[DAG]
        """
        return self._enumerator.sample(cpdag, n, rng)
