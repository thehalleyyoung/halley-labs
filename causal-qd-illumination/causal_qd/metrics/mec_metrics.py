"""Markov Equivalence Class (MEC) recall metric."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from causal_qd.utils.graph_utils import skeleton

if TYPE_CHECKING:
    from causal_qd.archive.grid_archive import GridArchive as Archive
    from causal_qd.core.dag import DAG


def _cpdag(adj: np.ndarray) -> np.ndarray:
    """Compute a simple CPDAG representation.

    A directed edge *i → j* is compelled if there exists a v-structure
    at *j* involving *i*.  Otherwise, the edge is treated as undirected.
    This is a simplified heuristic; a full implementation would use the
    Meek rules.
    """
    n = adj.shape[0]
    cpdag = adj.copy()
    for j in range(n):
        parents = [i for i in range(n) if adj[i, j]]
        for a in range(len(parents)):
            for b in range(a + 1, len(parents)):
                pa, pb = parents[a], parents[b]
                if not adj[pa, pb] and not adj[pb, pa]:
                    # v-structure: pa → j ← pb — edges are compelled
                    continue
                # Non-v-structure parents — mark as undirected
                cpdag[pa, j] = 1
                cpdag[j, pa] = 1
                cpdag[pb, j] = 1
                cpdag[j, pb] = 1
    return cpdag


def _mec_key(adj: np.ndarray) -> bytes:
    """Return a hashable MEC key from a DAG adjacency matrix."""
    cp = _cpdag(adj)
    return cp.tobytes()


class MECRecall:
    """Fraction of the true DAG's MEC that appears in the archive."""

    @staticmethod
    def compute(archive: Archive, true_dag: DAG) -> float:
        """Compute MEC recall.

        The metric checks how many DAGs in the archive belong to the
        same Markov equivalence class as the ground-truth DAG.

        Parameters
        ----------
        archive:
            Archive with elite entries exposing a ``dag`` attribute.
        true_dag:
            Ground-truth DAG with an ``adjacency_matrix`` attribute.

        Returns
        -------
        float
            Fraction of archive members in the true MEC.  Returns ``0.0``
            if the archive is empty.
        """
        if not archive.elites:
            return 0.0

        true_key = _mec_key(true_dag.adjacency_matrix)

        matching = sum(
            1
            for entry in archive.elites.values()
            if _mec_key(entry.dag.adjacency_matrix) == true_key
        )

        return matching / len(archive.elites)
