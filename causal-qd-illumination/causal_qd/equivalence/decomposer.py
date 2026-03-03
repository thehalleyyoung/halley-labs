"""Group DAGs by their Markov Equivalence Class."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from causal_qd.core.dag import DAG
from causal_qd.mec.hasher import CanonicalHasher
from causal_qd.types import GraphHash


class EquivalenceClassDecomposer:
    """Partition a collection of DAGs into Markov Equivalence Classes.

    Each group is identified by the canonical MEC hash produced by
    :class:`~causal_qd.mec.hasher.CanonicalHasher`.
    """

    def __init__(self) -> None:
        self._hasher = CanonicalHasher()

    def decompose(self, dags: List[DAG]) -> Dict[GraphHash, List[DAG]]:
        """Group *dags* by their MEC.

        Parameters
        ----------
        dags : List[DAG]
            Collection of directed acyclic graphs.

        Returns
        -------
        Dict[GraphHash, List[DAG]]
            Mapping from MEC hash to the list of DAGs in that class.
        """
        groups: Dict[GraphHash, List[DAG]] = defaultdict(list)
        for dag in dags:
            h = self._hasher.hash_mec(dag)
            groups[h].append(dag)
        return dict(groups)

    def representative(self, mec_hash: GraphHash, dags: List[DAG]) -> DAG:
        """Return a canonical representative for the MEC identified by *mec_hash*.

        The representative is chosen as the DAG whose canonical DAG hash
        is smallest among the members of the class.

        Parameters
        ----------
        mec_hash : GraphHash
            The MEC hash to look for.
        dags : List[DAG]
            Pool of DAGs to search.

        Returns
        -------
        DAG

        Raises
        ------
        ValueError
            If no DAG in *dags* belongs to the requested MEC.
        """
        candidates: list[tuple[GraphHash, DAG]] = []
        for dag in dags:
            if self._hasher.hash_mec(dag) == mec_hash:
                candidates.append((self._hasher.hash_dag(dag), dag))

        if not candidates:
            raise ValueError(
                f"No DAG in the provided list belongs to MEC {mec_hash}."
            )

        # Pick the candidate with the smallest canonical DAG hash.
        candidates.sort(key=lambda pair: pair[0])
        return candidates[0][1]
