"""Convert a DAG to its MEC representative (CPDAG / essential graph)."""
from __future__ import annotations

from causal_qd.core.dag import DAG
from causal_qd.mec.cpdag import CPDAGConverter
from causal_qd.types import AdjacencyMatrix


class DAGtoMEC:
    """Convert a DAG to the CPDAG that represents its MEC.

    This is a thin convenience wrapper around
    :class:`~causal_qd.mec.cpdag.CPDAGConverter`.
    """

    def __init__(self) -> None:
        self._converter = CPDAGConverter()

    def convert(self, dag: DAG) -> AdjacencyMatrix:
        """Return the CPDAG representation of *dag*'s MEC.

        Parameters
        ----------
        dag : DAG

        Returns
        -------
        AdjacencyMatrix
            CPDAG adjacency matrix.
        """
        return self._converter.dag_to_cpdag(dag)

    def get_essential_graph(self, dag: DAG) -> AdjacencyMatrix:
        """Return the essential graph (synonym for the CPDAG).

        The essential graph contains a directed edge *i → j* when that
        orientation is shared by every DAG in the MEC, and an undirected
        edge otherwise.

        Parameters
        ----------
        dag : DAG

        Returns
        -------
        AdjacencyMatrix
        """
        return self._converter.dag_to_cpdag(dag)
