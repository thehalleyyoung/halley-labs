"""Advanced equivalence class decomposition and intervention design.

Extends the basic MEC decomposition with chain component analysis,
inter-MEC diversity computation, and optimal intervention design.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.core.dag import DAG
from causal_qd.types import AdjacencyMatrix, GraphHash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _topological_sort(adj: AdjacencyMatrix) -> List[int]:
    """Kahn's topological sort."""
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).copy()
    queue: deque[int] = deque(i for i in range(n) if in_deg[i] == 0)
    order: List[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in range(n):
            if adj[node, child]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
    return order


def _dag_to_cpdag(adj: AdjacencyMatrix) -> AdjacencyMatrix:
    """Convert DAG to its Completed PDAG (CPDAG).

    An edge i → j is compelled if there exists k → j with k not
    adjacent to i.  Otherwise the edge is reversible and represented
    as undirected (both i → j and j → i set to 1).

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.

    Returns
    -------
    AdjacencyMatrix
        CPDAG adjacency matrix.
    """
    n = adj.shape[0]
    cpdag = adj.copy()
    compelled = np.zeros((n, n), dtype=np.bool_)

    order = _topological_sort(adj)
    for j in order:
        parents_j = list(np.where(adj[:, j])[0])
        for p in parents_j:
            if compelled[p, j]:
                continue
            is_compelled = False
            for k in parents_j:
                if k == p:
                    continue
                if not adj[p, k] and not adj[k, p]:
                    is_compelled = True
                    break
            if is_compelled:
                compelled[p, j] = True
                for k in parents_j:
                    if k != p and not adj[p, k] and not adj[k, p]:
                        compelled[k, j] = True

    for i in range(n):
        for j in range(n):
            if adj[i, j] and not compelled[i, j]:
                cpdag[j, i] = 1

    return cpdag


def _cpdag_hash(cpdag: AdjacencyMatrix) -> int:
    """Hash a CPDAG for MEC identification."""
    return hash(cpdag.tobytes())


# ---------------------------------------------------------------------------
# ChainComponentDecomposition
# ---------------------------------------------------------------------------


class ChainComponentDecomposition:
    """Decompose a CPDAG into its chain components.

    A chain component is a maximal connected subgraph of the CPDAG
    consisting of only undirected edges.  Within a chain component,
    edge orientations can vary across DAGs in the MEC.  Between
    chain components, all edges are compelled (directed).

    This decomposition is useful for understanding which parts of
    the structure are identifiable and which are not.
    """

    def decompose(
        self, cpdag: AdjacencyMatrix
    ) -> List[List[int]]:
        """Decompose the CPDAG into chain components.

        A chain component is a maximal set of nodes connected by
        undirected edges (edges present in both directions in the CPDAG).

        Parameters
        ----------
        cpdag : AdjacencyMatrix
            CPDAG adjacency matrix.

        Returns
        -------
        List[List[int]]
            List of chain components, each a list of node indices.
        """
        n = cpdag.shape[0]

        # Undirected edges: both i → j and j → i
        undirected = np.zeros((n, n), dtype=np.bool_)
        for i in range(n):
            for j in range(i + 1, n):
                if cpdag[i, j] and cpdag[j, i]:
                    undirected[i, j] = True
                    undirected[j, i] = True

        # Find connected components using only undirected edges
        visited = np.zeros(n, dtype=np.bool_)
        components: List[List[int]] = []

        for start in range(n):
            if visited[start]:
                continue
            component: List[int] = []
            queue: deque[int] = deque([start])
            visited[start] = True
            while queue:
                node = queue.popleft()
                component.append(node)
                for nb in range(n):
                    if undirected[node, nb] and not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            components.append(component)

        return components

    def compelled_edges(
        self, cpdag: AdjacencyMatrix
    ) -> List[Tuple[int, int]]:
        """Return all compelled (directed) edges in the CPDAG.

        An edge i → j is compelled if cpdag[i, j] = 1 and cpdag[j, i] = 0.

        Parameters
        ----------
        cpdag : AdjacencyMatrix

        Returns
        -------
        List[Tuple[int, int]]
            List of compelled directed edges.
        """
        n = cpdag.shape[0]
        compelled: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(n):
                if cpdag[i, j] and not cpdag[j, i]:
                    compelled.append((i, j))
        return compelled

    def reversible_edges(
        self, cpdag: AdjacencyMatrix
    ) -> List[Tuple[int, int]]:
        """Return all reversible (undirected) edges in the CPDAG.

        An edge is reversible if both i → j and j → i are in the CPDAG.
        Returns each undirected edge once as (min, max).

        Parameters
        ----------
        cpdag : AdjacencyMatrix

        Returns
        -------
        List[Tuple[int, int]]
        """
        n = cpdag.shape[0]
        reversible: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if cpdag[i, j] and cpdag[j, i]:
                    reversible.append((i, j))
        return reversible

    def compelled_subgraph(
        self, cpdag: AdjacencyMatrix
    ) -> AdjacencyMatrix:
        """Extract the subgraph of compelled edges.

        Parameters
        ----------
        cpdag : AdjacencyMatrix

        Returns
        -------
        AdjacencyMatrix
            Adjacency matrix containing only compelled edges.
        """
        n = cpdag.shape[0]
        result = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(n):
                if cpdag[i, j] and not cpdag[j, i]:
                    result[i, j] = 1
        return result


# ---------------------------------------------------------------------------
# AdvancedEquivalenceDecomposer
# ---------------------------------------------------------------------------


class AdvancedEquivalenceDecomposer:
    """Extended equivalence class decomposition with diversity metrics.

    Groups DAGs into MECs, computes inter-MEC diversity, and finds
    representative DAGs for each class.
    """

    def __init__(self) -> None:
        self._chain_decomp = ChainComponentDecomposition()

    def decompose_to_mecs(
        self, dags: List[DAG]
    ) -> Dict[int, List[DAG]]:
        """Group DAGs by their Markov Equivalence Class.

        Parameters
        ----------
        dags : List[DAG]
            Collection of DAGs.

        Returns
        -------
        Dict[int, List[DAG]]
            MEC hash → list of member DAGs.
        """
        groups: Dict[int, List[DAG]] = defaultdict(list)
        for dag in dags:
            cpdag = _dag_to_cpdag(dag.adjacency)
            h = _cpdag_hash(cpdag)
            groups[h].append(dag)
        return dict(groups)

    def mec_representative(
        self, dags: List[DAG]
    ) -> DAG:
        """Select a canonical representative from a set of MEC-equivalent DAGs.

        Chooses the DAG with the lexicographically smallest adjacency
        matrix (when flattened).

        Parameters
        ----------
        dags : List[DAG]
            DAGs in the same MEC.

        Returns
        -------
        DAG
            Representative DAG.
        """
        if not dags:
            raise ValueError("Empty DAG list.")
        return min(dags, key=lambda d: d.adjacency.tobytes())

    def inter_mec_diversity(
        self, mec_groups: Dict[int, List[DAG]]
    ) -> float:
        """Compute diversity between MECs.

        Diversity is measured as the average structural Hamming distance
        between representative DAGs of different MECs.

        Parameters
        ----------
        mec_groups : Dict[int, List[DAG]]
            MEC hash → member DAGs.

        Returns
        -------
        float
            Average inter-MEC SHD normalized by max possible edges.
        """
        reps = [self.mec_representative(dags) for dags in mec_groups.values()]
        if len(reps) < 2:
            return 0.0

        total_shd = 0.0
        count = 0
        n = reps[0].num_nodes
        max_edges = n * (n - 1)

        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                shd = int(np.sum(reps[i].adjacency != reps[j].adjacency))
                total_shd += shd / max(max_edges, 1)
                count += 1

        return total_shd / max(count, 1)

    def mec_sizes(
        self, mec_groups: Dict[int, List[DAG]]
    ) -> Dict[int, int]:
        """Return the size (number of DAGs) per MEC.

        Parameters
        ----------
        mec_groups

        Returns
        -------
        Dict[int, int]
        """
        return {h: len(dags) for h, dags in mec_groups.items()}

    def mec_summary(
        self, mec_groups: Dict[int, List[DAG]]
    ) -> Dict[str, float]:
        """Summary statistics of MEC decomposition.

        Parameters
        ----------
        mec_groups

        Returns
        -------
        Dict[str, float]
        """
        sizes = list(self.mec_sizes(mec_groups).values())
        return {
            "n_mecs": float(len(sizes)),
            "mean_size": float(np.mean(sizes)) if sizes else 0.0,
            "max_size": float(max(sizes)) if sizes else 0.0,
            "min_size": float(min(sizes)) if sizes else 0.0,
            "diversity": self.inter_mec_diversity(mec_groups),
        }


# ---------------------------------------------------------------------------
# InterventionDesign
# ---------------------------------------------------------------------------


class InterventionDesign:
    """Design interventions to maximally distinguish MECs.

    Given a collection of diverse DAGs from the archive, determines
    which interventions would provide the most information about the
    true causal structure.
    """

    def __init__(self) -> None:
        self._chain_decomp = ChainComponentDecomposition()

    def expected_information_gain(
        self,
        dags: List[DAG],
        target: int,
    ) -> float:
        """Compute expected information gain of intervening on *target*.

        An intervention on variable *target* distinguishes two DAGs
        if they imply different parent sets for *target* (since
        interventions cut incoming edges).  The information gain
        is the reduction in entropy of the DAG distribution.

        Parameters
        ----------
        dags : List[DAG]
            Current set of plausible DAGs.
        target : int
            Variable to intervene on.

        Returns
        -------
        float
            Expected information gain in bits.
        """
        if len(dags) <= 1:
            return 0.0

        # Group DAGs by the parent set of the target
        groups: Dict[FrozenSet[int], int] = defaultdict(int)
        for dag in dags:
            parents = frozenset(int(i) for i in np.where(dag.adjacency[:, target])[0])
            groups[parents] += 1

        # Prior entropy
        n = len(dags)
        prior_entropy = math.log2(n) if n > 0 else 0.0

        # Posterior entropy (weighted average of within-group entropies)
        posterior_entropy = 0.0
        for count in groups.values():
            if count > 0:
                p_group = count / n
                group_entropy = math.log2(count) if count > 1 else 0.0
                posterior_entropy += p_group * group_entropy

        return prior_entropy - posterior_entropy

    def greedy_intervention_selection(
        self,
        dags: List[DAG],
        n_interventions: int,
        candidate_targets: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Greedily select the most informative interventions.

        At each step, selects the variable whose intervention provides
        the highest expected information gain, then updates the DAG
        groups as if that intervention had been performed.

        Parameters
        ----------
        dags : List[DAG]
            Plausible DAGs.
        n_interventions : int
            Number of interventions to select.
        candidate_targets : List[int] | None
            Variables eligible for intervention.  If ``None``, all
            variables are considered.

        Returns
        -------
        List[Tuple[int, float]]
            Selected (target, information_gain) pairs in order.
        """
        if not dags:
            return []

        n_nodes = dags[0].num_nodes
        if candidate_targets is None:
            candidates = list(range(n_nodes))
        else:
            candidates = list(candidate_targets)

        selected: List[Tuple[int, float]] = []
        remaining_dags = list(dags)
        used = set()

        for _ in range(min(n_interventions, len(candidates))):
            best_target = -1
            best_gain = -1.0

            for target in candidates:
                if target in used:
                    continue
                gain = self.expected_information_gain(remaining_dags, target)
                if gain > best_gain:
                    best_gain = gain
                    best_target = target

            if best_target < 0:
                break

            selected.append((best_target, best_gain))
            used.add(best_target)

            # Simulate the intervention: keep only representative DAGs
            # per parent-set group
            groups: Dict[FrozenSet[int], List[DAG]] = defaultdict(list)
            for dag in remaining_dags:
                parents = frozenset(
                    int(i) for i in np.where(dag.adjacency[:, best_target])[0]
                )
                groups[parents].append(dag)

            # Keep the largest group (most likely) for continued search
            remaining_dags = max(groups.values(), key=len)

        return selected

    def intervention_distinguishability_matrix(
        self, dags: List[DAG]
    ) -> npt.NDArray[np.float64]:
        """Compute pairwise distinguishability under each intervention.

        Returns a matrix where entry (i, t) indicates whether DAG i
        can be distinguished from at least one other DAG by intervening
        on variable t.

        Parameters
        ----------
        dags : List[DAG]

        Returns
        -------
        ndarray, shape (n_dags, n_nodes)
            Binary matrix of distinguishability.
        """
        if not dags:
            return np.array([], dtype=np.float64)

        n_dags = len(dags)
        n_nodes = dags[0].num_nodes
        matrix = np.zeros((n_dags, n_nodes), dtype=np.float64)

        for t in range(n_nodes):
            # Group by parent set of target t
            parent_sets: Dict[FrozenSet[int], List[int]] = defaultdict(list)
            for i, dag in enumerate(dags):
                parents = frozenset(
                    int(p) for p in np.where(dag.adjacency[:, t])[0]
                )
                parent_sets[parents].append(i)

            # A DAG is distinguishable if its group has fewer than all DAGs
            if len(parent_sets) > 1:
                for group_indices in parent_sets.values():
                    for idx in group_indices:
                        matrix[idx, t] = 1.0

        return matrix
