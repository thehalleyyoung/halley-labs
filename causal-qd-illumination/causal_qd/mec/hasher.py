"""Canonical hashing for DAGs and Markov Equivalence Classes.

This module implements a Weisfeiler-Lehman (1-WL) style colour-refinement
algorithm to compute canonical forms of adjacency matrices.  The canonical
form is then hashed with SHA-256 to produce compact integer fingerprints
that are invariant under node relabelling.

Three levels of hashing are provided:

* **DAG hash** – uniquely identifies a single DAG structure.
* **MEC hash** – maps every DAG in the same Markov Equivalence Class to an
  identical hash by first converting to the CPDAG representative.
* **Adjacency hash** – hashes an arbitrary adjacency matrix (not
  necessarily acyclic).

The colour-refinement canonicalization is a *necessary* but not *sufficient*
condition for graph isomorphism.  Graphs that are truly isomorphic will
always receive the same canonical form, but certain pathological
non-isomorphic graphs (e.g. strongly-regular graphs) may also collide.
For guaranteed correctness, pair with a full isomorphism solver such as
nauty / Traces or the VF2 algorithm.

Typical usage::

    from causal_qd.core.dag import DAG
    from causal_qd.mec.hasher import CanonicalHasher

    hasher = CanonicalHasher()

    dag = DAG.random_dag(n=5, density=0.3)
    dag_hash = hasher.hash_dag(dag)
    mec_hash = hasher.hash_mec(dag)

    # Compare two adjacency matrices
    if hasher.are_isomorphic(adj1, adj2):
        print("Likely isomorphic (up to 1-WL)")
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from causal_qd.core.dag import DAG
from causal_qd.mec.cpdag import CPDAGConverter
from causal_qd.types import AdjacencyMatrix, GraphHash

# Maximum number of colour-refinement iterations before declaring
# convergence.  In practice the loop converges in O(n) iterations for
# almost all graphs encountered in causal discovery.
_MAX_WL_ITERATIONS: int = 500


class CanonicalHasher:
    """Produce canonical hashes for DAGs and their Markov Equivalence Classes.

    The hasher uses a **Weisfeiler-Lehman (1-WL) colour-refinement**
    algorithm to compute a canonical ordering of the nodes in an
    adjacency matrix.  The resulting permuted matrix is then hashed
    with SHA-256 to obtain a compact integer fingerprint.

    Two DAGs (or CPDAGs) that are isomorphic under node relabelling
    are guaranteed to receive the same hash.  Non-isomorphic graphs
    *almost always* receive different hashes, but the 1-WL test cannot
    distinguish certain pathological cases (e.g. some strongly-regular
    graphs).

    Attributes
    ----------
    _cpdag_converter : CPDAGConverter
        Converter used to obtain CPDAG representations for MEC hashing.

    Examples
    --------
    >>> hasher = CanonicalHasher()
    >>> dag = DAG.from_edges(3, [(0, 1), (1, 2)])
    >>> h = hasher.hash_dag(dag)
    >>> isinstance(h, int)
    True
    """

    def __init__(self) -> None:
        """Initialise the hasher with a fresh CPDAGConverter instance."""
        self._cpdag_converter = CPDAGConverter()

    # ------------------------------------------------------------------
    # Public API – high-level hashing
    # ------------------------------------------------------------------

    def hash_dag(self, dag: DAG) -> GraphHash:
        """Return a canonical hash uniquely identifying the DAG structure.

        The hash is invariant to node relabelling: two DAGs that are
        isomorphic will produce the same hash value.

        Parameters
        ----------
        dag : DAG
            The directed acyclic graph to hash.

        Returns
        -------
        GraphHash
            A non-negative integer hash derived from the canonical form
            of the DAG's adjacency matrix.
        """
        canonical = self._canonical_form(dag.adjacency)
        return self._hash_matrix(canonical)

    def hash_mec(self, dag: DAG) -> GraphHash:
        """Return a hash that is identical for all DAGs in the same MEC.

        The hash is derived from the CPDAG (completed partially directed
        acyclic graph), which is the unique graphical representative of the
        Markov Equivalence Class.  Any two DAGs that encode the same set of
        conditional independencies will share the same CPDAG and therefore
        the same MEC hash.

        Parameters
        ----------
        dag : DAG
            Any member DAG of the equivalence class.

        Returns
        -------
        GraphHash
            Integer hash of the canonical CPDAG.  All DAGs in the same MEC
            map to this value.
        """
        cpdag = self._cpdag_converter.dag_to_cpdag(dag)
        canonical = self._canonical_form(cpdag)
        return self._hash_matrix(canonical)

    def hash_adjacency(self, adjacency: AdjacencyMatrix) -> GraphHash:
        """Hash a raw adjacency matrix that need not be a valid DAG.

        This is useful for hashing CPDAGs, PDAGs, undirected skeletons,
        or any other square binary matrix representation of a graph.

        Parameters
        ----------
        adjacency : AdjacencyMatrix
            An ``(n, n)`` integer matrix where ``adjacency[i, j] == 1``
            indicates an edge from node *i* to node *j*.

        Returns
        -------
        GraphHash
            Canonical integer hash of the adjacency matrix.
        """
        canonical = self._canonical_form(adjacency)
        return self._hash_matrix(canonical)

    # ------------------------------------------------------------------
    # Public API – isomorphism & certificates
    # ------------------------------------------------------------------

    def are_isomorphic(
        self, adj1: AdjacencyMatrix, adj2: AdjacencyMatrix
    ) -> bool:
        """Check whether two graphs are isomorphic by comparing canonical forms.

        The comparison uses the 1-WL colour-refinement canonical form.
        This is a **necessary but not sufficient** condition for true graph
        isomorphism – if this method returns ``False`` the graphs are
        *definitely* not isomorphic; if it returns ``True`` they are
        isomorphic *with very high probability* but not with certainty.

        For a guaranteed result, use a full isomorphism solver (e.g. VF2
        or nauty/Traces).

        Parameters
        ----------
        adj1 : AdjacencyMatrix
            First adjacency matrix, shape ``(n, n)``.
        adj2 : AdjacencyMatrix
            Second adjacency matrix, shape ``(m, m)``.

        Returns
        -------
        bool
            ``True`` if the canonical forms are identical (likely
            isomorphic); ``False`` if they differ (definitely not
            isomorphic).

        Notes
        -----
        Graphs of different sizes are immediately reported as
        non-isomorphic without running the refinement.
        """
        a1 = np.asarray(adj1, dtype=np.int8)
        a2 = np.asarray(adj2, dtype=np.int8)
        if a1.shape != a2.shape:
            return False
        c1 = self._canonical_form(a1)
        c2 = self._canonical_form(a2)
        return bool(np.array_equal(c1, c2))

    def canonical_certificate(self, adjacency: AdjacencyMatrix) -> bytes:
        """Return the canonical form of the graph as a raw byte string.

        The byte string is suitable for storage in databases, sets, or
        dictionaries as an isomorphism-invariant key.  Two graphs that
        receive the same certificate (under the 1-WL approximation) are
        considered equivalent.

        Parameters
        ----------
        adjacency : AdjacencyMatrix
            An ``(n, n)`` integer adjacency matrix.

        Returns
        -------
        bytes
            The canonical adjacency matrix serialised to contiguous bytes
            in C (row-major) order.

        Examples
        --------
        >>> hasher = CanonicalHasher()
        >>> cert = hasher.canonical_certificate(adj)
        >>> isinstance(cert, bytes)
        True
        """
        canonical = self._canonical_form(adjacency)
        return np.ascontiguousarray(canonical, dtype=np.int8).tobytes()

    # ------------------------------------------------------------------
    # Public API – order-aware hashing
    # ------------------------------------------------------------------

    def hash_with_order(
        self,
        adjacency: AdjacencyMatrix,
        node_order: list[int],
    ) -> GraphHash:
        """Hash an adjacency matrix while respecting a partial node ordering.

        Nodes are grouped into *equivalence classes* defined by their
        position in ``node_order``.  The canonical-form permutation is
        restricted so that nodes only swap positions within the same
        equivalence class; the relative ordering of classes is preserved.

        This is useful when some external semantics (e.g. a topological
        ordering or a domain-specific variable grouping) should be
        respected by the hash.

        Parameters
        ----------
        adjacency : AdjacencyMatrix
            An ``(n, n)`` integer adjacency matrix.
        node_order : list[int]
            A list of length *n* where ``node_order[i]`` gives the
            *class label* for node *i*.  Nodes sharing the same label
            belong to the same equivalence class and may be permuted
            freely; nodes with different labels retain their relative
            order (lower label first).

        Returns
        -------
        GraphHash
            Canonical hash that is invariant to permutations *within*
            equivalence classes but sensitive to permutations *across*
            classes.

        Raises
        ------
        ValueError
            If the length of *node_order* does not match the number of
            nodes in the adjacency matrix.

        Examples
        --------
        >>> hasher = CanonicalHasher()
        >>> adj = np.eye(4, dtype=np.int8)
        >>> h = hasher.hash_with_order(adj, [0, 0, 1, 1])
        >>> isinstance(h, int)
        True
        """
        adj = np.asarray(adjacency, dtype=np.int8)
        n = adj.shape[0]
        if len(node_order) != n:
            raise ValueError(
                f"node_order length ({len(node_order)}) != "
                f"adjacency size ({n})"
            )

        canonical = self._canonical_form_with_order(adj, node_order)
        return self._hash_matrix(canonical)

    # ------------------------------------------------------------------
    # Internal – Weisfeiler-Lehman colour refinement
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_form(adjacency: AdjacencyMatrix) -> AdjacencyMatrix:
        """Compute a canonical adjacency matrix via 1-WL colour refinement.

        The algorithm proceeds as follows:

        1. **Initialise colours.**  Each node receives an initial colour
           equal to the tuple ``(in_degree, out_degree)``.
        2. **Iterative refinement.**  At every iteration each node's
           colour is replaced by::

               new_colour = (old_colour,
                             sorted tuple of (edge_direction, neighbour_colour)
                             for every neighbour)

           where *edge_direction* encodes whether the neighbour is
           reached by an incoming edge, an outgoing edge, or both
           (undirected / bidirectional).
        3. **Convergence.**  The loop terminates when the number of
           distinct colour classes does not increase between two
           consecutive iterations.
        4. **Canonical ordering.**  Nodes are sorted lexicographically
           by their final colour tuple.  Ties are broken by comparing
           the full neighbour-colour multisets, ensuring a deterministic
           permutation.
        5. **Permutation.**  The adjacency matrix is permuted according
           to the canonical ordering and returned.

        This is equivalent to the 1-dimensional Weisfeiler-Lehman (1-WL)
        graph isomorphism heuristic.  It distinguishes *almost all*
        non-isomorphic graphs but can fail on certain regular graphs.

        Parameters
        ----------
        adjacency : AdjacencyMatrix
            An ``(n, n)`` integer matrix.

        Returns
        -------
        AdjacencyMatrix
            The permuted adjacency matrix in canonical form.
        """
        adj = np.asarray(adjacency, dtype=np.int8)
        n = adj.shape[0]
        if n == 0:
            return adj.copy()

        # ----------------------------------------------------------
        # Step 1 – initial colour assignment: (in_degree, out_degree)
        # ----------------------------------------------------------
        in_deg = adj.sum(axis=0)   # column sums
        out_deg = adj.sum(axis=1)  # row sums

        colours: list[tuple] = [
            (int(in_deg[v]), int(out_deg[v])) for v in range(n)
        ]

        # Pre-compute directed neighbour lists for each node.
        # For every node v we store tuples (direction, neighbour_index)
        # where direction is:
        #   0 – v -> u  (outgoing)
        #   1 – u -> v  (incoming)
        #   2 – both directions (bidirectional / undirected)
        out_nbrs: list[list[int]] = [[] for _ in range(n)]
        in_nbrs: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    out_nbrs[i].append(j)
                    in_nbrs[j].append(i)

        # ----------------------------------------------------------
        # Step 2 – iterative colour refinement
        # ----------------------------------------------------------
        prev_num_classes = len(set(colours))
        for _ in range(_MAX_WL_ITERATIONS):
            new_colours: list[tuple] = []
            for v in range(n):
                # Build the multiset of (direction, neighbour_colour).
                nbr_sig: list[tuple] = []
                for u in out_nbrs[v]:
                    if adj[u, v]:
                        # bidirectional edge
                        nbr_sig.append((2, colours[u]))
                    else:
                        nbr_sig.append((0, colours[u]))
                for u in in_nbrs[v]:
                    if not adj[v, u]:
                        # pure incoming edge (not already counted as bidi)
                        nbr_sig.append((1, colours[u]))
                nbr_sig.sort()
                new_colours.append((colours[v], tuple(nbr_sig)))
            colours = new_colours

            cur_num_classes = len(set(colours))
            if cur_num_classes == prev_num_classes:
                # Stable – no new colour classes were created.
                break
            prev_num_classes = cur_num_classes

        # ----------------------------------------------------------
        # Step 3 – derive canonical node ordering from final colours
        # ----------------------------------------------------------
        # Primary sort: colour tuple (lexicographic).
        # Secondary tie-break: full sorted neighbour structure using
        # the *final* colour tuples of neighbours, ensuring a
        # deterministic result even when colours alone do not
        # distinguish all nodes.
        tiebreak: list[tuple] = []
        for v in range(n):
            full_nbr = sorted(
                (int(adj[v, u]), int(adj[u, v]), colours[u])
                for u in range(n)
                if adj[v, u] or adj[u, v]
            )
            tiebreak.append(tuple(full_nbr))

        perm = sorted(
            range(n),
            key=lambda v: (colours[v], tiebreak[v]),
        )

        # ----------------------------------------------------------
        # Step 4 – permute the adjacency matrix
        # ----------------------------------------------------------
        canonical: AdjacencyMatrix = adj[np.ix_(perm, perm)]
        return canonical

    @staticmethod
    def _canonical_form_with_order(
        adjacency: AdjacencyMatrix,
        node_order: Sequence[int],
    ) -> AdjacencyMatrix:
        """Canonical form that only permutes within equivalence classes.

        Nodes are partitioned into equivalence classes given by
        *node_order*.  Within each class the standard 1-WL colour
        refinement is used to derive a canonical sub-ordering.  The
        classes themselves are arranged in ascending order of their
        label.

        Parameters
        ----------
        adjacency : AdjacencyMatrix
            An ``(n, n)`` integer matrix.
        node_order : Sequence[int]
            Class label for each node.

        Returns
        -------
        AdjacencyMatrix
            Canonically permuted adjacency matrix.
        """
        adj = np.asarray(adjacency, dtype=np.int8)
        n = adj.shape[0]
        if n == 0:
            return adj.copy()

        # Group node indices by their equivalence-class label.
        class_map: dict[int, list[int]] = {}
        for node_idx, label in enumerate(node_order):
            class_map.setdefault(label, []).append(node_idx)

        # ----------------------------------------------------------
        # Compute WL colours on the full graph (needed for intra-class
        # tie-breaking that accounts for inter-class edges).
        # ----------------------------------------------------------
        in_deg = adj.sum(axis=0)
        out_deg = adj.sum(axis=1)
        colours: list[tuple] = [
            (int(in_deg[v]), int(out_deg[v])) for v in range(n)
        ]

        out_nbrs: list[list[int]] = [[] for _ in range(n)]
        in_nbrs: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if adj[i, j]:
                    out_nbrs[i].append(j)
                    in_nbrs[j].append(i)

        prev_num_classes = len(set(colours))
        for _ in range(_MAX_WL_ITERATIONS):
            new_colours: list[tuple] = []
            for v in range(n):
                nbr_sig: list[tuple] = []
                for u in out_nbrs[v]:
                    if adj[u, v]:
                        nbr_sig.append((2, colours[u]))
                    else:
                        nbr_sig.append((0, colours[u]))
                for u in in_nbrs[v]:
                    if not adj[v, u]:
                        nbr_sig.append((1, colours[u]))
                nbr_sig.sort()
                new_colours.append((colours[v], tuple(nbr_sig)))
            colours = new_colours
            cur_num_classes = len(set(colours))
            if cur_num_classes == prev_num_classes:
                break
            prev_num_classes = cur_num_classes

        # ----------------------------------------------------------
        # Build permutation: classes in label order, nodes within each
        # class sorted by their WL colour + neighbour tiebreak.
        # ----------------------------------------------------------
        tiebreak: list[tuple] = []
        for v in range(n):
            full_nbr = sorted(
                (int(adj[v, u]), int(adj[u, v]), colours[u])
                for u in range(n)
                if adj[v, u] or adj[u, v]
            )
            tiebreak.append(tuple(full_nbr))

        perm: list[int] = []
        for label in sorted(class_map.keys()):
            members = class_map[label]
            members_sorted = sorted(
                members,
                key=lambda v: (colours[v], tiebreak[v]),
            )
            perm.extend(members_sorted)

        canonical: AdjacencyMatrix = adj[np.ix_(perm, perm)]
        return canonical

    # ------------------------------------------------------------------
    # Internal – matrix hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_matrix(matrix: AdjacencyMatrix) -> GraphHash:
        """SHA-256 based integer hash of a numpy matrix.

        The matrix is first converted to a contiguous C-order byte
        buffer and then hashed with SHA-256.  The first 8 bytes of the
        digest are interpreted as a big-endian unsigned integer to
        produce the final hash value.

        Parameters
        ----------
        matrix : AdjacencyMatrix
            The matrix to hash (typically an ``int8`` array).

        Returns
        -------
        GraphHash
            Non-negative integer derived from the first 64 bits of the
            SHA-256 digest.
        """
        raw = np.ascontiguousarray(matrix, dtype=np.int8).tobytes()
        digest = hashlib.sha256(raw).digest()
        return int.from_bytes(digest[:8], byteorder="big")
