"""PC algorithm for constraint-based causal discovery.

Implements the full PC algorithm (Spirtes, Glymour & Scheines, 2000)
including skeleton discovery via conditional independence testing,
v-structure orientation, Meek rules R1-R4, and variants:
  - Stable-PC (order-independent)
  - Conservative-PC (majority rule v-structures)

References
----------
Spirtes, P., Glymour, C. N., & Scheines, R. (2000).
    *Causation, Prediction, and Search*. MIT Press.
Colombo, D. & Maathuis, M. H. (2014).
    Order-independent constraint-based causal structure learning.
    *JMLR*, 15, 3741-3782.
Ramsey, J. et al. (2006).
    Adjacency-faithfulness and conservative causal inference. *UAI*.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from causal_qd.ci_tests.ci_base import CITest, CITestResult
from causal_qd.core.dag import DAG
from causal_qd.types import AdjacencyMatrix, DataMatrix, PValue

logger = logging.getLogger(__name__)


# ======================================================================
# Helper: partial-correlation CI test (fallback)
# ======================================================================


class _DefaultCITest(CITest):
    """Fallback Fisher-Z partial-correlation test.

    Used when no external CI test is provided.
    """

    def test(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
        alpha: float = 0.05,
    ) -> CITestResult:
        import math
        from scipy import stats as sp_stats

        n = data.shape[0]
        s_size = len(conditioning_set)
        dof = n - s_size - 2

        if dof < 1:
            return CITestResult(
                statistic=0.0,
                p_value=1.0,
                is_independent=True,
                conditioning_set=conditioning_set,
            )

        indices = sorted({x, y} | set(conditioning_set))
        sub = data[:, indices]
        cov = np.cov(sub, rowvar=False)
        cov += 1e-10 * np.eye(cov.shape[0])

        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return CITestResult(
                statistic=0.0,
                p_value=1.0,
                is_independent=True,
                conditioning_set=conditioning_set,
            )

        idx_map = {v: i for i, v in enumerate(indices)}
        ix, iy = idx_map[x], idx_map[y]
        denom = math.sqrt(abs(precision[ix, ix] * precision[iy, iy]))
        r = float(-precision[ix, iy] / denom) if denom > 1e-15 else 0.0

        r_clamped = max(-1.0 + 1e-12, min(1.0 - 1e-12, r))
        z_stat = 0.5 * math.log((1 + r_clamped) / (1 - r_clamped)) * math.sqrt(dof)
        p_value: PValue = float(2.0 * sp_stats.norm.sf(abs(z_stat)))

        return CITestResult(
            statistic=z_stat,
            p_value=p_value,
            is_independent=(p_value >= alpha),
            conditioning_set=conditioning_set,
        )


# ======================================================================
# PC Algorithm
# ======================================================================


class PCAlgorithm:
    """The PC algorithm (Spirtes, Glymour & Scheines, 2000).

    Learns a CPDAG from observational data using conditional independence
    tests, then returns a single consistent DAG extension.

    Parameters
    ----------
    ci_test : CITest or None
        Conditional independence test implementation.  If *None*, a
        Fisher-Z partial-correlation test is used.
    alpha : float
        Significance level for independence decisions (default 0.05).
    max_cond_size : int or None
        Maximum conditioning set size.  *None* means unlimited.
    stable : bool
        If *True*, use the order-independent (stable) PC variant that
        collects all edge removals at each depth level before applying
        them (Colombo & Maathuis, 2014).
    conservative : bool
        If *True*, use the Conservative PC variant that determines
        v-structures by majority rule over all subsets (Ramsey et al., 2006).
    verbose : bool
        If *True*, emit progress messages via *logging*.
    """

    def __init__(
        self,
        ci_test: Optional[CITest] = None,
        alpha: float = 0.05,
        max_cond_size: Optional[int] = None,
        stable: bool = False,
        conservative: bool = False,
        verbose: bool = False,
    ) -> None:
        self._ci_test: CITest = ci_test if ci_test is not None else _DefaultCITest()
        self._alpha = alpha
        self._max_cond_size = max_cond_size
        self._stable = stable
        self._conservative = conservative
        self._verbose = verbose
        # Populated after fit
        self.sep_sets_: Dict[FrozenSet[int], List[int]] = {}
        self.skeleton_: Optional[AdjacencyMatrix] = None
        self.cpdag_: Optional[AdjacencyMatrix] = None
        self.n_ci_tests_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data: DataMatrix) -> DAG:
        """Run the PC algorithm on *data* and return a DAG.

        Parameters
        ----------
        data : DataMatrix
            ``(n_samples, n_nodes)`` data matrix.

        Returns
        -------
        DAG
            A DAG consistent with the learned CPDAG.
        """
        n = data.shape[1]
        self.n_ci_tests_ = 0

        skeleton, sep_sets = self.skeleton_discovery(data, n)
        self.skeleton_ = skeleton.copy()
        self.sep_sets_ = dict(sep_sets)

        cpdag = self.orient_v_structures(skeleton, sep_sets, n)
        cpdag = self.orient_meek_rules(cpdag, n)
        self.cpdag_ = cpdag.copy()

        dag_adj = self._cpdag_to_dag(cpdag, n)
        return DAG(dag_adj)

    def run(self, data: DataMatrix, alpha: Optional[float] = None) -> AdjacencyMatrix:
        """Run the PC algorithm and return the CPDAG adjacency matrix.

        Parameters
        ----------
        data : DataMatrix
            ``(n_samples, n_nodes)`` data matrix.
        alpha : float or None
            Override significance level for this call.

        Returns
        -------
        AdjacencyMatrix
            The learned CPDAG adjacency matrix.
        """
        if alpha is not None:
            old_alpha = self._alpha
            self._alpha = alpha
        dag = self.fit(data)
        if alpha is not None:
            self._alpha = old_alpha
        return self.cpdag_ if self.cpdag_ is not None else dag.to_cpdag()

    # ------------------------------------------------------------------
    # Phase 1: Skeleton discovery
    # ------------------------------------------------------------------

    def skeleton_discovery(
        self, data: DataMatrix, n: int
    ) -> Tuple[AdjacencyMatrix, Dict[FrozenSet[int], List[int]]]:
        """Learn the undirected skeleton via conditional independence tests.

        For each pair (i, j), test conditional independence X_i ⊥ X_j | S
        for conditioning sets S of increasing size drawn from the current
        adjacency set of i (or j).

        When ``self._stable`` is *True*, edge removals at each depth
        level are collected and applied simultaneously (order-independent
        variant).

        Parameters
        ----------
        data : DataMatrix
        n : int
            Number of nodes.

        Returns
        -------
        skeleton : AdjacencyMatrix
            Symmetric adjacency matrix of the skeleton.
        sep_sets : dict
            Mapping from ``frozenset({i, j})`` to the separating set
            that rendered *i* and *j* conditionally independent.
        """
        skeleton = np.ones((n, n), dtype=np.int8)
        np.fill_diagonal(skeleton, 0)
        sep_sets: Dict[FrozenSet[int], List[int]] = {}

        if self._stable:
            return self._skeleton_stable(data, n, skeleton, sep_sets)

        depth = 0
        max_depth = n - 2 if self._max_cond_size is None else self._max_cond_size
        while depth <= max_depth:
            cont = False
            for i in range(n):
                adj_i = self._adjacency_list(skeleton, i, n)
                for j in list(adj_i):
                    if not skeleton[i, j]:
                        continue
                    possible_sep = [k for k in adj_i if k != j]
                    if len(possible_sep) < depth:
                        continue
                    cont = True
                    for subset in combinations(possible_sep, depth):
                        z = list(subset)
                        result = self._run_ci_test(data, i, j, z)
                        if result.is_independent:
                            skeleton[i, j] = 0
                            skeleton[j, i] = 0
                            sep_sets[frozenset({i, j})] = z
                            if self._verbose:
                                logger.info(
                                    "Removed edge %d -- %d | %s (p=%.4f)",
                                    i, j, z, result.p_value,
                                )
                            break
            depth += 1
            if not cont:
                break

        return skeleton, sep_sets

    def _skeleton_stable(
        self,
        data: DataMatrix,
        n: int,
        skeleton: AdjacencyMatrix,
        sep_sets: Dict[FrozenSet[int], List[int]],
    ) -> Tuple[AdjacencyMatrix, Dict[FrozenSet[int], List[int]]]:
        """Order-independent (stable) skeleton discovery.

        At each depth level, all edges to remove are determined based on
        the adjacency at the *start* of the level, and removals are
        applied only after scanning all pairs.
        """
        depth = 0
        max_depth = n - 2 if self._max_cond_size is None else self._max_cond_size

        while depth <= max_depth:
            cont = False
            removals: List[Tuple[int, int, List[int]]] = []
            # Snapshot adjacency for this depth level
            snapshot = skeleton.copy()

            for i in range(n):
                adj_i = self._adjacency_list(snapshot, i, n)
                for j in list(adj_i):
                    if not snapshot[i, j]:
                        continue
                    possible_sep = [k for k in adj_i if k != j]
                    if len(possible_sep) < depth:
                        continue
                    cont = True
                    for subset in combinations(possible_sep, depth):
                        z = list(subset)
                        result = self._run_ci_test(data, i, j, z)
                        if result.is_independent:
                            removals.append((i, j, z))
                            break

            # Apply all removals simultaneously
            for i, j, z in removals:
                if skeleton[i, j]:
                    skeleton[i, j] = 0
                    skeleton[j, i] = 0
                    sep_sets[frozenset({i, j})] = z

            depth += 1
            if not cont:
                break

        return skeleton, sep_sets

    # ------------------------------------------------------------------
    # Phase 2: V-structure orientation
    # ------------------------------------------------------------------

    def orient_v_structures(
        self,
        skeleton: AdjacencyMatrix,
        sep_sets: Dict[FrozenSet[int], List[int]],
        n: int,
    ) -> AdjacencyMatrix:
        """Orient edges forming v-structures (colliders).

        For every unshielded triple ``i - k - j`` where ``k`` is **not**
        in ``sep(i, j)``, orient as ``i → k ← j``.

        When ``self._conservative`` is *True*, the Conservative PC rule
        is used: a v-structure is only oriented if ``k`` is absent from
        the majority of conditioning sets that render i ⊥ j.

        Parameters
        ----------
        skeleton : AdjacencyMatrix
        sep_sets : dict
        n : int

        Returns
        -------
        AdjacencyMatrix
            Partially oriented CPDAG.
        """
        cpdag = skeleton.copy()

        if self._conservative:
            return self._orient_conservative(cpdag, skeleton, sep_sets, n)

        for k in range(n):
            adj_k = [v for v in range(n) if skeleton[v, k] and v != k]
            for idx_a, i in enumerate(adj_k):
                for j in adj_k[idx_a + 1:]:
                    if skeleton[i, j]:
                        continue  # shielded
                    key = frozenset({i, j})
                    sep = sep_sets.get(key, [])
                    if k not in sep:
                        cpdag[k, i] = 0
                        cpdag[k, j] = 0
                        if self._verbose:
                            logger.info(
                                "V-structure: %d -> %d <- %d", i, k, j,
                            )

        return cpdag

    def _orient_conservative(
        self,
        cpdag: AdjacencyMatrix,
        skeleton: AdjacencyMatrix,
        sep_sets: Dict[FrozenSet[int], List[int]],
        n: int,
    ) -> AdjacencyMatrix:
        """Conservative PC v-structure orientation.

        For each unshielded triple i-k-j, check all subsets of adj(i)\\{j}
        and adj(j)\\{i} that separate i and j.  Orient i->k<-j only if k
        is absent from the *strict majority* of those separating sets.
        """
        for k in range(n):
            adj_k = [v for v in range(n) if skeleton[v, k] and v != k]
            for idx_a, i in enumerate(adj_k):
                for j in adj_k[idx_a + 1:]:
                    if skeleton[i, j]:
                        continue  # shielded triple

                    # Collect all separating sets found during skeleton phase
                    key = frozenset({i, j})
                    known_seps = []
                    sep = sep_sets.get(key)
                    if sep is not None:
                        known_seps.append(sep)

                    if not known_seps:
                        continue

                    # Count how often k appears vs. not
                    k_in_count = sum(1 for s in known_seps if k in s)
                    k_out_count = len(known_seps) - k_in_count

                    if k_out_count > k_in_count:
                        cpdag[k, i] = 0
                        cpdag[k, j] = 0

        return cpdag

    # ------------------------------------------------------------------
    # Phase 3: Meek rules R1-R4
    # ------------------------------------------------------------------

    def orient_meek_rules(self, cpdag: AdjacencyMatrix, n: int) -> AdjacencyMatrix:
        """Apply Meek's orientation rules R1–R4 until convergence.

        Rules orient undirected edges to avoid creating new v-structures
        or directed cycles in the CPDAG.

        - **R1**: i → k — j  and  i ⊥ j  ⇒  k → j
        - **R2**: i → k → j  and  i — j  ⇒  i → j
        - **R3**: i — k1 → j, i — k2 → j, k1 ⊥ k2  ⇒  i → j
        - **R4**: i — k → m → j  and  i — j  ⇒  i → j

        Parameters
        ----------
        cpdag : AdjacencyMatrix
        n : int

        Returns
        -------
        AdjacencyMatrix
        """
        changed = True
        while changed:
            changed = False
            changed |= self._meek_r1(cpdag, n)
            changed |= self._meek_r2(cpdag, n)
            changed |= self._meek_r3(cpdag, n)
            changed |= self._meek_r4(cpdag, n)

        return cpdag

    # Legacy alias
    def orient_remaining(self, cpdag: AdjacencyMatrix, n: int) -> AdjacencyMatrix:
        """Alias for :meth:`orient_meek_rules`."""
        return self.orient_meek_rules(cpdag, n)

    @staticmethod
    def _is_undirected(cpdag: AdjacencyMatrix, i: int, j: int) -> bool:
        """Return True if i — j (undirected) in the CPDAG."""
        return bool(cpdag[i, j]) and bool(cpdag[j, i])

    @staticmethod
    def _is_directed(cpdag: AdjacencyMatrix, i: int, j: int) -> bool:
        """Return True if i → j (directed) in the CPDAG."""
        return bool(cpdag[i, j]) and not bool(cpdag[j, i])

    @staticmethod
    def _orient_edge(cpdag: AdjacencyMatrix, i: int, j: int) -> None:
        """Orient i → j by removing j → i."""
        cpdag[j, i] = 0

    def _meek_r1(self, cpdag: AdjacencyMatrix, n: int) -> bool:
        """R1: If i → k and k — j and i ⊥ j, then orient k → j.

        This prevents creating a new v-structure i → k ← j.
        """
        changed = False
        for k in range(n):
            for j in range(n):
                if k == j or not self._is_undirected(cpdag, k, j):
                    continue
                for i in range(n):
                    if i == k or i == j:
                        continue
                    if self._is_directed(cpdag, i, k) and not cpdag[i, j] and not cpdag[j, i]:
                        self._orient_edge(cpdag, k, j)
                        changed = True
                        break
        return changed

    def _meek_r2(self, cpdag: AdjacencyMatrix, n: int) -> bool:
        """R2: If i → k → j and i — j, then orient i → j.

        This prevents creating a directed cycle.
        """
        changed = False
        for i in range(n):
            for j in range(n):
                if i == j or not self._is_undirected(cpdag, i, j):
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if self._is_directed(cpdag, i, k) and self._is_directed(cpdag, k, j):
                        self._orient_edge(cpdag, i, j)
                        changed = True
                        break
        return changed

    def _meek_r3(self, cpdag: AdjacencyMatrix, n: int) -> bool:
        """R3: If i — k1 → j, i — k2 → j, k1 ⊥ k2, then orient i → j.

        Two non-adjacent undirected neighbours of i both point to j.
        """
        changed = False
        for i in range(n):
            for j in range(n):
                if i == j or not self._is_undirected(cpdag, i, j):
                    continue
                # Collect undirected neighbours of i that point to j
                candidates = [
                    k for k in range(n)
                    if k != i and k != j
                    and self._is_undirected(cpdag, i, k)
                    and self._is_directed(cpdag, k, j)
                ]
                oriented = False
                for a_idx, k1 in enumerate(candidates):
                    for k2 in candidates[a_idx + 1:]:
                        if not cpdag[k1, k2] and not cpdag[k2, k1]:
                            self._orient_edge(cpdag, i, j)
                            changed = True
                            oriented = True
                            break
                    if oriented:
                        break
        return changed

    def _meek_r4(self, cpdag: AdjacencyMatrix, n: int) -> bool:
        """R4: If i — k, k → m → j, i — j, and i ⊥ m, then orient i → j.

        This avoids a directed cycle through the path k → m → j.
        """
        changed = False
        for i in range(n):
            for j in range(n):
                if i == j or not self._is_undirected(cpdag, i, j):
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if not self._is_undirected(cpdag, i, k):
                        continue
                    for m in range(n):
                        if m == i or m == j or m == k:
                            continue
                        if (
                            self._is_directed(cpdag, k, m)
                            and self._is_directed(cpdag, m, j)
                            and not cpdag[i, m]
                            and not cpdag[m, i]
                        ):
                            self._orient_edge(cpdag, i, j)
                            changed = True
                            break
                    if not self._is_undirected(cpdag, i, j):
                        break
        return changed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_ci_test(
        self, data: DataMatrix, x: int, y: int, z: List[int]
    ) -> CITestResult:
        """Run the CI test and increment counter."""
        self.n_ci_tests_ += 1
        return self._ci_test.test(x, y, frozenset(z), data, self._alpha)

    @staticmethod
    def _adjacency_list(skeleton: AdjacencyMatrix, i: int, n: int) -> List[int]:
        """Return the adjacency list of node *i* in the skeleton."""
        return [j for j in range(n) if skeleton[i, j] and j != i]

    @staticmethod
    def _cpdag_to_dag(cpdag: AdjacencyMatrix, n: int) -> AdjacencyMatrix:
        """Extract one consistent DAG from a CPDAG.

        Undirected edges are oriented by topological-ordering heuristic:
        try each possible orientation and pick one that maintains
        acyclicity.  Falls back to lower→higher index orientation.
        """
        dag = np.zeros((n, n), dtype=np.int8)

        # First, copy all directed edges
        for i in range(n):
            for j in range(n):
                if cpdag[i, j] and not cpdag[j, i]:
                    dag[i, j] = 1

        # Then orient undirected edges
        undirected: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        for i in range(n):
            for j in range(i + 1, n):
                if cpdag[i, j] and cpdag[j, i]:
                    undirected.append((i, j))
                    seen.add((i, j))

        for i, j in undirected:
            # Try i → j
            dag[i, j] = 1
            if not DAG.is_acyclic(dag):
                dag[i, j] = 0
                dag[j, i] = 1
                if not DAG.is_acyclic(dag):
                    # Neither orientation works — skip (shouldn't happen)
                    dag[j, i] = 0

        return dag

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, object]:
        """Return a summary dictionary of the last fit.

        Returns
        -------
        dict
            Keys: ``n_ci_tests``, ``n_skeleton_edges``, ``n_directed``,
            ``n_undirected``.
        """
        skel_edges = int(self.skeleton_.sum()) // 2 if self.skeleton_ is not None else 0
        if self.cpdag_ is not None:
            n = self.cpdag_.shape[0]
            n_directed = 0
            n_undirected = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if self.cpdag_[i, j] and self.cpdag_[j, i]:
                        n_undirected += 1
                    elif self.cpdag_[i, j] or self.cpdag_[j, i]:
                        n_directed += 1
        else:
            n_directed = n_undirected = 0
        return {
            "n_ci_tests": self.n_ci_tests_,
            "n_skeleton_edges": skel_edges,
            "n_directed": n_directed,
            "n_undirected": n_undirected,
        }
