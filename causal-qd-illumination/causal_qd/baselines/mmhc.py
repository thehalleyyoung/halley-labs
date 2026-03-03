"""Max-Min Hill-Climbing (MMHC) hybrid causal discovery algorithm.

Implements the full MMHC algorithm (Tsamardinos, Brown & Aliferis, 2006)
with two phases:
  1. **Restrict** — MMPC (Max-Min Parents and Children) learns an undirected
     skeleton via conditional independence tests.
  2. **Maximize** — Greedy hill-climbing over DAGs restricted to the
     skeleton, scored by a decomposable scoring function.

Includes efficient score caching, multi-restart hill-climbing option,
and configurable depth limits for the CI testing phase.

References
----------
Tsamardinos, I., Brown, L. E. & Aliferis, C. F. (2006).
    The max-min hill-climbing Bayesian network structure learning
    algorithm.  *Machine Learning*, 65(1), 31-78.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from causal_qd.ci_tests.ci_base import CITest, CITestResult
from causal_qd.core.dag import DAG
from causal_qd.scores.score_base import DecomposableScore
from causal_qd.types import AdjacencyMatrix, DataMatrix

logger = logging.getLogger(__name__)


# ======================================================================
# Fallback CI test and scorer
# ======================================================================


class _FallbackCITest(CITest):
    """Simple marginal-correlation CI test (fallback)."""

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
            return CITestResult(0.0, 1.0, True, conditioning_set)

        indices = sorted({x, y} | set(conditioning_set))
        sub = data[:, indices]
        cov = np.cov(sub, rowvar=False)
        cov += 1e-10 * np.eye(cov.shape[0])
        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return CITestResult(0.0, 1.0, True, conditioning_set)

        idx_map = {v: i for i, v in enumerate(indices)}
        ix, iy = idx_map[x], idx_map[y]
        denom = math.sqrt(abs(precision[ix, ix] * precision[iy, iy]))
        r = float(-precision[ix, iy] / denom) if denom > 1e-15 else 0.0
        r = max(-1 + 1e-12, min(1 - 1e-12, r))
        z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(dof)
        p = float(2.0 * sp_stats.norm.sf(abs(z)))
        return CITestResult(z, p, p >= alpha, conditioning_set)


class _FallbackBIC(DecomposableScore):
    """Gaussian BIC (fallback scorer)."""

    def local_score(self, node: int, parents: List[int], data: DataMatrix) -> float:
        n_samples = data.shape[0]
        y = data[:, node]
        k = len(parents)
        if k > 0:
            X = data[:, parents]
            try:
                coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                residuals = y - X @ coef
            except np.linalg.LinAlgError:
                residuals = y - y.mean()
        else:
            residuals = y - y.mean()
        rss = max(float(np.sum(residuals ** 2)), 1e-12)
        return -n_samples / 2.0 * np.log(rss / n_samples) - (k + 1) * np.log(n_samples) / 2.0


# ======================================================================
# MMHC Algorithm
# ======================================================================


class MMHCAlgorithm:
    """Max-Min Hill-Climbing algorithm (Tsamardinos et al., 2006).

    Parameters
    ----------
    ci_test : CITest or None
        Conditional independence test for the restrict phase.
        If *None*, a partial-correlation test is used.
    score_fn : DecomposableScore or None
        Decomposable score for the maximize phase.
        If *None*, Gaussian BIC is used.
    alpha : float
        Significance level for the CI tests (default 0.05).
    max_cond_depth : int
        Maximum conditioning set size in MMPC (default 3).
    max_hc_iter : int
        Maximum hill-climbing iterations (default 5000).
    n_restarts : int
        Number of random restarts for hill-climbing (default 1).
    tabu_length : int
        Tabu list length (0 = no tabu, default 0).
    verbose : bool
        Emit progress via *logging*.
    """

    def __init__(
        self,
        ci_test: Optional[CITest] = None,
        score_fn: Optional[DecomposableScore] = None,
        alpha: float = 0.05,
        max_cond_depth: int = 3,
        max_hc_iter: int = 5_000,
        n_restarts: int = 1,
        tabu_length: int = 0,
        verbose: bool = False,
    ) -> None:
        self._ci_test: CITest = ci_test if ci_test is not None else _FallbackCITest()
        self._score_fn: DecomposableScore = score_fn if score_fn is not None else _FallbackBIC()
        self._alpha = alpha
        self._max_depth = max_cond_depth
        self._max_hc_iter = max_hc_iter
        self._n_restarts = max(1, n_restarts)
        self._tabu_length = tabu_length
        self._verbose = verbose
        # Diagnostics
        self.skeleton_: Optional[AdjacencyMatrix] = None
        self.pc_sets_: Optional[List[Set[int]]] = None
        self.n_ci_tests_: int = 0
        self.n_hc_steps_: int = 0
        # Score cache
        self._score_cache: Dict[Tuple[int, FrozenSet[int]], float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data: DataMatrix) -> DAG:
        """Run MMHC on *data* and return the learned DAG.

        Parameters
        ----------
        data : DataMatrix

        Returns
        -------
        DAG
        """
        n = data.shape[1]
        self.n_ci_tests_ = 0
        self.n_hc_steps_ = 0
        self._score_cache.clear()

        skeleton = self.restrict_phase(data, n)
        self.skeleton_ = skeleton.copy()

        dag_adj = self.maximize_phase(data, skeleton, n)
        return DAG(dag_adj)

    def run(self, data: DataMatrix) -> AdjacencyMatrix:
        """Run MMHC and return adjacency matrix."""
        return self.fit(data).adjacency

    # ------------------------------------------------------------------
    # RESTRICT PHASE: MMPC
    # ------------------------------------------------------------------

    def max_min_parents_children(
        self, data: DataMatrix, target: int, n: int,
    ) -> Set[int]:
        """Identify the parents-and-children set of *target* using MMPC.

        **Forward pass**: iteratively add the variable with the highest
        minimum association (max-min heuristic).  Association is
        measured as ``1 − max_p_value`` over conditioning subsets of
        the current PC set.

        **Backward pass**: remove any variable that becomes
        conditionally independent of *target* given a subset of the
        remaining PC set.

        Parameters
        ----------
        data : DataMatrix
        target : int
        n : int

        Returns
        -------
        set of int
            Estimated parent/child set.
        """
        pc: Set[int] = set()
        candidates = set(range(n)) - {target}

        # Forward: add associated variables
        changed = True
        while changed and candidates - pc:
            changed = False
            best_var = -1
            best_min_assoc = -1.0

            for x in candidates - pc:
                min_assoc = self._min_association(data, target, x, list(pc))
                if min_assoc > best_min_assoc:
                    best_min_assoc = min_assoc
                    best_var = x

            if best_var >= 0 and best_min_assoc > 1.0 - self._alpha:
                pc.add(best_var)
                changed = True
                if self._verbose:
                    logger.info("MMPC fwd: target=%d, add %d (assoc=%.4f)",
                                target, best_var, best_min_assoc)

        # Backward: remove spurious variables
        for x in list(pc):
            remaining = list(pc - {x})
            max_depth = min(len(remaining), self._max_depth)
            removed = False
            for depth in range(max_depth + 1):
                for subset in combinations(remaining, depth):
                    result = self._run_ci_test(data, target, x, list(subset))
                    if result.is_independent:
                        pc.discard(x)
                        removed = True
                        if self._verbose:
                            logger.info("MMPC bwd: target=%d, remove %d", target, x)
                        break
                if removed:
                    break

        return pc

    def _min_association(
        self, data: DataMatrix, target: int, x: int, current_pc: List[int],
    ) -> float:
        """Compute the max-min association heuristic.

        For each conditioning subset of *current_pc* (up to depth
        ``self._max_depth``), test CI of target ⊥ x | subset.  The
        association is ``1 − p_value``.  Return the *minimum*
        association over all subsets (i.e. worst-case dependence).
        """
        max_depth = min(len(current_pc), self._max_depth)
        min_assoc = float("inf")

        for depth in range(max_depth + 1):
            for subset in combinations(current_pc, depth):
                result = self._run_ci_test(data, target, x, list(subset))
                assoc = 1.0 - result.p_value
                min_assoc = min(min_assoc, assoc)

        return min_assoc if min_assoc != float("inf") else 0.0

    def restrict_phase(self, data: DataMatrix, n: int) -> AdjacencyMatrix:
        """Learn an undirected skeleton using MMPC for each node.

        Parameters
        ----------
        data : DataMatrix
        n : int

        Returns
        -------
        AdjacencyMatrix
            Symmetric skeleton adjacency matrix.
        """
        pc_sets: List[Set[int]] = []
        for target in range(n):
            pc = self.max_min_parents_children(data, target, n)
            pc_sets.append(pc)

        self.pc_sets_ = pc_sets

        # Symmetrise
        skeleton = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in pc_sets[i]:
                if i in pc_sets[j]:
                    skeleton[i, j] = 1
                    skeleton[j, i] = 1

        return skeleton

    # ------------------------------------------------------------------
    # MAXIMIZE PHASE: Hill-Climbing
    # ------------------------------------------------------------------

    def hill_climbing(
        self,
        data: DataMatrix,
        skeleton: AdjacencyMatrix,
        n: int,
        init_adj: Optional[AdjacencyMatrix] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[AdjacencyMatrix, float]:
        """Greedy hill-climbing over DAGs restricted to skeleton edges.

        Supports add, remove, and reverse operations with optional
        tabu list to escape local optima.

        Parameters
        ----------
        data : DataMatrix
        skeleton : AdjacencyMatrix
        n : int
        init_adj : AdjacencyMatrix or None
            Starting DAG (default: empty).
        rng : Generator or None
            RNG for random restarts.

        Returns
        -------
        adj : AdjacencyMatrix
        score : float
        """
        if init_adj is not None:
            adj = init_adj.copy()
        else:
            adj = np.zeros((n, n), dtype=np.int8)

        current_score = self._total_score(data, adj, n)
        tabu: List[Tuple[str, int, int]] = []

        for iteration in range(self._max_hc_iter):
            best_gain = 0.0
            best_op: Optional[Tuple[str, int, int]] = None

            for i in range(n):
                for j in range(n):
                    if i == j or not skeleton[i, j]:
                        continue

                    if not adj[i, j]:
                        # Try add i → j
                        op_key = ("add", i, j)
                        if op_key in tabu:
                            continue
                        adj[i, j] = 1
                        if DAG.is_acyclic(adj):
                            gain = self._total_score(data, adj, n) - current_score
                            if gain > best_gain:
                                best_gain = gain
                                best_op = op_key
                        adj[i, j] = 0
                    else:
                        # Try remove i → j
                        op_key_rm = ("remove", i, j)
                        if op_key_rm not in tabu:
                            adj[i, j] = 0
                            gain = self._total_score(data, adj, n) - current_score
                            if gain > best_gain:
                                best_gain = gain
                                best_op = op_key_rm
                            adj[i, j] = 1

                        # Try reverse i → j to j → i
                        op_key_rev = ("reverse", i, j)
                        if op_key_rev not in tabu and skeleton[j, i]:
                            adj[i, j] = 0
                            adj[j, i] = 1
                            if DAG.is_acyclic(adj):
                                gain = self._total_score(data, adj, n) - current_score
                                if gain > best_gain:
                                    best_gain = gain
                                    best_op = op_key_rev
                            adj[j, i] = 0
                            adj[i, j] = 1

            if best_op is None:
                break

            # Apply best operation
            op, i, j = best_op
            if op == "add":
                adj[i, j] = 1
            elif op == "remove":
                adj[i, j] = 0
            elif op == "reverse":
                adj[i, j] = 0
                adj[j, i] = 1
            current_score += best_gain
            self.n_hc_steps_ += 1

            # Maintain tabu list
            if self._tabu_length > 0:
                tabu.append(best_op)
                if len(tabu) > self._tabu_length:
                    tabu.pop(0)

        return adj, current_score

    def maximize_phase(
        self, data: DataMatrix, skeleton: AdjacencyMatrix, n: int,
    ) -> AdjacencyMatrix:
        """Maximize phase with optional multi-restart.

        Parameters
        ----------
        data : DataMatrix
        skeleton : AdjacencyMatrix
        n : int

        Returns
        -------
        AdjacencyMatrix
        """
        rng = np.random.default_rng(42)
        best_adj = None
        best_score = -np.inf

        for restart in range(self._n_restarts):
            if restart == 0:
                init = None
            else:
                # Random starting DAG from skeleton
                init = self._random_dag_from_skeleton(skeleton, n, rng)
            adj, score = self.hill_climbing(data, skeleton, n, init, rng)
            if score > best_score:
                best_score = score
                best_adj = adj.copy()

        return best_adj if best_adj is not None else np.zeros((n, n), dtype=np.int8)

    @staticmethod
    def _random_dag_from_skeleton(
        skeleton: AdjacencyMatrix, n: int, rng: np.random.Generator,
    ) -> AdjacencyMatrix:
        """Generate a random DAG by directing skeleton edges along a random ordering."""
        perm = rng.permutation(n)
        order = {node: idx for idx, node in enumerate(perm)}
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if skeleton[i, j]:
                    if order[i] < order[j]:
                        adj[i, j] = 1
                    else:
                        adj[j, i] = 1
        return adj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_ci_test(
        self, data: DataMatrix, x: int, y: int, z: List[int],
    ) -> CITestResult:
        self.n_ci_tests_ += 1
        return self._ci_test.test(x, y, frozenset(z), data, self._alpha)

    def _cached_local_score(
        self, data: DataMatrix, node: int, parents: List[int],
    ) -> float:
        key = (node, frozenset(parents))
        val = self._score_cache.get(key)
        if val is not None:
            return val
        val = self._score_fn.local_score(node, parents, data)
        self._score_cache[key] = val
        return val

    def _total_score(
        self, data: DataMatrix, adj: AdjacencyMatrix, n: int,
    ) -> float:
        total = 0.0
        for j in range(n):
            parents = sorted(int(i) for i in np.nonzero(adj[:, j])[0])
            total += self._cached_local_score(data, j, parents)
        return total

    def summary(self) -> Dict[str, object]:
        """Return diagnostics from the last fit."""
        return {
            "n_ci_tests": self.n_ci_tests_,
            "n_hc_steps": self.n_hc_steps_,
            "skeleton_edges": int(self.skeleton_.sum()) // 2 if self.skeleton_ is not None else 0,
        }
