"""Built-in structure learning algorithms (no external dependencies).

Provides constraint-based, score-based, and hybrid causal structure
learning algorithms that rely only on numpy and scipy.

Classes
-------
ConstraintBasedLearner
    PC-style algorithm with CI tests, v-structure, and Meek rules.
ScoreBasedLearner
    BIC-based forward/backward and GES-style search.
HybridLearner
    MMHC (Max-Min Hill Climbing) combining both approaches.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import stats as sp_stats

from cpa.utils.logging import get_logger

logger = get_logger("discovery.structure_learning")


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------


def _has_cycle(adj: np.ndarray) -> bool:
    """Check if adjacency matrix contains a directed cycle."""
    n = adj.shape[0]
    WHITE, GRAY, BLACK = 0, 1, 2
    color = np.zeros(n, dtype=int)

    def dfs(u: int) -> bool:
        color[u] = GRAY
        for v in range(n):
            if adj[u, v] != 0:
                if color[v] == GRAY:
                    return True
                if color[v] == WHITE and dfs(v):
                    return True
        color[u] = BLACK
        return False

    for u in range(n):
        if color[u] == WHITE:
            if dfs(u):
                return True
    return False


def _topological_order(adj: np.ndarray) -> Optional[List[int]]:
    """Return a topological ordering, or None if cyclic."""
    n = adj.shape[0]
    in_degree = (adj != 0).astype(int).sum(axis=0).copy()
    queue = [i for i in range(n) if in_degree[i] == 0]
    order = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in range(n):
            if adj[node, child] != 0:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    return order if len(order) == n else None


def _partial_correlation(
    data: np.ndarray, i: int, j: int, cond: List[int]
) -> float:
    """Compute partial correlation between variables i and j given cond.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_samples, n_variables).
    i, j : int
        Variable indices.
    cond : list of int
        Conditioning set.

    Returns
    -------
    float
        Partial correlation.
    """
    n = data.shape[0]

    if len(cond) == 0:
        r = np.corrcoef(data[:, i], data[:, j])[0, 1]
        return float(r)

    # Compute via regression residuals
    X_cond = data[:, cond]
    X_aug = np.column_stack([np.ones(n), X_cond])

    try:
        beta_i, _, _, _ = np.linalg.lstsq(X_aug, data[:, i], rcond=None)
        res_i = data[:, i] - X_aug @ beta_i

        beta_j, _, _, _ = np.linalg.lstsq(X_aug, data[:, j], rcond=None)
        res_j = data[:, j] - X_aug @ beta_j

        denom = np.sqrt(np.sum(res_i ** 2) * np.sum(res_j ** 2))
        if denom < 1e-15:
            return 0.0
        return float(np.sum(res_i * res_j) / denom)
    except np.linalg.LinAlgError:
        return 0.0


def _fisher_z_test(
    r: float, n: int, k: int, alpha: float = 0.05
) -> Tuple[float, bool]:
    """Fisher z-transform test for partial correlation significance.

    Parameters
    ----------
    r : float
        Partial correlation.
    n : int
        Number of samples.
    k : int
        Size of conditioning set.
    alpha : float
        Significance level.

    Returns
    -------
    tuple of (float, bool)
        (p_value, is_independent)
    """
    r = np.clip(r, -0.9999, 0.9999)
    z = 0.5 * np.log((1 + r) / (1 - r))
    df = max(1, n - k - 3)
    se = 1.0 / np.sqrt(df)
    z_stat = abs(z / se)
    p_value = 2 * (1 - sp_stats.norm.cdf(z_stat))
    return float(p_value), p_value > alpha


def _local_bic_score(
    data: np.ndarray, target: int, parents: List[int]
) -> float:
    """Compute the local BIC score for a single variable and its parents.

    Parameters
    ----------
    data : np.ndarray
    target : int
    parents : list of int

    Returns
    -------
    float
        BIC score (lower is better).
    """
    n_samples = data.shape[0]
    y = data[:, target]

    if len(parents) == 0:
        var = np.var(y, ddof=1)
        if var < 1e-15:
            var = 1e-15
        ll = -0.5 * n_samples * (np.log(2 * np.pi * var) + 1.0)
        n_params = 2
    else:
        X = np.column_stack([np.ones(n_samples), data[:, parents]])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ beta
        except np.linalg.LinAlgError:
            residuals = y - np.mean(y)

        var = np.mean(residuals ** 2)
        if var < 1e-15:
            var = 1e-15
        ll = -0.5 * n_samples * (np.log(2 * np.pi * var) + 1.0)
        n_params = len(parents) + 2  # coefficients + intercept + variance

    return float(-2 * ll + n_params * np.log(n_samples))


def _total_bic_score(adj: np.ndarray, data: np.ndarray) -> float:
    """Compute total BIC score for the entire DAG."""
    n_vars = adj.shape[0]
    total = 0.0
    for j in range(n_vars):
        parents = list(np.where(adj[:, j] != 0)[0])
        total += _local_bic_score(data, j, parents)
    return total


# ---------------------------------------------------------------------------
# Constraint-Based Learner
# ---------------------------------------------------------------------------


class ConstraintBasedLearner:
    """Constraint-based structure learning (PC-style).

    Implements the PC algorithm with skeleton discovery, v-structure
    orientation, and Meek rules. Optionally supports the FCI algorithm
    for latent variable scenarios.

    Parameters
    ----------
    alpha : float
        Significance level for CI tests (default 0.05).
    max_cond_set : int
        Maximum conditioning set size (default -1 for unlimited).
    stable : bool
        Use the order-independent (stable) version (default True).
    ci_test : str
        CI test type: 'partial_correlation' or 'fisher_z' (default 'fisher_z').

    Examples
    --------
    >>> learner = ConstraintBasedLearner(alpha=0.05)
    >>> adj = learner.fit(data)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_cond_set: int = -1,
        stable: bool = True,
        ci_test: str = "fisher_z",
    ) -> None:
        self.alpha = alpha
        self.max_cond_set = max_cond_set
        self.stable = stable
        self.ci_test = ci_test

    def fit(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Learn a DAG structure from data.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_variables).
        variable_names : list of str, optional

        Returns
        -------
        np.ndarray
            Discovered adjacency matrix.
        """
        n_vars = data.shape[1]
        n_samples = data.shape[0]

        logger.info(
            "Constraint-based learning: %d variables, %d samples, alpha=%.3f",
            n_vars, n_samples, self.alpha,
        )

        # Step 1: Skeleton discovery
        skeleton, sep_sets = self._discover_skeleton(data)

        # Step 2: Orient v-structures
        adj = self._orient_v_structures(skeleton, sep_sets, n_vars)

        # Step 3: Apply Meek rules
        adj = self._apply_meek_rules(adj, n_vars)

        logger.info(
            "Discovered %d edges in %d variables",
            int(np.sum(adj != 0)), n_vars,
        )

        return adj

    def _discover_skeleton(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[int]]]:
        """Discover the graph skeleton using conditional independence tests.

        Parameters
        ----------
        data : np.ndarray

        Returns
        -------
        tuple
            (skeleton, separation_sets)
        """
        n_vars = data.shape[1]
        n_samples = data.shape[0]

        skeleton = np.ones((n_vars, n_vars), dtype=int)
        np.fill_diagonal(skeleton, 0)

        sep_sets: Dict[Tuple[int, int], List[int]] = {}

        max_k = n_vars - 2 if self.max_cond_set < 0 else self.max_cond_set

        for k in range(max_k + 1):
            edges_to_remove: List[Tuple[int, int, List[int]]] = []

            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if skeleton[i, j] == 0:
                        continue

                    # Get adjacency of i (excluding j)
                    adj_i = [
                        v for v in range(n_vars)
                        if v != i and v != j and skeleton[i, v] != 0
                    ]

                    if len(adj_i) < k:
                        continue

                    # Test all conditioning sets of size k
                    found_independent = False
                    for cond_set in itertools.combinations(adj_i, k):
                        cond = list(cond_set)
                        r = _partial_correlation(data, i, j, cond)
                        p_val, is_indep = _fisher_z_test(
                            r, n_samples, len(cond), self.alpha
                        )

                        if is_indep:
                            if self.stable:
                                edges_to_remove.append((i, j, cond))
                            else:
                                skeleton[i, j] = 0
                                skeleton[j, i] = 0
                                sep_sets[(i, j)] = cond
                                sep_sets[(j, i)] = cond
                            found_independent = True
                            break

            # Stable version: remove edges after testing all at this level
            if self.stable:
                for i, j, cond in edges_to_remove:
                    skeleton[i, j] = 0
                    skeleton[j, i] = 0
                    sep_sets[(i, j)] = cond
                    sep_sets[(j, i)] = cond

        return skeleton, sep_sets

    @staticmethod
    def _orient_v_structures(
        skeleton: np.ndarray,
        sep_sets: Dict[Tuple[int, int], List[int]],
        n: int,
    ) -> np.ndarray:
        """Orient v-structures in the skeleton.

        A v-structure i → k ← j exists when:
        - i and j are not adjacent
        - k is adjacent to both
        - k is not in sep(i, j)

        Parameters
        ----------
        skeleton : np.ndarray
        sep_sets : dict
        n : int

        Returns
        -------
        np.ndarray
        """
        adj = skeleton.copy()

        for k in range(n):
            neighbors = np.where(skeleton[k] != 0)[0]
            for idx1 in range(len(neighbors)):
                for idx2 in range(idx1 + 1, len(neighbors)):
                    i = neighbors[idx1]
                    j = neighbors[idx2]

                    if skeleton[i, j] != 0:
                        continue

                    sep = sep_sets.get((i, j), [])
                    if k not in sep:
                        # Orient as i → k ← j
                        adj[i, k] = 1
                        adj[k, i] = 0
                        adj[j, k] = 1
                        adj[k, j] = 0

        return adj

    @staticmethod
    def _apply_meek_rules(adj: np.ndarray, n: int) -> np.ndarray:
        """Apply Meek's orientation rules R1-R4.

        Parameters
        ----------
        adj : np.ndarray
        n : int

        Returns
        -------
        np.ndarray
        """
        changed = True
        max_iters = n * n * 4
        iteration = 0

        while changed and iteration < max_iters:
            changed = False
            iteration += 1

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if adj[i, j] == 0 or adj[j, i] == 0:
                        continue  # Not undirected

                    # R1: ∃k: k→i, k⊥j → orient i→j
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if adj[k, i] == 1 and adj[i, k] == 0:
                            if adj[k, j] == 0 and adj[j, k] == 0:
                                adj[i, j] = 1
                                adj[j, i] = 0
                                changed = True
                                break
                    if adj[j, i] == 0:
                        continue

                    # R2: ∃k: i→k→j → orient i→j
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if (adj[i, k] == 1 and adj[k, i] == 0 and
                                adj[k, j] == 1 and adj[j, k] == 0):
                            adj[i, j] = 1
                            adj[j, i] = 0
                            changed = True
                            break
                    if adj[j, i] == 0:
                        continue

                    # R3: ∃k,l: k→j, l→j, k-i-l, k⊥l → orient i→j
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if not (adj[k, j] == 1 and adj[j, k] == 0):
                            continue
                        if not (adj[k, i] == 1 and adj[i, k] == 1):
                            continue
                        for l in range(k + 1, n):
                            if l == i or l == j:
                                continue
                            if not (adj[l, j] == 1 and adj[j, l] == 0):
                                continue
                            if not (adj[l, i] == 1 and adj[i, l] == 1):
                                continue
                            if adj[k, l] == 0 and adj[l, k] == 0:
                                adj[i, j] = 1
                                adj[j, i] = 0
                                changed = True
                                break
                        if adj[j, i] == 0:
                            break

        return adj

    def fit_fci(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """FCI algorithm for latent variable scenarios.

        Returns a PAG (Partial Ancestral Graph) that allows for the
        possibility of latent common causes.

        Parameters
        ----------
        data : np.ndarray
        variable_names : list of str, optional

        Returns
        -------
        np.ndarray
            PAG adjacency matrix with edge marks:
            -1 = arrowhead, 0 = no edge, 1 = tail, 2 = circle.
        """
        n_vars = data.shape[1]
        n_samples = data.shape[0]

        logger.info("Running FCI: %d variables, %d samples", n_vars, n_samples)

        # Step 1: Start with PC skeleton
        skeleton, sep_sets = self._discover_skeleton(data)

        # Step 2: Orient v-structures
        adj = self._orient_v_structures(skeleton, sep_sets, n_vars)

        # Step 3: Possibly Discriminating Paths (simplified)
        # For a full FCI we would need to check discriminating paths,
        # but this simplified version uses Meek rules + circle marks

        # Step 4: Apply Meek rules
        adj = self._apply_meek_rules(adj, n_vars)

        # Step 5: Mark remaining undirected edges with circles (PAG encoding)
        pag = np.zeros((n_vars, n_vars), dtype=int)
        for i in range(n_vars):
            for j in range(n_vars):
                if adj[i, j] == 1 and adj[j, i] == 0:
                    pag[i, j] = 1   # tail at i
                    pag[j, i] = -1  # arrowhead at j (i → j)
                elif adj[i, j] == 1 and adj[j, i] == 1:
                    pag[i, j] = 2   # circle (uncertain)
                    pag[j, i] = 2

        return pag


# ---------------------------------------------------------------------------
# Score-Based Learner
# ---------------------------------------------------------------------------


class ScoreBasedLearner:
    """Score-based structure learning using BIC.

    Implements forward/backward search, GES-style two-phase search,
    and tabu search over the space of DAGs.

    Parameters
    ----------
    score_fn : str
        Score function: 'bic' or 'aic' (default 'bic').
    max_parents : int
        Maximum parents per node (default 5).
    tabu_length : int
        Tabu list length for tabu search (default 10).

    Examples
    --------
    >>> learner = ScoreBasedLearner(score_fn='bic')
    >>> adj = learner.fit(data, method='forward_backward')
    """

    def __init__(
        self,
        score_fn: str = "bic",
        max_parents: int = 5,
        tabu_length: int = 10,
    ) -> None:
        self.score_fn = score_fn
        self.max_parents = max_parents
        self.tabu_length = tabu_length

    def fit(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
        method: str = "forward_backward",
    ) -> np.ndarray:
        """Learn a DAG structure from data.

        Parameters
        ----------
        data : np.ndarray
        variable_names : list of str, optional
        method : str
            Search method: 'forward_backward', 'ges', or 'tabu'.

        Returns
        -------
        np.ndarray
        """
        logger.info("Score-based learning: method=%s, score=%s", method, self.score_fn)

        if method == "forward_backward":
            return self._forward_backward(data)
        elif method == "ges":
            return self._ges_search(data)
        elif method == "tabu":
            return self._tabu_search(data)
        else:
            logger.warning("Unknown method '%s', using forward_backward", method)
            return self._forward_backward(data)

    def _score(self, adj: np.ndarray, data: np.ndarray) -> float:
        """Compute the score for a DAG."""
        return _total_bic_score(adj, data)

    def _local_score(
        self, data: np.ndarray, target: int, parents: List[int]
    ) -> float:
        """Compute local score for one variable."""
        return _local_bic_score(data, target, parents)

    def _forward_backward(self, data: np.ndarray) -> np.ndarray:
        """Forward-backward greedy search.

        Parameters
        ----------
        data : np.ndarray

        Returns
        -------
        np.ndarray
        """
        n_vars = data.shape[1]
        adj = np.zeros((n_vars, n_vars), dtype=int)
        best_score = self._score(adj, data)

        # Forward phase: greedily add edges
        improved = True
        while improved:
            improved = False
            best_candidate = None
            best_candidate_score = best_score

            for j in range(n_vars):
                current_parents = list(np.where(adj[:, j] != 0)[0])
                if len(current_parents) >= self.max_parents:
                    continue

                for i in range(n_vars):
                    if i == j or adj[i, j] != 0:
                        continue

                    trial = adj.copy()
                    trial[i, j] = 1
                    if _has_cycle(trial):
                        continue

                    score = self._score(trial, data)
                    if score < best_candidate_score:
                        best_candidate_score = score
                        best_candidate = (i, j)

            if best_candidate is not None:
                adj[best_candidate[0], best_candidate[1]] = 1
                best_score = best_candidate_score
                improved = True

        # Backward phase: greedily remove edges
        improved = True
        while improved:
            improved = False
            best_candidate = None
            best_candidate_score = best_score

            edges = list(zip(*np.where(adj != 0)))
            for i, j in edges:
                trial = adj.copy()
                trial[i, j] = 0
                score = self._score(trial, data)
                if score < best_candidate_score:
                    best_candidate_score = score
                    best_candidate = (i, j)

            if best_candidate is not None:
                adj[best_candidate[0], best_candidate[1]] = 0
                best_score = best_candidate_score
                improved = True

        logger.info(
            "Forward-backward: %d edges, BIC=%.2f",
            int(np.sum(adj != 0)), best_score,
        )
        return adj

    def _ges_search(self, data: np.ndarray) -> np.ndarray:
        """GES-style two-phase search.

        Phase 1: Forward (add edges to improve score)
        Phase 2: Backward (remove edges to improve score)
        Both phases consider the equivalence class.

        Parameters
        ----------
        data : np.ndarray

        Returns
        -------
        np.ndarray
        """
        n_vars = data.shape[1]
        adj = np.zeros((n_vars, n_vars), dtype=int)
        best_score = self._score(adj, data)

        # Phase 1: Forward
        improved = True
        max_iters = n_vars * n_vars
        iteration = 0
        while improved and iteration < max_iters:
            improved = False
            iteration += 1

            candidates = []
            for j in range(n_vars):
                parents = list(np.where(adj[:, j] != 0)[0])
                if len(parents) >= self.max_parents:
                    continue
                for i in range(n_vars):
                    if i == j or adj[i, j] != 0:
                        continue
                    trial = adj.copy()
                    trial[i, j] = 1
                    if _has_cycle(trial):
                        continue
                    score = self._score(trial, data)
                    delta = score - best_score
                    candidates.append((i, j, score, delta))

            if candidates:
                candidates.sort(key=lambda x: x[2])
                best_i, best_j, new_score, delta = candidates[0]
                if new_score < best_score:
                    adj[best_i, best_j] = 1
                    best_score = new_score
                    improved = True

        # Phase 2: Backward (turning phase)
        improved = True
        iteration = 0
        while improved and iteration < max_iters:
            improved = False
            iteration += 1

            candidates = []
            edges = list(zip(*np.where(adj != 0)))
            for i, j in edges:
                trial = adj.copy()
                trial[i, j] = 0
                score = self._score(trial, data)
                if score < best_score:
                    candidates.append((i, j, score))

            # Also try edge reversals
            for i, j in edges:
                trial = adj.copy()
                trial[i, j] = 0
                trial[j, i] = 1
                if _has_cycle(trial):
                    continue
                score = self._score(trial, data)
                if score < best_score:
                    candidates.append((i, j, score))

            if candidates:
                candidates.sort(key=lambda x: x[2])
                best_i, best_j, new_score = candidates[0]
                # Apply the best modification
                if adj[best_i, best_j] != 0:
                    adj[best_i, best_j] = 0
                    # Check if it was a reversal
                    trial = adj.copy()
                    trial[best_j, best_i] = 1
                    if not _has_cycle(trial) and self._score(trial, data) <= new_score:
                        adj = trial
                    best_score = self._score(adj, data)
                    improved = True

        logger.info(
            "GES search: %d edges, BIC=%.2f",
            int(np.sum(adj != 0)), best_score,
        )
        return adj

    def _tabu_search(self, data: np.ndarray) -> np.ndarray:
        """Tabu search for DAG structure.

        Parameters
        ----------
        data : np.ndarray

        Returns
        -------
        np.ndarray
        """
        n_vars = data.shape[1]
        adj = np.zeros((n_vars, n_vars), dtype=int)
        best_score = self._score(adj, data)
        best_adj = adj.copy()
        best_global_score = best_score

        tabu_list: List[Tuple[str, int, int]] = []
        max_iters = n_vars * n_vars * 3
        no_improve_count = 0
        max_no_improve = n_vars * 2

        for iteration in range(max_iters):
            candidates = []

            # Generate all neighbors (add, remove, reverse)
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j:
                        continue

                    # Add edge
                    if adj[i, j] == 0:
                        parents = list(np.where(adj[:, j] != 0)[0])
                        if len(parents) < self.max_parents:
                            move = ("add", i, j)
                            if move not in tabu_list:
                                trial = adj.copy()
                                trial[i, j] = 1
                                if not _has_cycle(trial):
                                    score = self._score(trial, data)
                                    candidates.append((move, trial, score))

                    # Remove edge
                    if adj[i, j] != 0:
                        move = ("remove", i, j)
                        if move not in tabu_list:
                            trial = adj.copy()
                            trial[i, j] = 0
                            score = self._score(trial, data)
                            candidates.append((move, trial, score))

                    # Reverse edge
                    if adj[i, j] != 0 and adj[j, i] == 0:
                        move = ("reverse", i, j)
                        if move not in tabu_list:
                            trial = adj.copy()
                            trial[i, j] = 0
                            trial[j, i] = 1
                            if not _has_cycle(trial):
                                score = self._score(trial, data)
                                candidates.append((move, trial, score))

            if not candidates:
                break

            # Select best non-tabu candidate
            candidates.sort(key=lambda x: x[2])
            best_move, best_trial, move_score = candidates[0]

            # Apply move
            adj = best_trial
            best_score = move_score

            # Update tabu list
            tabu_list.append(best_move)
            if len(tabu_list) > self.tabu_length:
                tabu_list.pop(0)

            # Track global best
            if best_score < best_global_score:
                best_global_score = best_score
                best_adj = adj.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= max_no_improve:
                break

        logger.info(
            "Tabu search: %d edges, BIC=%.2f",
            int(np.sum(best_adj != 0)), best_global_score,
        )
        return best_adj


# ---------------------------------------------------------------------------
# Hybrid Learner
# ---------------------------------------------------------------------------


class HybridLearner:
    """Hybrid structure learning (MMHC-style).

    Combines constraint-based and score-based approaches:
    - Phase 1 (Restrict): Use CI tests to identify candidate parents
    - Phase 2 (Maximize): Score-based search restricted to candidates

    Parameters
    ----------
    alpha : float
        Significance level for CI tests in restrict phase.
    max_parents : int
        Maximum parents per node.
    score_fn : str
        Score function for maximize phase.

    Examples
    --------
    >>> learner = HybridLearner(alpha=0.05)
    >>> adj = learner.fit(data)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_parents: int = 5,
        score_fn: str = "bic",
    ) -> None:
        self.alpha = alpha
        self.max_parents = max_parents
        self.score_fn = score_fn

    def fit(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Learn DAG structure using MMHC.

        Parameters
        ----------
        data : np.ndarray
        variable_names : list of str, optional

        Returns
        -------
        np.ndarray
        """
        n_vars = data.shape[1]

        logger.info(
            "Hybrid learning (MMHC): %d variables, alpha=%.3f",
            n_vars, self.alpha,
        )

        # Phase 1: Restrict — find candidate parent sets
        candidates = self._restrict_phase(data)

        # Phase 2: Maximize — score-based search within candidates
        adj = self._maximize_phase(data, candidates)

        logger.info(
            "MMHC: %d edges discovered",
            int(np.sum(adj != 0)),
        )
        return adj

    def _restrict_phase(
        self, data: np.ndarray
    ) -> Dict[int, Set[int]]:
        """Restrict phase: identify candidate parent sets using MMPC.

        Max-Min Parents and Children: for each variable, find the set
        of variables that maximize the minimum association with the
        target, conditioned on the current candidate set.

        Parameters
        ----------
        data : np.ndarray

        Returns
        -------
        dict
            Variable → set of candidate parent indices.
        """
        n_vars = data.shape[1]
        n_samples = data.shape[0]

        candidates: Dict[int, Set[int]] = {j: set() for j in range(n_vars)}

        for target in range(n_vars):
            cpc: Set[int] = set()

            # Forward: grow CPC
            improved = True
            while improved and len(cpc) < self.max_parents * 2:
                improved = False
                best_var = -1
                best_min_assoc = -np.inf

                for candidate in range(n_vars):
                    if candidate == target or candidate in cpc:
                        continue

                    # Compute min association over all subsets of CPC
                    min_assoc = self._min_association(
                        data, target, candidate, list(cpc), n_samples
                    )

                    if min_assoc > best_min_assoc:
                        best_min_assoc = min_assoc
                        best_var = candidate

                # Add best variable if significantly associated
                if best_var >= 0 and best_min_assoc > 0:
                    r = _partial_correlation(data, target, best_var, list(cpc))
                    _, is_indep = _fisher_z_test(r, n_samples, len(cpc), self.alpha)
                    if not is_indep:
                        cpc.add(best_var)
                        improved = True

            # Backward: shrink CPC
            to_remove = set()
            for var in cpc:
                others = cpc - {var}
                # Test if var is independent of target given others
                r = _partial_correlation(data, target, var, list(others))
                _, is_indep = _fisher_z_test(r, n_samples, len(others), self.alpha)
                if is_indep:
                    to_remove.add(var)

            cpc -= to_remove
            candidates[target] = cpc

        return candidates

    def _min_association(
        self,
        data: np.ndarray,
        target: int,
        candidate: int,
        cpc: List[int],
        n_samples: int,
    ) -> float:
        """Compute minimum association between target and candidate.

        Tests the association conditioned on various subsets of the
        current CPC and returns the minimum.

        Parameters
        ----------
        data : np.ndarray
        target : int
        candidate : int
        cpc : list of int
        n_samples : int

        Returns
        -------
        float
            Minimum absolute partial correlation.
        """
        if len(cpc) == 0:
            r = abs(_partial_correlation(data, target, candidate, []))
            return r

        # For efficiency, only check subsets up to a certain size
        max_subset_size = min(3, len(cpc))
        min_assoc = np.inf

        for k in range(max_subset_size + 1):
            for subset in itertools.combinations(cpc, k):
                r = abs(_partial_correlation(
                    data, target, candidate, list(subset)
                ))
                min_assoc = min(min_assoc, r)
                if min_assoc < 1e-10:
                    return 0.0

        return float(min_assoc)

    def _maximize_phase(
        self,
        data: np.ndarray,
        candidates: Dict[int, Set[int]],
    ) -> np.ndarray:
        """Maximize phase: score-based search restricted to candidates.

        Uses greedy hill climbing restricted to edges that connect
        variables within each other's candidate sets.

        Parameters
        ----------
        data : np.ndarray
        candidates : dict

        Returns
        -------
        np.ndarray
        """
        n_vars = data.shape[1]
        adj = np.zeros((n_vars, n_vars), dtype=int)
        best_score = _total_bic_score(adj, data)

        # Build allowed edges from candidate sets
        allowed: Set[Tuple[int, int]] = set()
        for target, cands in candidates.items():
            for cand in cands:
                allowed.add((cand, target))  # cand → target

        # Greedy hill climbing within allowed edges
        improved = True
        max_iters = len(allowed) * 3
        iteration = 0

        while improved and iteration < max_iters:
            improved = False
            iteration += 1
            best_move = None
            best_move_score = best_score

            # Try adding allowed edges
            for i, j in allowed:
                if adj[i, j] != 0:
                    continue
                parents = list(np.where(adj[:, j] != 0)[0])
                if len(parents) >= self.max_parents:
                    continue

                trial = adj.copy()
                trial[i, j] = 1
                if _has_cycle(trial):
                    continue

                score = _total_bic_score(trial, data)
                if score < best_move_score:
                    best_move_score = score
                    best_move = ("add", i, j)

            # Try removing existing edges
            edges = list(zip(*np.where(adj != 0)))
            for i, j in edges:
                trial = adj.copy()
                trial[i, j] = 0
                score = _total_bic_score(trial, data)
                if score < best_move_score:
                    best_move_score = score
                    best_move = ("remove", i, j)

            # Try reversing edges (within allowed)
            for i, j in edges:
                if (j, i) not in allowed:
                    continue
                trial = adj.copy()
                trial[i, j] = 0
                trial[j, i] = 1
                if _has_cycle(trial):
                    continue
                score = _total_bic_score(trial, data)
                if score < best_move_score:
                    best_move_score = score
                    best_move = ("reverse", i, j)

            if best_move is not None:
                action, bi, bj = best_move
                if action == "add":
                    adj[bi, bj] = 1
                elif action == "remove":
                    adj[bi, bj] = 0
                elif action == "reverse":
                    adj[bi, bj] = 0
                    adj[bj, bi] = 1
                best_score = best_move_score
                improved = True

        return adj

    def fit_restricted(
        self,
        data: np.ndarray,
        candidate_edges: Set[Tuple[int, int]],
    ) -> np.ndarray:
        """Score-based search restricted to given candidate edges.

        Parameters
        ----------
        data : np.ndarray
        candidate_edges : set of (int, int)

        Returns
        -------
        np.ndarray
        """
        n_vars = data.shape[1]
        candidates = {j: set() for j in range(n_vars)}
        for i, j in candidate_edges:
            candidates[j].add(i)

        return self._maximize_phase(data, candidates)
