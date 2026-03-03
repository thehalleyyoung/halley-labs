"""BDeu score for discrete / categorical data.

Implements the Bayesian Dirichlet equivalent uniform (BDeu) score,
which is the standard Bayesian score for discrete Bayesian networks
with a symmetric Dirichlet prior.

The BDeu score for a node X_i with r_i states and parent set Pa_i
having q_i joint configurations is:

    BDeu(i, Pa_i) = sum_{j=1}^{q_i} [
        log Gamma(alpha_ij) - log Gamma(alpha_ij + N_ij)
        + sum_{k=1}^{r_i} [
            log Gamma(alpha_ijk + N_ijk) - log Gamma(alpha_ijk)
        ]
    ]

where
    alpha_ij  = ESS / q_i          (prior count per parent config)
    alpha_ijk = ESS / (q_i * r_i)  (prior count per cell)
    N_ij      = sum_k N_ijk        (total count for parent config j)
    N_ijk     = count(X_i = k, Pa_i = j)

References
----------
.. [1] Heckerman D, Geiger D, Chickering D. Learning Bayesian networks:
       The combination of knowledge and statistical data. MLJ 20, 1995.
.. [2] Buntine W. Theory refinement on Bayesian networks.  UAI 1991.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discretize_column(
    col: NDArray, n_bins: int = 5, method: str = "quantile"
) -> NDArray:
    """Discretize a continuous column into integer bins.

    Parameters
    ----------
    col : NDArray, shape (n,)
    n_bins : int
    method : str
        ``'quantile'`` for equal-frequency, ``'uniform'`` for equal-width.

    Returns
    -------
    NDArray of int, shape (n,)
    """
    if method == "quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(col, quantiles))
    else:
        edges = np.linspace(col.min(), col.max(), n_bins + 1)
    # np.searchsorted returns bin index; clip to [0, n_bins - 1]
    binned = np.searchsorted(edges[1:-1], col, side="right")
    return binned.astype(int)


def _is_continuous(col: NDArray, unique_ratio: float = 0.5) -> bool:
    """Heuristic: treat column as continuous if > unique_ratio of values are unique."""
    n_unique = len(np.unique(col))
    return n_unique / max(len(col), 1) > unique_ratio


def _parent_config_index(
    data: NDArray,
    parents: List[int],
    arities: NDArray,
) -> NDArray:
    """Map each row to a flat parent-configuration index.

    For parent variables with arities [r_{p1}, r_{p2}, ...], the
    configuration index is computed via mixed-radix encoding:

        idx = x_{p1} + r_{p1} * (x_{p2} + r_{p2} * ( ... ))

    Parameters
    ----------
    data : NDArray, shape (n, p)
    parents : list of int
    arities : NDArray, shape (p_total,)

    Returns
    -------
    NDArray of int, shape (n,)
    """
    if len(parents) == 0:
        return np.zeros(data.shape[0], dtype=int)
    idx = np.zeros(data.shape[0], dtype=int)
    multiplier = 1
    for pa in reversed(parents):
        idx += data[:, pa].astype(int) * multiplier
        multiplier *= int(arities[pa])
    return idx


# ---------------------------------------------------------------------------
# BDeuScore
# ---------------------------------------------------------------------------

class BDeuScore:
    """BDeu score for discrete (categorical) data.

    Parameters
    ----------
    data : NDArray
        Discrete observation matrix ``(n_samples, n_variables)`` with
        integer-coded categories starting from 0.
    equivalent_sample_size : float
        Equivalent sample size (ESS) for the symmetric Dirichlet prior.
        Common choices: 1.0 (Jeffreys-like), 10.0 (more informative).
    max_parents : Optional[int]
        Hard upper bound on the number of parents per node.
    auto_discretize : bool
        If *True*, continuous-looking columns are automatically binned.
    n_bins : int
        Number of bins when auto-discretizing.
    """

    def __init__(
        self,
        data: NDArray,
        equivalent_sample_size: float = 1.0,
        max_parents: Optional[int] = None,
        auto_discretize: bool = True,
        n_bins: int = 5,
    ) -> None:
        raw = np.asarray(data)
        if auto_discretize:
            raw = self._discretize_if_needed(raw, n_bins)
        self.data = raw.astype(int)
        self.n_samples, self.n_variables = self.data.shape
        self.equivalent_sample_size = float(equivalent_sample_size)
        self.max_parents = max_parents

        # Compute arities (number of unique values per variable)
        self.arities = np.array(
            [int(self.data[:, j].max() - self.data[:, j].min() + 1)
             for j in range(self.n_variables)],
            dtype=int,
        )
        # Shift data so minimum value per column is 0
        self._col_mins = self.data.min(axis=0)
        self.data = self.data - self._col_mins

    # ---- Discretization ------------------------------------------------

    @staticmethod
    def _discretize_if_needed(data: NDArray, n_bins: int = 5) -> NDArray:
        """Auto-discretize columns that appear continuous."""
        out = data.copy()
        for j in range(data.shape[1]):
            col = data[:, j]
            if _is_continuous(col):
                out[:, j] = _discretize_column(col, n_bins=n_bins)
        return out

    # ---- Sufficient statistics -----------------------------------------

    def sufficient_statistics(
        self, node: int, parents: Sequence[int]
    ) -> Dict[str, Any]:
        """Compute contingency counts N_{ijk} for *node* given *parents*.

        Returns
        -------
        dict with keys:
            'counts': NDArray of shape (q_i, r_i) — N_{ijk} counts
            'parent_counts': NDArray of shape (q_i,) — N_{ij} marginals
            'r_i': int — arity of node
            'q_i': int — number of parent configurations
        """
        parents = list(parents)
        r_i = int(self.arities[node])
        if len(parents) == 0:
            q_i = 1
        else:
            q_i = int(np.prod(self.arities[parents]))

        pa_idx = _parent_config_index(self.data, parents, self.arities)
        node_vals = self.data[:, node].astype(int)

        counts = np.zeros((q_i, r_i), dtype=int)
        # Vectorised counting
        np.add.at(counts, (pa_idx, node_vals), 1)

        parent_counts = counts.sum(axis=1)  # N_ij
        return {
            "counts": counts,
            "parent_counts": parent_counts,
            "r_i": r_i,
            "q_i": q_i,
        }

    # ---- BDeu computation ----------------------------------------------

    def local_score(self, node: int, parents: Sequence[int]) -> float:
        """Return the local BDeu score for *node* given *parents*."""
        parents = list(parents)
        self._validate(node, parents)
        if self.max_parents is not None and len(parents) > self.max_parents:
            return -math.inf
        return self.log_bdeu_term(node, parents)

    def log_bdeu_term(self, node: int, parents: Sequence[int]) -> float:
        """Compute the log BDeu term for family (node, parents).

        BDeu(i, Pa_i) = sum_j [
            gammaln(alpha_ij) - gammaln(alpha_ij + N_ij)
            + sum_k [gammaln(alpha_ijk + N_ijk) - gammaln(alpha_ijk)]
        ]
        """
        parents = list(parents)
        stats = self.sufficient_statistics(node, parents)
        return self._log_bdeu(
            stats["counts"],
            self.equivalent_sample_size,
            stats["r_i"],
            stats["q_i"],
        )

    @staticmethod
    def _log_bdeu(
        counts: NDArray,
        ess: float,
        r_i: int,
        q_i: int,
    ) -> float:
        """Core BDeu formula.

        Parameters
        ----------
        counts : NDArray, shape (q_i, r_i)
            N_{ijk} counts.
        ess : float
            Equivalent sample size.
        r_i : int
            Number of states of the child node.
        q_i : int
            Number of parent configurations.

        Returns
        -------
        float
        """
        alpha_ij = ess / max(q_i, 1)
        alpha_ijk = ess / max(q_i * r_i, 1)

        N_ij = counts.sum(axis=1)  # shape (q_i,)

        score = 0.0
        # Vectorised computation across parent configurations
        score += np.sum(gammaln(alpha_ij) - gammaln(alpha_ij + N_ij))
        score += np.sum(gammaln(alpha_ijk + counts) - gammaln(alpha_ijk))

        return float(score)

    # ---- Full DAG score ------------------------------------------------

    def score_dag(self, adj_matrix: NDArray) -> float:
        """Return the total BDeu score of a DAG.

        ``adj_matrix[i, j] != 0`` means i -> j.
        """
        adj = np.asarray(adj_matrix)
        total = 0.0
        for j in range(self.n_variables):
            parents = list(np.nonzero(adj[:, j])[0])
            total += self.local_score(j, parents)
        return total

    # ---- Score-based utilities ------------------------------------------

    def score_edge_addition(
        self, node: int, current_parents: Sequence[int], new_parent: int
    ) -> float:
        """Return the score change when adding *new_parent*."""
        old = self.local_score(node, current_parents)
        new = self.local_score(node, list(current_parents) + [new_parent])
        return new - old

    def score_edge_removal(
        self, node: int, current_parents: Sequence[int], removed: int
    ) -> float:
        """Return the score change when removing *removed*."""
        old = self.local_score(node, current_parents)
        new = self.local_score(node, [p for p in current_parents if p != removed])
        return new - old

    def select_best_parents(
        self, node: int, max_parents: Optional[int] = None
    ) -> Tuple[List[int], float]:
        """Exhaustively search for the best parent set for *node*.

        Parameters
        ----------
        node : int
        max_parents : int or None
            Maximum parent-set size to consider.  Defaults to
            ``self.max_parents`` or ``min(n_variables - 1, 4)``.

        Returns
        -------
        best_parents, best_score
        """
        if max_parents is None:
            max_parents = self.max_parents or min(self.n_variables - 1, 4)
        candidates = [v for v in range(self.n_variables) if v != node]
        best_parents: List[int] = []
        best_score = self.local_score(node, [])

        for size in range(1, min(max_parents, len(candidates)) + 1):
            for combo in combinations(candidates, size):
                s = self.local_score(node, list(combo))
                if s > best_score:
                    best_score = s
                    best_parents = list(combo)
        return best_parents, best_score

    # ---- Missing data handling -----------------------------------------

    def local_score_missing(
        self, node: int, parents: Sequence[int], missing_indicator: int = -1
    ) -> float:
        """BDeu score with rows containing missing values excluded.

        Rows where *node* or any parent has value ``missing_indicator``
        are dropped before computing the score.
        """
        parents = list(parents)
        family = [node] + parents
        mask = np.ones(self.n_samples, dtype=bool)
        for v in family:
            mask &= self.data[:, v] != missing_indicator
        if mask.sum() == 0:
            return -math.inf

        sub_data = self.data[mask]
        # Recompute counts on the subset
        r_i = int(self.arities[node])
        q_i = int(np.prod(self.arities[parents])) if parents else 1
        pa_idx = _parent_config_index(sub_data, parents, self.arities)
        node_vals = sub_data[:, node].astype(int)
        counts = np.zeros((q_i, r_i), dtype=int)
        np.add.at(counts, (pa_idx, node_vals), 1)
        return self._log_bdeu(counts, self.equivalent_sample_size, r_i, q_i)

    # ---- Sparse parent configs -----------------------------------------

    def sufficient_statistics_sparse(
        self, node: int, parents: Sequence[int]
    ) -> Dict[str, Any]:
        """Sparse contingency counts — only observed parent configs.

        Useful when q_i is very large but most configs are unobserved.

        Returns
        -------
        dict with keys:
            'counts_dict': dict mapping parent_config -> NDArray of shape (r_i,)
            'r_i', 'q_i': arities
        """
        parents = list(parents)
        r_i = int(self.arities[node])
        q_i = int(np.prod(self.arities[parents])) if parents else 1

        pa_idx = _parent_config_index(self.data, parents, self.arities)
        node_vals = self.data[:, node].astype(int)

        counts_dict: Dict[int, NDArray] = defaultdict(
            lambda: np.zeros(r_i, dtype=int)
        )
        for pi, nv in zip(pa_idx, node_vals):
            counts_dict[int(pi)][nv] += 1

        return {"counts_dict": dict(counts_dict), "r_i": r_i, "q_i": q_i}

    def local_score_sparse(self, node: int, parents: Sequence[int]) -> float:
        """BDeu using sparse counts — efficient for high-arity parents."""
        parents = list(parents)
        self._validate(node, parents)
        if self.max_parents is not None and len(parents) > self.max_parents:
            return -math.inf

        stats = self.sufficient_statistics_sparse(node, parents)
        r_i = stats["r_i"]
        q_i = stats["q_i"]
        alpha_ij = self.equivalent_sample_size / max(q_i, 1)
        alpha_ijk = self.equivalent_sample_size / max(q_i * r_i, 1)

        score = 0.0
        observed_configs = stats["counts_dict"]
        n_unobserved = q_i - len(observed_configs)

        # Unobserved parent configs contribute:
        # gammaln(alpha_ij) - gammaln(alpha_ij) + r_i * (gammaln(alpha_ijk) - gammaln(alpha_ijk)) = 0
        # So we only need to sum over observed configs.

        for pa_config, cell_counts in observed_configs.items():
            N_ij = int(cell_counts.sum())
            score += gammaln(alpha_ij) - gammaln(alpha_ij + N_ij)
            score += float(np.sum(
                gammaln(alpha_ijk + cell_counts) - gammaln(alpha_ijk)
            ))

        return float(score)

    # ---- Validation ----------------------------------------------------

    def _validate(self, node: int, parents: Sequence[int]) -> None:
        if not 0 <= node < self.n_variables:
            raise ValueError(f"node {node} out of range")
        for p in parents:
            if not 0 <= p < self.n_variables:
                raise ValueError(f"parent {p} out of range")
            if p == node:
                raise ValueError("node cannot be its own parent")

    def __repr__(self) -> str:
        return (
            f"BDeuScore(n={self.n_samples}, p={self.n_variables}, "
            f"ess={self.equivalent_sample_size})"
        )
