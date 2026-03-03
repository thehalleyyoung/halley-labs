"""BIC score extended for mixed observational/interventional data.

Implements the truncated factorization: for each node j, only data from
regimes where j was NOT an intervention target contributes to the local
score.  This correctly accounts for do-calculus semantics in structure
learning from heterogeneous experimental data.
"""
from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np

from causal_qd.scores.score_base import ScoreFunction
from causal_qd.types import AdjacencyMatrix, DataMatrix, QualityScore

_VARIANCE_FLOOR: float = 1e-300


class InterventionalBICScore(ScoreFunction):
    """BIC score extended for mixed observational/interventional data.

    For each node j with parents Pa(j), only uses data from regimes
    where j was NOT an intervention target. This implements the
    truncated factorization for interventional data.

    Parameters
    ----------
    data_regimes : list of (data_matrix, intervention_targets) tuples
        Each *data_matrix* is ``(n_samples, n_vars)``.
        *intervention_targets* is a set of variable indices that were
        intervened on in that regime.  Use an empty set for observational
        data.
    penalty_discount : float, optional
        Multiplicative factor for the BIC penalty term (default ``1.0``).
    """

    def __init__(
        self,
        data_regimes: List[Tuple[np.ndarray, Set[int]]],
        penalty_discount: float = 1.0,
    ) -> None:
        if not data_regimes:
            raise ValueError("data_regimes must be non-empty.")
        self._data_regimes = data_regimes
        self._penalty_discount = penalty_discount

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def score(self, dag: AdjacencyMatrix, data: DataMatrix = None) -> QualityScore:
        """Total score = sum of local scores for each node.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Adjacency matrix of the candidate DAG.
        data : DataMatrix, optional
            Ignored; data is supplied via *data_regimes* at construction.

        Returns
        -------
        QualityScore
            Scalar score (higher is better).
        """
        n = dag.shape[0]
        total = 0.0
        for j in range(n):
            parents = np.where(dag[:, j])[0]
            total += self.local_score(j, parents)
        return total

    def local_score(self, node: int, parents: np.ndarray) -> float:
        """Local BIC score for *node* given *parents*, using only non-intervened data.

        Collects samples from all regimes where *node* was NOT an
        intervention target, fits a linear-Gaussian regression of *node*
        on *parents*, and returns the BIC score.

        Parameters
        ----------
        node : int
            Index of the child variable.
        parents : np.ndarray
            Indices of the parent variables.

        Returns
        -------
        float
            Local BIC score contribution.  Returns ``0.0`` when no
            non-intervened samples are available for *node*.
        """
        parents = np.asarray(parents).ravel()

        # Collect rows from regimes where node was NOT intervened on.
        blocks: list[np.ndarray] = []
        for data_matrix, targets in self._data_regimes:
            if node not in targets:
                blocks.append(data_matrix)

        if not blocks:
            return 0.0

        pooled = np.concatenate(blocks, axis=0)
        m = pooled.shape[0]
        if m == 0:
            return 0.0

        y = pooled[:, node]

        # --- OLS regression -------------------------------------------
        if len(parents) > 0:
            X = np.column_stack([np.ones(m), pooled[:, parents]])
        else:
            X = np.ones((m, 1))

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(X.shape[1])
            coeffs[0] = np.mean(y)

        residuals = y - X @ coeffs
        sigma2 = float(np.mean(residuals ** 2))
        sigma2 = max(sigma2, _VARIANCE_FLOOR)

        # --- Log-likelihood (Gaussian) --------------------------------
        log_likelihood = (
            -0.5 * m * np.log(2.0 * np.pi)
            - 0.5 * m * np.log(sigma2)
            - 0.5 * m
        )

        # --- BIC penalty ---------------------------------------------
        k = len(parents) + 1  # intercept + coefficients
        penalty = -0.5 * self._penalty_discount * k * np.log(m)

        return float(log_likelihood + penalty)

    # ------------------------------------------------------------------ #
    #  Representation
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        n_regimes = len(self._data_regimes)
        return (
            f"InterventionalBICScore(n_regimes={n_regimes}, "
            f"penalty_discount={self._penalty_discount})"
        )
